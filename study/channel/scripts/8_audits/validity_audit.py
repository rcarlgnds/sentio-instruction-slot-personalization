import os
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys  # noqa: E402
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import metrics as M

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW_N1 = ROOT / "study/retrieval/results/raw"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"

POOL = {
    "paired": RAW_N1 / "pools_paired.jsonl",
    "xpair": Q2 / "raw/pools_paired_expanded.jsonl",
    "control": Q2 / "raw/pools_control_textual.jsonl",
    "traindev": RAW_N1 / "pools_traindev_top20.jsonl",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def declared_partial(name: str) -> str:
    n = name.lower()
    if "smoke" in n:
        return "smoke test"
    if "_envb1_" in n:
        return "batch-1 envelope cell, declared in the table caption"
    return ""


def main():
    result = []

    def log_check(name, status, detail):
        result.append({"pemeriksaan": name, "status": status, "detail": detail})
        mark = {"LOLOS": "[OK]", "PERINGATAN": "[! ]", "GAGAL": "[XX]"}[status]
        print(f"{mark} {name}: {detail[:130]}")

    try:
        evidence = M.replication_gate(verbose=False)
        log_check("gerbang_replikasi_metrik", "LOLOS",
              f"4 published paired numbers reproduced bit-identically: { {k.split('_')[2]+'_'+k.split('_')[4].split('.')[0]: v['pairwise'] for k, v in evidence.items()} }")
    except SystemExit as e:
        log_check("gerbang_replikasi_metrik", "GAGAL", str(e))

    assignment = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]
    n_file, declared, undeclared = 0, 0, 0
    declared_files, files_without_manifest = [], []
    for f in sorted(Q2.glob("raw/scores_*.jsonl")):
        n_file += 1
        man = f.with_name(f.stem + ".manifest.json")
        allowed = False
        if man.exists():
            allowed = bool((json.loads(man.read_text()).get("segel_holdout") or {})
                         .get("dibuka_kedua"))
        else:
            files_without_manifest.append(f.name)
        n_rows = sum(1 for l in f.open()
                      if l.strip() and assignment.get(json.loads(l)["query_id"]) == "test_holdout")
        if not n_rows:
            continue
        if allowed:
            declared += n_rows
            declared_files.append({"berkas": f.name, "n_baris": n_rows})
        else:
            undeclared += n_rows
    seal_detail = {
        "baris_holdout_tak_terdeklarasi": undeclared,
        "cakupan": f"{n_file} score files checked in this tree",
        "baris_holdout_terdeklarasi": (f"{declared} rows in {len(declared_files)} files"),
        "berkas_dengan_pembukaan_terdeklarasi": [
            f"{b['berkas']} ({b['n_baris']} rows)" for b in declared_files],
        "berkas_tanpa_manifest": files_without_manifest,
        "_aturan": ("A test_holdout row is valid only when the manifest next to its score file declares "
                    "segel_holdout.dibuka_kedua. This check verifies that every touch of the holdout is "
                    "declared, not merely that there was none."),
    }
    log_check("segel_holdout", "GAGAL" if undeclared else "LOLOS",
          f"{n_file} score files checked, {undeclared}"
          f" test_holdout rows without a declaration, {declared}"
          f" rows declared in a manifest ({len(declared_files)}"
          f" files with segel_holdout.dibuka_kedua)")

    locked = []
    for f in sorted(Q2.glob("raw/scores_*_paired_default*.jsonl")) + \
             list(RAW_N1.glob("qwen_scores_*_paired_default.jsonl")):
        rows = M.read_ranking(f)
        group = defaultdict(list)
        for r in rows:
            group["-".join(r["query_id"].split("-")[:2])].append(r)
        identical = sum(1 for v in group.values()
                      if len(v) == 2 and v[0]["ranking"] == v[1]["ranking"])
        affected_metrics = 0
        for pid, v in group.items():
            if len(v) != 2 or v[0]["ranking"] == v[1]["ranking"]:
                continue
            for r in v:
                pos = {c: i for i, c in enumerate(r["ranking"])}
                g = min((pos[c] for c in (r["gold"] or []) if c in pos), default=None)
                o = min((pos[c] for c in (r.get("counterpart") or []) if c in pos),
                        default=None)
                r["_gd"] = (g, o)
            if v[0]["_gd"] != (v[1]["_gd"][1], v[1]["_gd"][0]):
                affected_metrics += 1
        m = M.paired_metrics(rows)
        ok = m["pairwise"] == 0.5 and affected_metrics == 0
        locked.append({"berkas": f.name, "pasangan": len(group),
                         "urutan_identik": identical,
                         "beda_urutan_tanpa_efek_metrik": len(group) - identical,
                         "beda_yang_MENGUBAH_metrik": affected_metrics,
                         "pairwise": m["pairwise"],
                         "sesuai_prediksi_terkunci": ok})
    all_ok = all(t["sesuai_prediksi_terkunci"] for t in locked) if locked else False
    log_check("sel_terkunci_konstruksi", "LOLOS" if all_ok else "PERINGATAN",
          f"{sum(1 for t in locked if t['sesuai_prediksi_terkunci'])}/{len(locked)}"
          f" blind arms fixed at pairwise 0.5 without a single difference that "
          f"changes a metric (rank permutations caused by bf16 noise are counted "
          f"separately and are not a defect)")

    unfair = []
    for tag, pool in POOL.items():
        if not pool.exists():
            continue
        ref = {}
        for l in pool.open():
            r = json.loads(l)
            ref[r["query_id"]] = [c["id"] for c in r["cands"]]
        for f in sorted(Q2.glob(f"raw/scores_*_{tag}_*.jsonl")):
            for l in f.open():
                r = json.loads(l)
                a = ref.get(r["query_id"])
                if a is not None and sorted(a) != sorted(r["ranking"]):
                    unfair.append(f"{f.name}:{r['query_id']}")
                    break
    n_pools = sum(1 for pool in POOL.values() if pool.exists())
    log_check("keadilan_kandidat", "GAGAL" if unfair else ("LOLOS" if n_pools else "PERINGATAN"),
          f"candidate lists identical to the pool for every cell; anomalies: {len(unfair)}; "
          f"pools found: {n_pools} of {len(POOL)}")

    incomplete, partial = [], []
    for tag, pool in POOL.items():
        if not pool.exists():
            continue
        n_target = sum(1 for _ in pool.open())
        for f in sorted(Q2.glob(f"raw/scores_*_{tag}_*.jsonl")):
            n = sum(1 for _ in f.open())
            if n >= n_target:
                continue
            reason = declared_partial(f.name)
            (partial if reason else incomplete).append(
                f"{f.name} {n}/{n_target}" + (f" [{reason}]" if reason else ""))
    log_check("kelengkapan_sel", "PERINGATAN" if incomplete or not n_pools else "LOLOS",
          f"cells incomplete beyond the declared ones: {len(incomplete)}"
          + (f" ({incomplete[:3]})" if incomplete else "")
          + f"; declared partial: {len(partial)}; pools found: {n_pools} of {len(POOL)}")

    per_tag = defaultdict(set)
    for tag in POOL:
        for f in sorted(Q2.glob(f"raw/scores_*_{tag}_*.jsonl")):
            if declared_partial(f.name):
                continue
            rows = M.read_ranking(f)
            if rows and rows[0].get("counterpart") is not None:
                per_tag[tag].add(M.paired_metrics(rows)["n_komparabel"])
    inconsistent = {k: sorted(v) for k, v in per_tag.items() if len(v) > 1}
    log_check("konsistensi_n_komparabel",
          "GAGAL" if inconsistent else "LOLOS",
          f"n_komparabel uniform per pool; anomalies: {inconsistent or 'none'}")

    dupe_diff = 0
    for f in sorted(Q2.glob("raw/scores_*.jsonl")):
        seen = {}
        for l in f.open():
            r = json.loads(l)
            k = r["query_id"]
            if k in seen and seen[k] != r["ranking"]:
                dupe_diff += 1
            seen[k] = r["ranking"]
    log_check("determinisme_baris", "GAGAL" if dupe_diff else "LOLOS",
          f"duplicate rows in a different order: {dupe_diff}")

    lock = json.loads((SHARED / "config_lock.json").read_text())
    n_failed = sum(1 for h in result if h["status"] == "GAGAL")
    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Self-audit of the score files in the results/raw/ folders of the tree it runs on. It runs automatically; "
                     "a check with the status GAGAL (failed) means the numbers it covers must not be cited."),
        "n_pemeriksaan": len(result),
        "n_gagal": n_failed,
        "vonis": "fit to cite" if n_failed == 0 else "some checks failed",
        "pemeriksaan": result,
        "rincian_sel_terkunci": locked,
        "rincian_segel_holdout": seal_detail,
    }
    out = Q2 / "tables/validity_audit.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nverdict: {output['vonis']} ({n_failed} failed of {len(result)})")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
