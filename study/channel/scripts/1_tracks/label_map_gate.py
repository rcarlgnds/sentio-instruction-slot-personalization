import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))

from label_map import (CORPUS, load_id_label_map,  # noqa: E402
                       load_text_label_map, source_summary)

ROOT = Path(__file__).resolve().parents[4]
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
POOLS = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
POOLS_X = Q2 / "raw/pools_paired_expanded.jsonl"
TAB = Q2 / "tables"
VENV = ROOT / "system/.venv/bin/python"
OVERLAY = [ROOT / "system/.scorer-libs", ROOT / "system/.st-libs"]

EXPECTED_GOLD_DIFF = {"paired": 21, "broad": 30}
EXPECTED_LIBS = {"transformers": "4.55.0", "sentence_transformers": "6.0.1"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def map_fingerprint(mapping: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(mapping):
        m = mapping[k] or {}
        h.update(f"{k}|{m.get('department')}|{m.get('semester')}\n".encode())
    return h.hexdigest()[:16]


def read_pool(p: Path) -> list:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def check_pool(name, rows, id_map, text_map, unique_prefix):
    n_cand = n_no_id = n_no_text = n_conflicts = 0
    gold_diff, changed = [], []
    for r in rows:
        gold = set(r.get("gold") or [])
        for c in r["cands"]:
            n_cand += 1
            text = (c.get("text") or "")
            key = text[:180].strip()
            mi = id_map.get(c["id"])
            mt = text_map.get(key)
            if mi is None:
                n_no_id += 1
            if mt is None:
                n_no_text += 1
            if mi is None or mt is None:
                continue
            di, dt = mi.get("department"), mt.get("department")
            si, st = mi.get("semester"), mt.get("semester")
            if di == dt and si == st:
                continue
            if key in unique_prefix:
                n_conflicts += 1
            rec = {"query_id": r.get("query_id"), "cand_id": c["id"],
                   "gold": c["id"] in gold,
                   "dept_peta_teks": dt, "dept_peta_id": di,
                   "semester_peta_teks": st, "semester_peta_id": si,
                   "prefiks_unik_di_korpus": key in unique_prefix}
            changed.append(rec)
            if c["id"] in gold:
                gold_diff.append(rec)
    return {"n_kandidat": n_cand, "n_tak_terpetakan_id": n_no_id,
            "n_tak_terpetakan_teks": n_no_text,
            "n_label_berubah": len(changed), "n_gold_berubah": len(gold_diff),
            "n_konflik_prefiks_unik": n_conflicts}, changed


def seal_gate(pool):
    assignment = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]
    leak = []
    for name, rows in pool.items():
        for r in rows:
            if assignment.get(r.get("query_id")) == "test_holdout":
                leak.append(f"{name}:{r.get('query_id')}")
    return leak


def check_overlay():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in OVERLAY]
                                        + [str(Path(__file__).resolve().parent)])
    code = ("import json, transformers, sentence_transformers as st;"
            "print(json.dumps({'transformers': transformers.__version__,"
            "'transformers_path': transformers.__file__,"
            "'sentence_transformers': st.__version__,"
            "'sentence_transformers_path': st.__file__}))")
    try:
        out = subprocess.run([str(VENV), "-c", code], capture_output=True,
                             text=True, env=env, timeout=300, cwd=str(ROOT / "system"))
    except Exception as e:
        return {"lolos": False, "galat": f"subprocess gagal: {e}"}
    if out.returncode != 0:
        return {"lolos": False, "galat": out.stderr.strip()[-400:]}
    try:
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"lolos": False, "galat": f"unreadable output: {out.stdout[-300:]}"}
    wrong = [k for k, v in EXPECTED_LIBS.items() if info.get(k) != v]
    from_overlay = all(any(str(p) in (info.get(k + "_path") or "") for p in OVERLAY)
                       for k in EXPECTED_LIBS)
    info["lolos"] = not wrong and from_overlay
    if wrong:
        info["galat"] = f"version does not match: {wrong}"
    elif not from_overlay:
        info["galat"] = "imported library not from the overlay"
    return info


def free_vram():
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=memory.total,memory.used,memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        total, use, free = (int(x) for x in out.stdout.strip().split("\n")[0].split(", "))
        return {"total_mib": total, "terpakai_mib": use, "bebas_mib": free}
    except Exception as e:
        return {"galat": str(e)}


def main() -> int:
    t0 = datetime.now(timezone.utc)
    id_map = load_id_label_map()
    text_map = load_text_label_map()
    fingerprint_a = map_fingerprint(id_map)
    id_map_reloaded = load_id_label_map()
    fingerprint_b = map_fingerprint(id_map_reloaded)

    prefix_counts = {}
    for rows in CORPUS.open(encoding="utf-8"):
        k = (json.loads(rows).get("text") or "")[:180].strip()
        if k:
            prefix_counts[k] = prefix_counts.get(k, 0) + 1
    unique_prefix = {k for k, n in prefix_counts.items() if n == 1}

    pool = {"paired": read_pool(POOLS), "broad": read_pool(POOLS_X)}
    summary, changes = {}, {}
    for name, rows in pool.items():
        summary[name], changes[name] = check_pool(
            name, rows, id_map, text_map, unique_prefix)
        summary[name]["n_baris"] = len(rows)

    leak = seal_gate(pool)
    overlay = check_overlay()
    sha_in = {"corpus.jsonl": sha(CORPUS), "pools_paired.jsonl": sha(POOLS),
              "pools_paired_expanded.jsonl": sha(POOLS_X)}

    result = []
    for name in ("paired", "broad"):
        r = summary[name]
        result.append((f"G1 id coverage {name}", r["n_tak_terpetakan_id"] == 0,
                      f"{r['n_kandidat'] - r['n_tak_terpetakan_id']}/{r['n_kandidat']} mapped"))
        result.append((f"G2 unique-prefix conflicts {name}", r["n_konflik_prefiks_unik"] == 0,
                      f"{r['n_konflik_prefiks_unik']} conflicts"))
        result.append((f"G3 reproduction of the gold defect {name}",
                      r["n_gold_berubah"] == EXPECTED_GOLD_DIFF[name],
                      f"{r['n_gold_berubah']} golds changed, expected {EXPECTED_GOLD_DIFF[name]}"))
    result.append(("G4 holdout seal", not leak,
                  "no holdout query_id" if not leak else f"LEAK {leak[:3]}"))
    result.append(("G6 library overlay", bool(overlay.get("lolos")),
                  overlay.get("galat") or f"transformers {overlay.get('transformers')} "
                                          f"st {overlay.get('sentence_transformers')}"))
    result.append(("G7 map determinism", fingerprint_a == fingerprint_b, f"{fingerprint_a} vs {fingerprint_b}"))

    failed = [n for n, ok, _ in result if not ok]
    report = {
        "dibuat": t0.isoformat(),
        "peta": source_summary(),
        "sidik_peta_id": fingerprint_a,
        "n_kunci_peta_id": len(id_map),
        "n_kunci_peta_teks": len(text_map),
        "n_chunk_korpus": sum(prefix_counts.values()),
        "n_prefiks_unik": len(unique_prefix),
        "per_kolam": summary,
        "segel_bocor": leak,
        "overlay": overlay,
        "sha256_masukan": sha_in,
        "vram": free_vram(),
        "pemeriksaan": [{"nama": n, "lolos": ok, "catatan": c} for n, ok, c in result],
        "vonis": "LOLOS" if not failed else "GAGAL",
        "gagal": failed,
    }
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "label_map_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (TAB / "label_change_report.json").write_text(json.dumps(
        {"dibuat": t0.isoformat(), "peta": source_summary(),
         "definisi": ("every pool candidate whose department or semester differs between the 180-character "
                      "text-prefix map and the candidate-identity map"),
         "ringkas": summary, "per_kolam": changes},
        indent=1, ensure_ascii=False) + "\n")

    width = max(len(n) for n, _, _ in result)
    for n, ok, c in result:
        print(f"  {'PASS' if ok else 'FAIL'}  {n:{width}}  {c}")
    v = report["vram"]
    print(f"  INFO  {'G8 vram':{width}}  {v.get('bebas_mib')} MiB free "
          f"of {v.get('total_mib')} MiB")
    print(f"\nverdict {report['vonis']}")
    print(f"wrote {TAB / 'label_map_gate.json'}")
    print(f"wrote {TAB / 'label_change_report.json'}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
