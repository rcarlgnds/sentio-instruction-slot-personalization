import os
import hashlib
import json
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

ARM = {
    "C3_0.6b_default": "qwen_scores_0.6b_paired_default.jsonl",
    "C3_0.6b_profil": "qwen_scores_0.6b_paired_profile.jsonl",
    "C4_4b_default": "qwen_scores_4b_paired_default.jsonl",
    "C4_4b_profil": "qwen_scores_4b_paired_profile.jsonl",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kendall_tau_b(a: list, b: list) -> float:
    pos_b = {x: i for i, x in enumerate(b)}
    order = [pos_b[x] for x in a if x in pos_b]
    n = len(order)
    if n < 2:
        return 1.0
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (order[i] - order[j])
            if d < 0:
                concord += 1
            elif d > 0:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total else 1.0


def pairs_from(rows: list) -> dict:
    group = {}
    for r in rows:
        pid = "-".join(r["query_id"].split("-")[:2])
        group.setdefault(pid, []).append(r)
    return {k: v for k, v in group.items() if len(v) == 2}


def row_wins(r) -> bool | None:
    pos = {c: i for i, c in enumerate(r["ranking"])}
    g = min((pos[c] for c in (r["gold"] or []) if c in pos), default=None)
    o = min((pos[c] for c in (r.get("counterpart") or []) if c in pos), default=None)
    if g is None or o is None:
        return None
    return g < o


def move_dir(rows_default: list, profile_rows: list) -> dict:
    idx = {r["query_id"]: r for r in rows_default}
    gains = drops = fixed = 0
    for r in profile_rows:
        d = idx.get(r["query_id"])
        if d is None:
            continue
        a, b = row_wins(d), row_wins(r)
        if a is None or b is None:
            continue
        if a is False and b is True:
            gains += 1
        elif a is True and b is False:
            drops += 1
        else:
            fixed += 1
    return {"kalah_jadi_menang": gains, "menang_jadi_kalah": drops,
            "tidak_berubah": fixed, "selisih_bersih": gains - drops}


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())
    print("metric replication gate:")
    replication_evidence = M.replication_gate()

    result = {}
    for name, fpath in ARM.items():
        p = RAW_N1 / fpath
        rows = M.read_ranking(p)
        M.seal_gate([r["query_id"] for r in rows])
        group = pairs_from(rows)
        flip_top1 = flip_order = 0
        taus = []
        flip_example = []
        for pid, (r1, r2) in sorted(group.items()):
            a, b = r1["ranking"], r2["ranking"]
            if a[:1] != b[:1]:
                flip_top1 += 1
                if len(flip_example) < 5:
                    flip_example.append({"pair_id": pid,
                                        "top1_profil_1": a[0], "top1_profil_2": b[0]})
            if a != b:
                flip_order += 1
            taus.append(kendall_tau_b(a, b))
        n = len(group)
        m = M.paired_metrics(rows)
        result[name] = {
            "n_pasangan": n,
            "flip_top1": round(flip_top1 / n, 4),
            "flip_urutan_penuh": round(flip_order / n, 4),
            "kendall_tau_mean": round(sum(taus) / len(taus), 6),
            "pairwise": m["pairwise"],
            "n_komparabel": m["n_komparabel"],
            "contoh_flip_top1": flip_example,
            "sumber_sha256": sha(p),
        }
        print(f"  {name:18} flip_top1 {flip_top1}/{n}  flip_urutan {flip_order}/{n}  "
              f"tau {sum(taus)/len(taus):.4f}  pairwise {m['pairwise']}")

    direction = {}
    for family, d_key, p_key in (("C3_0.6b", "C3_0.6b_default", "C3_0.6b_profil"),
                                   ("C4_4b", "C4_4b_default", "C4_4b_profil")):
        rd = M.read_ranking(RAW_N1 / ARM[d_key])
        rp = M.read_ranking(RAW_N1 / ARM[p_key])
        direction[family] = move_dir(rd, rp)
        a = direction[family]
        print(f"  direction {family:10} loss->win {a['kalah_jadi_menang']}  "
              f"win->loss {a['menang_jadi_kalah']}  "
              f"net {a['selisih_bersih']:+d} of {M.paired_metrics(rp)['n_komparabel']}")

    default_flat = all(result[k]["flip_urutan_penuh"] == 0.0
                       for k in result if k.endswith("default"))
    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Flip rate when the profile is swapped on bit-identical queries and pools; it separates "
                     "profile blindness from a lexical key."),
        "protokol": {
            "n_pasangan": 52,
            "kueri_identik_antar_profil": True,
            "pool_identik_antar_profil": True,
            "yang_berbeda": "student_profile and gold only",
            "populasi": "paired track, test_holdout not touched",
        },
        "per_arm": result,
        "arah_gerakan_instruksi": direction,
        "gerbang_replikasi_metrik": replication_evidence,
        "temuan": {
            "arm_default_urutan_identik_antar_profil": default_flat,
            "_arti": (
                "flip_urutan_penuh = 0 in the default variant shows that the text scorers return "
                "bit-identical rankings for the two profiles, so pairwise 0.5 is invariance to the "
                "profile, not a statistical accident and not a lexical key. In the profile-instruction "
                "variant the flips stay small, so the instruction slot moves a few rankings without "
                "recovering the program signal. The arah_gerakan_instruksi block separates this further: "
                "the profile instruction moves decisions on many rows, but losses turned wins and wins "
                "turned losses nearly balance, so the instruction is not ignored, yet its movement is "
                "random with respect to the intended signal."),
        },
        "batasan": (
            "the flip rate of the DITS arm is not computed here, because the per-query rankings of "
            "the production arm were not stored; by design that arm does depend on the profile "
            "(pairwise 0.9688 in the published table)"),
        "label_source": "deterministic (re-reading the frozen score files)",
    }
    out = Q2 / "tables/flip_rate.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/flip_rate.manifest.json").write_text(json.dumps({
        "run_id": "q2_flip_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/3_slot_changes/flip_rate.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {b: sha(RAW_N1 / b) for b in ARM.values()},
        "output": {"path": "results/tables/flip_rate.json", "sha256": sha(out),
                   "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
