import os
import argparse
import collections
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from inference_ladder import wins_per_row  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
TABLE = ROOT / "study/channel/results/tables/cluster_bootstrap.json"
MANIFEST = RAW / "cluster_bootstrap.manifest.json"
SEED = 42

MODEL = [
    ("Qwen3-Reranker-0.6B",
     "scores_qwen3rr_Qwen3-Reranker-0.6B_xpair_default.jsonl",
     "scores_qwen3rr_Qwen3-Reranker-0.6B_xpair_profile.jsonl"),
    ("Qwen3-Reranker-4B",
     "scores_qwen3rr_Qwen3-Reranker-4B_xpair_default.jsonl",
     "scores_qwen3rr_Qwen3-Reranker-4B_xpair_profile.jsonl"),
    ("zerank-2",
     "scores_zerank2official_zerank-2_xpair_default.jsonl",
     "scores_zerank2official_zerank-2_xpair_profile.jsonl"),
    ("FollowIR-7B",
     "scores_followir_FollowIR-7B_xpair_none.jsonl",
     "scores_followir_FollowIR-7B_xpair_slot.jsonl"),
    ("ctxl-6B",
     "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_xpair_none.jsonl",
     "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_xpair_slot.jsonl"),
    ("bge-reranker-v2-gemma",
     "scores_gemma_bge-reranker-v2-gemma_xpair_none.jsonl",
     "scores_gemma_bge-reranker-v2-gemma_xpair_slot.jsonl"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def group_of(qid: str) -> str:
    return qid.rsplit("-", 1)[0]


def icc_one_way(cluster: list) -> float:
    kl = [v for v in cluster if len(v) > 1]
    m = statistics.mean(len(v) for v in kl)
    mean = statistics.mean(x for v in kl for x in v)
    msb = sum(len(v) * (statistics.mean(v) - mean) ** 2 for v in kl) / (len(kl) - 1)
    msw = sum((x - statistics.mean(v)) ** 2 for v in kl for x in v) / (
        sum(len(v) for v in kl) - len(kl))
    return (msb - msw) / (msb + (m - 1) * msw)


def bootstrap(group_key: list, delta: np.ndarray, n_rows: np.ndarray,
              b: int, rng) -> tuple:
    idx = rng.integers(0, len(delta), size=(b, len(delta)))
    d = delta[idx].sum(axis=1) / n_rows[idx].sum(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", type=int, default=10_000)
    args = ap.parse_args()

    missing = [f for _, a, b in MODEL for f in (a, b) if not (RAW / f).exists()]
    if missing:
        raise SystemExit("ABORT: cells missing, table not written:\n  " + "\n  ".join(missing))

    rng = np.random.default_rng(SEED)
    board = {}
    for name, f_blind, f_slot in MODEL:
        W = wins_per_row(RAW / f_blind)
        S = wins_per_row(RAW / f_slot)
        rows = sorted(set(W) & set(S))
        per_group = collections.defaultdict(lambda: [0, 0])
        for q in rows:
            g = per_group[group_of(q)]
            g[0] += S[q] - W[q]
            g[1] += 1
        key = sorted(per_group)
        delta = np.array([per_group[k][0] for k in key], dtype=np.int32)
        n_rows = np.array([per_group[k][1] for k in key], dtype=np.int32)
        d = float(delta.sum()) / int(n_rows.sum())

        lo_g, hi_g = bootstrap(key, delta, n_rows, args.resample, rng)
        s_rows = np.array([S[q] - W[q] for q in rows], dtype=np.int32)
        ones = np.ones(len(rows), dtype=np.int32)
        lo_b, hi_b = bootstrap(rows, s_rows, ones, args.resample, rng)

        cluster = [[S[q] for q in rows if group_of(q) == k] for k in key]
        icc = icc_one_way(cluster)
        m = float(n_rows.mean())
        board[name] = {
            "n_baris": int(n_rows.sum()),
            "n_grup": len(key),
            "baris_per_grup_rata": round(m, 3),
            "selisih_berpasangan": round(d, 6),
            "selisih_poin_persen": round(d * 100, 3),
            "selang_klaster_95": [round(lo_g, 6), round(hi_g, 6)],
            "selang_baris_95": [round(lo_b, 6), round(hi_b, 6)],
            "batas_klaster_poin_persen": round(max(abs(lo_g), abs(hi_g)) * 100, 3),
            "batas_baris_poin_persen": round(max(abs(lo_b), abs(hi_b)) * 100, 3),
            "lebar_klaster_dibagi_lebar_baris": round(
                (hi_g - lo_g) / (hi_b - lo_b), 4) if hi_b > lo_b else None,
            "icc_satu_arah_tanpa_dibatasi": round(icc, 6),
            "design_effect": round(1 + (m - 1) * icc, 4),
            "berkas": {"buta": f_blind, "slot": f_slot},
            "sha256": {"buta": sha(RAW / f_blind), "slot": sha(RAW / f_slot)},
        }

    wider = [name for name, h in board.items()
                   if h["lebar_klaster_dibagi_lebar_baris"] and
                   h["lebar_klaster_dibagi_lebar_baris"] > 1.0]
    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("Group bootstrap of the instruction-slot effect on the broad track, so the 1,621 rows "
                     "from 600 groups are not treated as 1,621 independent observations."),
        "resample": args.resample,
        "seed": SEED,
        "metode": ("resample the 600 groups with replacement, recompute the paired difference, and take the "
                   "2.5 and 97.5 percentiles. The row comparison uses the same procedure with each row as "
                   "its own cluster, so only the resampling unit differs."),
        "kenapa_icc_negatif": ("A scorer that ignores the profile returns one ranking for all rows of a group, so "
                               "exactly one row of that group wins. Rows within a group exclude each other rather than "
                               "reinforce each other, so the intraclass correlation is negative and treating the rows "
                               "as independent is conservative."),
        "model_yang_selangnya_melebar": wider,
        "vonis": ("Grouping does not loosen the bound on this track"
                  if not wider else
                  f"Grouping widens the interval on {len(wider)} model"),
        "label_source": "deterministic (gold matching, no rater)",
        "papan": board,
    }
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "cluster_bootstrap_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/4_slot_vs_baseline/cluster_bootstrap.py",
        "output": {"path": "results/tables/cluster_bootstrap.json", "sha256": sha(TABLE)},
        "resample": args.resample, "seed": SEED,
        "label_source": "deterministic",
    }, indent=1))

    print(f"  {args.resample:,} resample, seed {SEED}, {len(board)} model\n")
    print(f"  {'model':22s} {'d pp':>7} {'row bound':>12} {'cluster bound':>14} "
          f"{'width ratio':>12} {'ICC':>8}")
    print("  " + "-" * 80)
    for name, h in board.items():
        print(f"  {name:22s} {h['selisih_poin_persen']:>7.3f} "
              f"{h['batas_baris_poin_persen']:>12.3f} "
              f"{h['batas_klaster_poin_persen']:>14.3f} "
              f"{h['lebar_klaster_dibagi_lebar_baris']:>12.4f} "
              f"{h['icc_satu_arah_tanpa_dibatasi']:>+8.3f}")
    print(f"\n  {output['vonis']}")
    print("  written:", TABLE)


if __name__ == "__main__":
    main()
