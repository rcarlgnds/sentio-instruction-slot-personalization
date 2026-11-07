import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ir_measures
from ir_measures import RR, R, nDCG

ROOT = Path(os.environ.get("SENTIO_ROOT") or Path(__file__).resolve().parents[4])
RAW_RET = ROOT / "study/retrieval/results/raw"
RAW_CH = ROOT / "study/channel/results/raw"
OUT = ROOT / "study/channel/results/tables/posthoc_holdout_metrics.json"
MANIFEST = RAW_CH / "posthoc_holdout_metrics.manifest.json"
INPUTS = {
    "C3_0.6b_default": RAW_RET / "qwen_scores_0.6b_holdout_top20_default.jsonl",
    "C3_0.6b_profile": RAW_RET / "qwen_scores_0.6b_holdout_top20_profile.jsonl",
    "C4_4b_default": RAW_RET / "qwen_scores_4b_holdout_top20_default.jsonl",
    "C4_4b_profile": RAW_RET / "qwen_scores_4b_holdout_top20_profile.jsonl",
    "Z_zerank2_default": RAW_CH / "scores_zerank2_zerank-2_holdout_default.jsonl",
    "Z_zerank2_profile": RAW_CH / "scores_zerank2_zerank-2_holdout_profile.jsonl",
}
POOL = RAW_RET / "pools_holdout_top20.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(path: Path) -> dict:
    qrels, run = {}, {}
    for line in path.open():
        row = json.loads(line)
        qid = row["query_id"]
        qrels[qid] = {g: 1 for g in (row["gold"] or [])}
        ranking = row["ranking"]
        run[qid] = {c: float(len(ranking) - i) for i, c in enumerate(ranking)}
    agg = ir_measures.calc_aggregate([nDCG @ 5, RR, R @ 1], qrels, run)
    out = {str(k): round(v, 4) for k, v in agg.items()}
    return {"n": len(qrels), "nDCG@5": out["nDCG@5"], "MRR": out["RR"], "R@1": out["R@1"]}


def main():
    missing = [k for k, p in INPUTS.items() if not p.exists()]
    if missing:
        raise SystemExit(f"ABORT: score files missing: {missing}")
    table = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "populasi": "test_holdout, 283 queries, candidates = the top 20 of the arm A ranking on the holdout",
        "segel_holdout": {
            "dibuka_kedua": True,
            "pembukaan_pertama": "confirmatory run of the pre-registered family (eval_test_holdout_run1)",
            "pembukaan_kedua": "only to score three frozen post hoc arms",
            "yang_berubah_di_antara_keduanya": "no configuration, threshold, or weight",
        },
        "metode": "ir_measures calc_aggregate over the stored rankings, the same convention as "
                  "qwen_rerankers_c3_c4.json",
        "per_arm": {k: metrics(p) for k, p in INPUTS.items()},
        "label_source": "deterministic",
    }
    OUT.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    MANIFEST.write_text(json.dumps({
        "run_id": "posthoc_holdout_metrics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/0_common/posthoc_holdout_metrics.py",
        "sumber": {**{k: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for k, p in INPUTS.items()},
                   "pools": {"path": str(POOL.relative_to(ROOT)), "sha256": sha(POOL)}},
        "output": {"path": str(OUT.relative_to(ROOT)), "sha256": sha(OUT)},
        "segel_holdout_dibuka_kedua": True,
        "label_source": "deterministic",
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(table["per_arm"], indent=1))


if __name__ == "__main__":
    main()
