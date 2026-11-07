"""Pair metrics shared by the scored cells: wins, top-1 gold, and nDCG@5.

A pair is won when the best-ranked gold passage beats the other student's gold, and it counts only when
both are in the pool. replication_gate() checks that the saved Qwen3 paired scores give 0.5000, 0.5104,
0.5000, and 0.5208 under this definition, and seal_gate() stops on any held-out query.
"""

import os
import json
import math
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW_N1 = ROOT / "study/retrieval/results/raw"
SHARED = ROOT / "study/_shared"

PAIRED_REF = {
    "qwen_scores_0.6b_paired_default.jsonl": 0.5,
    "qwen_scores_0.6b_paired_profile.jsonl": 0.5104,
    "qwen_scores_4b_paired_default.jsonl": 0.5,
    "qwen_scores_4b_paired_profile.jsonl": 0.5208,
}


def seal_gate(query_ids) -> dict:
    assignment = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]
    holdout = [q for q in query_ids if assignment.get(q) == "test_holdout"]
    if holdout:
        raise SystemExit(
            f"ABORT SEAL: {len(holdout)} test_holdout query_id read (example {holdout[:3]}"
            f"). A post-hoc arm must not touch the held-out split.")
    per_split = {}
    for q in query_ids:
        s = assignment.get(q) or "tak_terdaftar_paired"
        per_split[s] = per_split.get(s, 0) + 1
    return {"lolos": True, "per_split": per_split}


def _ndcg_at_k(ranking, gold, k=5) -> float:
    gold = set(gold)
    dcg = sum(1.0 / math.log2(i + 2)
              for i, c in enumerate(ranking[:k]) if c in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def paired_metrics(rows) -> dict:
    wins, top1, ndcg, n_total, n_compared = [], [], [], 0, 0
    for r in rows:
        n_total += 1
        gold = set(r["gold"] or [])
        other = set(r.get("counterpart") or [])
        pos = {c: i for i, c in enumerate(r["ranking"])}
        g = min((pos[c] for c in gold if c in pos), default=None)
        o = min((pos[c] for c in other if c in pos), default=None)
        if g is not None and o is not None:
            n_compared += 1
            wins.append(1.0 if g < o else 0.0)
        top1.append(1.0 if r["ranking"] and r["ranking"][0] in gold else 0.0)
        ndcg.append(_ndcg_at_k(r["ranking"], gold))
    return {
        "n_baris": n_total,
        "n_komparabel": n_compared,
        "pairwise": round(sum(wins) / len(wins), 4) if wins else None,
        "top1_gold": round(sum(top1) / len(top1), 4) if top1 else None,
        "ndcg5": round(sum(ndcg) / len(ndcg), 4) if ndcg else None,
    }


def read_ranking(path: Path) -> list:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def replication_gate(verbose=True) -> dict:
    evidence = {}
    for fname, ref in PAIRED_REF.items():
        p = RAW_N1 / fname
        if not p.exists():
            raise SystemExit(f"ABORT: reference file missing {p}")
        rows = read_ranking(p)
        seal_gate([r["query_id"] for r in rows])
        m = paired_metrics(rows)
        if m["pairwise"] != ref:
            raise SystemExit(
                f"ABORT REPLICATION: {fname} pairwise from this module {m['pairwise']}"
                f" != published {ref}"
                f". The metric definition does not match, so the numbers of a new arm "
                f"are not valid until these agree.")
        evidence[fname] = {"pairwise": m["pairwise"], "acuan": ref,
                         "n_komparabel": m["n_komparabel"]}
        if verbose:
            print(f"  [OK] {fname:44} {m['pairwise']:.4f} == {ref} "
                  f"(n_komparabel {m['n_komparabel']})")
    return evidence


if __name__ == "__main__":
    print("PAIRED METRIC REPLICATION GATE")
    b = replication_gate()
    print("PASS: this module's metric definition matches the published numbers.")
    print(json.dumps(b, indent=2))
