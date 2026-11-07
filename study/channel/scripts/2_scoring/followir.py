"""Scores one model on a FollowIR collection (Core17, News21, or Robust04) and reports p-MRR.

FollowIR (Weller et al., NAACL 2025) is a public instruction-following benchmark. p-MRR measures whether
a ranking moves the way a changed instruction asks. The p-MRR code is adapted from MTEB under Apache 2.0
(see the notice above get_rank_from_dict, and NOTICE.md). The collections download from the Hugging Face
Hub.

    python followir.py --dataset Core17 --family qwen3rr --model Qwen/Qwen3-Reranker-0.6B
"""

import os
import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
for _stage in sorted(p for p in (ROOT / "study/channel/scripts").iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))

DATASET = {
    "Core17": "mteb/Core17InstructionRetrieval",
    "News21": "mteb/News21InstructionRetrieval",
    "Robust04": "mteb/Robust04InstructionRetrieval",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# P-MRR metric, adapted from MTEB
# Source: MTEB (https://github.com/embeddings-benchmark/mteb), file
# mteb/_evaluators/retrieval_metrics.py, functions get_rank_from_dict, rank_score, and
# calculate_pmrr, licensed under the Apache License 2.0 (LICENSES/Apache-2.0.txt in this
# package, see NOTICE.md). Changes made here: type hints are removed and local names are
# shortened. get_rank_from_dict returns 0.0 instead of 0 as the score of a document that is
# not ranked. calculate_pmrr computes the same per-query mean of rank_score without pandas,
# skips a missing query without logging a warning, and also returns the per-query values.
def get_rank_from_dict(dict_of_results: dict, doc_id: str):
    sorted_by_score = sorted(dict_of_results.items(),
                             key=lambda x: (x[1], x[0]), reverse=True)
    for i, (id_, score) in enumerate(sorted_by_score):
        if id_ == doc_id:
            return i + 1, score
    return len(sorted_by_score) + 1, 0.0


def rank_score(x: dict) -> float:
    if x["og_rank"] >= x["new_rank"]:
        return ((1 / x["og_rank"]) / (1 / x["new_rank"])) - 1
    return 1 - ((1 / x["new_rank"]) / (1 / x["og_rank"]))


def calculate_pmrr(original_run: dict, new_run: dict, changed_qrels: dict):
    per_qid = {}
    for qid, docs in changed_qrels.items():
        if qid + "-og" not in original_run or qid + "-changed" not in new_run:
            continue
        og, new = original_run[qid + "-og"], new_run[qid + "-changed"]
        scores = []
        for d in docs:
            r_og, s_og = get_rank_from_dict(og, d)
            r_new, s_new = get_rank_from_dict(new, d)
            scores.append(rank_score({"og_rank": r_og, "new_rank": r_new}))
        if scores:
            per_qid[qid] = sum(scores) / len(scores)
    if not per_qid:
        return None, {}
    return sum(per_qid.values()) / len(per_qid), per_qid


def ndcg_at_k(ranking, qrels_q, k=5):
    dcg = sum((qrels_q.get(c, 0)) / math.log2(i + 2)
              for i, c in enumerate(ranking[:k]))
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET), required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--kandidat", type=int, default=1000,
                    help="candidate depth per query; 1000 = the official protocol")
    ap.add_argument("--min-free-mib", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: cap the number of queries. The output name gets the suffix _smoke, so a "
                         "full run later still treats the full table as missing. The p-MRR of a capped run only "
                         "checks that the pipeline works and is not reported.")
    args = ap.parse_args()

    from score_arms import FAMILY, TASK_DEFAULT

    tag = f"{args.dataset}_{args.family}_{Path(args.model).name}"
    if args.limit:
        tag += "_smoke"
    out_run = Q2 / f"raw/followir_run_{tag}.jsonl"
    if out_run.exists() and out_run.stat().st_size > 0:
        n_present = sum(1 for _ in out_run.open())
    else:
        n_present = 0

    from datasets import load_dataset
    nm = DATASET[args.dataset]
    print(f"loading {nm} ...", flush=True)
    queries = list(load_dataset(nm, "queries", split="test"))
    if args.limit:
        queries = queries[:args.limit]
    corpus = {r["id"]: ((r.get("title") or "") + "\n" + (r.get("text") or "")).strip()
              for r in load_dataset(nm, "corpus", split="test")}
    top = {r["query-id"]: r["corpus-ids"]
           for r in load_dataset(nm, "top_ranked", split="test")}
    qrels = {}
    for r in load_dataset(nm, "qrels", split="test"):
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    diff = {r["query-id"]: r["corpus-ids"]
            for r in load_dataset(nm, "qrel_diff", split="qrel_diff")}
    print(f"  queries {len(queries)}  corpus {len(corpus)}  changed topics {len(diff)}")

    free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).split()[0])
    if free < args.min_free_mib + 2000:
        raise SystemExit(f"ABORT VRAM: free {free} MiB")

    key = open(Q2 / "raw/.gpu.lock", "w")
    print("waiting for the GPU lock ...", flush=True)
    fcntl.flock(key, fcntl.LOCK_EX)
    print("GPU lock acquired", flush=True)

    import torch
    print(f"loading {args.model} ({args.dtype}) ...", flush=True)
    scorer = FAMILY[args.family](args.model, args.dtype)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    mode = "a" if n_present else "w"
    with out_run.open(mode) as f:
        for i, q in enumerate(queries):
            if i < n_present:
                continue
            candidates = top.get(q["id"], [])[:args.kandidat]
            doc = [corpus.get(c, "")[:4000] for c in candidates]
            inst = q.get("instruction") or TASK_DEFAULT
            if args.family == "qwen3rr":
                scorer.instruction = inst
                query = q["text"]
            else:
                query = f"{inst}\n\nQuery: {q['text']}"
            torch.cuda.empty_cache()
            free_now = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True).split()[0])
            if free_now < args.min_free_mib:
                raise SystemExit(f"ABORT VRAM WHILE RUNNING: {free_now} MiB "
                                 f"at query {i+1}")
            t1 = time.time()
            scores = scorer.score(query, doc, args.batch, args.max_len)
            f.write(json.dumps({
                "query_id": q["id"],
                "scores": {c: round(s, 6) for c, s in zip(candidates, scores)},
                "latency_s": round(time.time() - t1, 3)}) + "\n")
            f.flush()
            torch.cuda.empty_cache()
            print(f"  {i+1}/{len(queries)} {time.time()-t0:.0f}s "
                  f"vram={torch.cuda.max_memory_allocated()/2**20:.0f}MiB",
                  flush=True)

    runs = {}
    for l in out_run.open():
        r = json.loads(l)
        runs[r["query_id"]] = r["scores"]
    og_run = {k: v for k, v in runs.items() if k.endswith("-og")}
    ch_run = {k: v for k, v in runs.items() if k.endswith("-changed")}
    pmrr, per_qid = calculate_pmrr(og_run, ch_run, diff)

    nd = {"og": [], "changed": []}
    for qid, sk in runs.items():
        rank = [c for c, _ in sorted(sk.items(), key=lambda x: (x[1], x[0]),
                                     reverse=True)]
        nd["og" if qid.endswith("-og") else "changed"].append(
            ndcg_at_k(rank, qrels.get(qid, {})))
    lock = json.loads((SHARED / "config_lock.json").read_text())
    result = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("External positive control: the FollowIR benchmark (NAACL 2025) tests whether the arms "
                     "follow search instructions at all."),
        "dataset": nm, "model": args.model, "keluarga": args.family,
        "dtype": args.dtype, "kandidat_per_kueri": args.kandidat,
        "n_kueri": len(queries), "n_topik_berubah": len(diff),
        "p_MRR": round(pmrr, 6) if pmrr is not None else None,
        "p_MRR_per_topik": {k: round(v, 6) for k, v in per_qid.items()},
        "ndcg5_og": round(sum(nd["og"]) / len(nd["og"]), 6) if nd["og"] else None,
        "ndcg5_changed": round(sum(nd["changed"]) / len(nd["changed"]), 6)
                          if nd["changed"] else None,
        "implementasi_metrik": ("rank_score and calculate_pmrr adapted from mteb/_evaluators/retrieval_metrics.py "
                                "(Apache 2.0, see NOTICE.md)"),
        "catatan_lisensi": ("the collections come from the MTEB InstructionRetrieval datasets, but their documents "
                            "are TREC and LDC material with their own access terms; the package uses them only to "
                            "compute metrics and does not redistribute them"),
        "durasi_s": round(time.time() - t0, 1),
        "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "label_source": "deterministic (model scores)",
    }
    out = Q2 / f"tables/followir_{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (Q2 / f"raw/followir_{tag}.manifest.json").write_text(json.dumps({
        "run_id": f"followir_{tag}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/2_scoring/followir.py",
        "perintah_persis": " ".join(sys.argv),
        "config_version": lock["config_version"],
        "status_arm": "post hoc, external positive control",
        "output": {"path": f"results/tables/followir_{tag}.json", "sha256": sha(out)},
    }, indent=2, ensure_ascii=False))
    print(f"\np-MRR {result['p_MRR']}  nDCG@5 og {result['ndcg5_og']} "
          f"changed {result['ndcg5_changed']}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
