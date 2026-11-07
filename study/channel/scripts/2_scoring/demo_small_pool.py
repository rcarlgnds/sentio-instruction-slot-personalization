"""Runs the real scorers on a small open-licensed pool and compares the result with the released scores.

The pool holds ten question pairs of the extended paired track whose gold passages come from
public-domain and CC BY books, scored under the blind, slot, and label setups. It needs a CUDA GPU and
the model weights. With --dry-run it only checks the demo texts against their hashes, with no GPU.

    python demo_small_pool.py --all
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))

from architecture_scorers import NEW_FAMILY  # noqa: E402
from score_arms import (FAMILY, TASK_DEFAULT, profile_instruction, source_label)  # noqa: E402

RAW = ROOT / "study/channel/results/raw_p2"
POOL = RAW / "pools/demo_pool_p2.jsonl"
ANCHORS = RAW / "pools/pools_p2_anchors.jsonl"

MODELS = {
    "Qwen3-0.6B": ("qwen3rr", "Qwen/Qwen3-Reranker-0.6B", "float16", 2048, 8),
    "Qwen3-4B": ("qwen3rr", "Qwen/Qwen3-Reranker-4B", "float16", 2048, 4),
    "zerank-2": ("zerank2official", "zeroentropy/zerank-2", "bfloat16", 1536, 2),
    "FollowIR-7B": ("followir", "jhu-clsp/FollowIR-7B", "bfloat16", 1536, 2),
    "ctxl-6B": ("ctxl", "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b", "bfloat16", 1536, 8),
    "bge-gemma": ("gemma", "BAAI/bge-reranker-v2-gemma", "bfloat16", 1536, 8),
}
CELLS = ("none", "slot", "both")
ARM_CELL = {"none": "default", "slot": "profile", "both": "profile_docmeta"}


def score_file(model: str, cell: str) -> Path:
    family, name = MODELS[model][:2]
    stem = name.split("/")[-1]
    suffix = ARM_CELL[cell] if family in ("qwen3rr", "zerank2official") else cell
    return RAW / f"scores_{family}_{stem}_xp2_{suffix}.jsonl"


def released_scores(model: str, cell: str) -> dict:
    out = {}
    with score_file(model, cell).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                for chunk, score in zip(r["ranking"], r["scores"]):
                    out[(r["query_id"], chunk)] = score
    return out


def read_pool() -> list:
    with POOL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inputs(row: dict, cell: str, family: str):
    instruction = profile_instruction(row["profile"]) if cell in ("slot", "both") else None
    docs = []
    for c in row["cands"]:
        label = source_label(c["label"]) + "\n\n" if cell == "both" else ""
        docs.append(label + c["text"][:4000])
    query = row["query"]
    if family == "zerank2official":
        query = f"{instruction or TASK_DEFAULT}\n\nQuery: {query}"
    return query, instruction, docs


def check_demo_files(rows: list) -> int:
    anchors = {}
    with ANCHORS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                anchors[r["id"]] = r["exact_sha256"]
    bad = 0
    for row in rows:
        for c in row["cands"]:
            if hashlib.sha256(c["text"].encode("utf-8")).hexdigest() != anchors.get(c["id"]):
                bad += 1
    found = 0
    for model in MODELS:
        for cell in CELLS:
            have = released_scores(model, cell)
            found += sum((row["query_id"], c["id"]) in have for row in rows for c in row["cands"])
    want = sum(len(row["cands"]) for row in rows) * len(MODELS) * len(CELLS)
    print(f"demo pool: {len(rows)} rows, {bad} passages differ from their anchors, "
          f"{found} of {want} released scores found")
    return 1 if bad or found != want else 0


def run_model(model: str, rows: list) -> dict:
    import torch
    family, name, dtype, max_len, batch = MODELS[model]
    if family in FAMILY:
        scorer = FAMILY[family](name, dtype)
    else:
        scorer = NEW_FAMILY[family](name, dtype)
    report = {}
    for cell in CELLS:
        have = released_scores(model, cell)
        largest, same_order = 0.0, 0
        for row in rows:
            query, instruction, docs = inputs(row, cell, family)
            if family == "qwen3rr":
                scorer.instruction = instruction or TASK_DEFAULT
            elif family != "zerank2official":
                scorer.set_instruction(instruction)
            new = scorer.score(query, docs, batch, max_len)
            old = [have[(row["query_id"], c["id"])] for c in row["cands"]]
            largest = max(largest, max(abs(round(a, 6) - b) for a, b in zip(new, old)))
            same_order += (new[0] > new[1]) == (old[0] > old[1])
        report[cell] = (largest, same_order)
        print(f"  {model:12s} {cell:5s} largest score gap {largest:.6f}, order kept on {same_order} of {len(rows)} rows",
              flush=True)
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = read_pool()
    status = check_demo_files(rows)
    if args.dry_run or status:
        return status
    if not (args.model or args.all):
        ap.error("name a --model, or pass --all or --dry-run")
    for model in (MODELS if args.all else [args.model]):
        run_model(model, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
