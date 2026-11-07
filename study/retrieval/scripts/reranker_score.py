import os
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REVISIONS = {"Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
             "Qwen/Qwen3-Reranker-4B": "22e683669bc0f0bd69640a1354a6d0aebcfeede5"}

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[3])

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
POOLS = RAW / "pools_traindev_top20.jsonl"
PROFILES = ROOT / "system/data/rads_sample/queries.jsonl"


def load_profiles() -> dict:
    prof = {}
    for line in PROFILES.open():
        r = json.loads(line)
        prof[r["query_id"]] = r.get("student_profile") or {}
    return prof


def profile_instruction(p: dict) -> str:
    dept = p.get("department") or p.get("department_name") or "unknown"
    sem = p.get("current_semester") or p.get("semester") or "?"
    return (f"Given a query from a {dept} student in semester {sem}, retrieve "
            f"passages from that student's own study program that answer the "
            f"query at a level appropriate for semester {sem}")

MODELS = {"0.6b": "Qwen/Qwen3-Reranker-0.6B", "4b": "Qwen/Qwen3-Reranker-4B"}
TASK = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements "
          "based on the Query and the Instruct provided. Note that the answer can "
          "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_LEN = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--instruction", choices=["default", "profile"], default="default",
                    help="profile = English instruction built from student_profile "
                         "per query (personalization variant ON)")
    ap.add_argument("--pools", default=None, help="override the pool file (for example paired)")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="also score test_holdout queries, which opens the held-out split a second time")
    args = ap.parse_args()
    pools_path = Path(args.pools) if args.pools else POOLS
    assignment = json.loads((Path(__file__).resolve().parents[2] / "_shared" / "split_manifest.json").read_text())["assignment"]
    n_holdout = sum(1 for l in pools_path.open() if l.strip() and assignment.get(json.loads(l)["query_id"]) == "test_holdout")
    if n_holdout and not args.allow_holdout:
        raise SystemExit(f"ABORT SEAL: {n_holdout} test_holdout query_id in the pool; pass --allow-holdout to record a second opening.")
    if n_holdout:
        print(f"SEAL OPENED AGAIN: {n_holdout} test_holdout queries scored", flush=True)
    profile = load_profiles() if args.instruction == "profile" else {}

    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=REVISIONS.get(name), torch_dtype=torch.float16).cuda().eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    pre = tok.encode(PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)

    tag = Path(args.pools).stem.replace("pools_", "") if args.pools else "ablation"
    out_path = RAW / f"qwen_scores_{args.model}_{tag}_{args.instruction}.jsonl"
    torch.cuda.reset_peak_memory_stats()
    n = 0
    t_all0 = time.perf_counter()
    with out_path.open("w") as out:
        for line in pools_path.open():
            row = json.loads(line)
            if args.limit and n >= args.limit:
                break
            inst = TASK
            if args.instruction == "profile":
                p = row.get("profile") or profile.get(row["query_id"], {})
                inst = profile_instruction(p)
            pairs = [f"<Instruct>: {inst}\n<Query>: {row['query']}\n<Document>: {c['text'][:4000]}"
                     for c in row["cands"]]
            t0 = time.perf_counter()
            score = []
            with torch.no_grad():
                for i in range(0, len(pairs), args.batch):
                    chunk = pairs[i:i + args.batch]
                    enc = [pre + tok.encode(p, add_special_tokens=False)[:MAX_LEN - len(pre) - len(suf)] + suf
                           for p in chunk]
                    maxlen = max(len(e) for e in enc)
                    ids = torch.full((len(enc), maxlen), tok.pad_token_id, dtype=torch.long)
                    att = torch.zeros((len(enc), maxlen), dtype=torch.long)
                    for j, e in enumerate(enc):
                        ids[j, maxlen - len(e):] = torch.tensor(e)
                        att[j, maxlen - len(e):] = 1
                    logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, -1, :]
                    duo = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
                    p_yes = torch.nn.functional.log_softmax(duo.float(), dim=1)[:, 1].exp()
                    score.extend(p_yes.tolist())
            dt = time.perf_counter() - t0
            order = sorted(zip((c["id"] for c in row["cands"]), score),
                          key=lambda x: (-x[1], x[0]))
            out.write(json.dumps({
                "query_id": row["query_id"], "gold": row["gold"],
                "counterpart": row.get("counterpart"),
                "ranking": [i for i, _ in order],
                "scores": [round(s, 6) for _, s in order],
                "latency_s": round(dt, 4),
            }) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"{n} queries, {time.perf_counter()-t_all0:.0f}s, "
                      f"vram_peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)

    print(json.dumps({
        "model": name, "n": n,
        "total_s": round(time.perf_counter() - t_all0, 1),
        "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }))


if __name__ == "__main__":
    main()
