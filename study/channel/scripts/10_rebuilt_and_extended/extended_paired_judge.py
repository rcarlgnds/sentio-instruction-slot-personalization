import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import extended_paired_common as U  # noqa: E402
import rebuilt_broad_sample as R2  # noqa: E402


def load_verdict_cache() -> dict:
    cache = {}
    if not U.JUDGE.exists():
        return cache
    rows = U.JUDGE.read_text(encoding="utf-8").split("\n")
    for i, b in enumerate(rows):
        if not b.strip():
            continue
        try:
            r = json.loads(b)
        except json.JSONDecodeError:
            if i == len(rows) - 1:
                continue
            raise SystemExit(f"ABORT: line {i + 1} of {U.JUDGE} is not valid JSON.")
        cache[r["kunci"]] = r
    return cache


def trim_partial_tail() -> int:
    if not U.JUDGE.exists():
        return 0
    data = U.JUDGE.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    truncate = data.rfind(b"\n") + 1
    with U.JUDGE.open("r+b") as f:
        f.truncate(truncate)
    return len(data) - truncate


class Judge(R2.Judge):
    def __init__(self, server=None):
        self.cache = load_verdict_cache()
        self.server = server
        self.fpath = None
        self.write_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.local = threading.local()
        self.new_verdicts = Counter()
        self.used = set()


def round_questions(round_no: int) -> list:
    ordering = {r["tp_id"]: r for r in U.read_jsonl(U.PAIRS)}
    write = {r["tp_id"]: r for r in U.read_jsonl(U.WRITE_FILE) if r["ronde"] == round_no}
    passed = [r for r in U.read_jsonl(U.SCREEN_FILE) if r["ronde"] == round_no and r["lolos"]]
    out = []
    for s in sorted(passed, key=lambda r: ordering[r["tp_id"]]["urut"]):
        t = write[s["tp_id"]]
        if t["question"] != s["question_screened"]:
            raise SystemExit(f"ABORT: {s['tp_id']}: the screened question differs from the written one.")
        out.append({"tp_id": s["tp_id"], "question": t["question"],
                    "paragraf": [{"chunk": p["chunk"], "text": p["text"]}
                                 for p in ordering[s["tp_id"]]["paragraf"]]})
    return out


ANSWER = ("penuh", "sebagian")


def group_exclusion(h: dict):
    if h["jelas"] != "ya":
        return "question_not_clear"
    if not all(p["a"] in ANSWER and p["b"] in ANSWER for p in h["paragraf"]):
        return "passage_not_an_answer"
    if h["paragraf"][0]["a"] != h["paragraf"][1]["a"]:
        return "passages_not_one_class"
    return None


def judge_question(judge, s: dict) -> dict:
    h = {"tp_id": s["tp_id"], "jelas": judge.rate("jelas", s["tp_id"], None, pertanyaan=s["question"]),
         "paragraf": []}
    if h["jelas"] == "ya":
        for p in s["paragraf"]:
            a = judge.rate("jawab", s["tp_id"], p["chunk"], pertanyaan=s["question"], paragraf=p["text"])
            b = judge.rate("jawab_b", s["tp_id"], p["chunk"], pertanyaan=s["question"],
                           paragraf=p["text"]) if a in ANSWER else None
            h["paragraf"].append({"chunk": p["chunk"], "a": a, "b": b, "full_both": a == b == "penuh"})
    h["exclusion"] = group_exclusion(h)
    h["group"] = h["exclusion"] is None
    h["class"] = h["paragraf"][0]["a"] if h["group"] else None
    h["strict"] = h["group"] and all(x["full_both"] for x in h["paragraf"])
    return h


def summarize(result: list) -> dict:
    return {"questions": len(result),
            "clear": dict(Counter(h["jelas"] for h in result)),
            "passages_wording_a": dict(Counter(p["a"] for h in result for p in h["paragraf"])),
            "passages_wording_b": dict(Counter(str(p["b"]) for h in result for p in h["paragraf"])),
            "passages_full_both": sum(p["full_both"] for h in result for p in h["paragraf"]),
            "exclusions": dict(Counter(h["exclusion"] or "group" for h in result)),
            "groups": sum(h["group"] for h in result),
            "groups_by_class": dict(Counter(h["class"] for h in result if h["group"])),
            "strict_groups": sum(h["strict"] for h in result)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ronde", type=int, required=True)
    ap.add_argument("--batas", type=int, default=None, help="smoke run on the first N questions")
    ap.add_argument("--paralel", type=int, default=R2.PARALLEL)
    args = ap.parse_args()
    st = U.require_preregistration()
    folder = U.BATCH / f"ronde_{args.ronde:02d}"
    if not (folder / "simpan_ronde.json").exists():
        raise SystemExit(f"ABORT: round {args.ronde} is not stored; run extended_paired_store.py first.")
    old = folder / "juri_ronde.json"
    if args.batas is None and old.exists() and json.loads(old.read_text()).get("rule") != U.JUDGE_RULE:
        raise SystemExit(f"ABORT: {old.name} was made under another judge rule; move it aside "
                         "(mv, never rm) and report it before judging this round again.")
    question = round_questions(args.ronde)
    target = question if args.batas is None else question[:args.batas]
    url, headers = R2.server_address()
    info = R2.server_info(url, headers)
    if R2.JUDGE_MODEL not in info["served_models"]:
        raise SystemExit(f"ABORT: the server serves {info['served_models']}, not {R2.JUDGE_MODEL}.")
    U.RAW_P2.mkdir(parents=True, exist_ok=True)
    result, t0 = [], time.time()
    with U.JUDGE.open("a", encoding="utf-8") as fpath:
        if fcntl is not None:
            try:
                fcntl.flock(fpath, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("ABORT: another judge process holds juri_p2.jsonl; run one at a time.")
        with U.JUDGE_SERVER.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": U.iso(), "ronde": args.ronde, "url": url, **info,
                                "sampling": R2.SAMPLING, "prompt_version": R2.JUDGE_VERSION,
                                "parallel_requests": args.paralel, "limit": args.batas}) + "\n")
        discarded = trim_partial_tail()
        if discarded:
            print(f"note: removed a partial last line ({discarded} bytes)", flush=True)
        judge = Judge(server=(url, headers))
        judge.fpath = fpath
        print(f"judging {len(target)} questions of round {args.ronde}; {len(judge.cache)} verdicts "
              "already stored", flush=True)
        ex = ThreadPoolExecutor(max_workers=args.paralel)
        try:
            task = {ex.submit(judge_question, judge, s): s["tp_id"] for s in target}
            for i, f in enumerate(as_completed(task), start=1):
                result.append(f.result())
                if i % 25 == 0 or i == len(task):
                    print(f"{U.iso()} {i}/{len(task)} questions, {sum(judge.new_verdicts.values())} new "
                          f"verdicts, {time.time() - t0:.0f} s", flush=True)
        except BaseException:
            judge.stop_event.set()
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    unreadable, empty = R2.unreadable_verdicts(judge)
    result.sort(key=lambda h: h["tp_id"])
    summary = {**summarize(result), "unreadable_replies": unreadable, "of_which_empty": empty,
                 "verdicts_used": len(judge.used), "new_verdicts": dict(judge.new_verdicts)}
    print(json.dumps(summary, indent=1))
    if args.batas is not None:
        U.write_json(folder / "juri_smoke.json", {"created": U.iso(), **summary})
        if unreadable:
            raise SystemExit(f"STOP for review: the smoke run has {unreadable} unreadable replies "
                             f"({empty} empty) among {len(judge.used)} verdicts.")
        return
    if unreadable > R2.MAX_UNREADABLE * len(judge.used):
        raise SystemExit(f"STOP: {unreadable} of {len(judge.used)} verdicts are unreadable ({empty} "
                         "empty), above the pre-registered 1%; this round is refused.")
    U.write_json(folder / "juri_ronde.json", {"created": U.iso(), "rule": U.JUDGE_RULE, **summary,
                                             "results": result})


if __name__ == "__main__":
    main()
