"""Finds the topic pairs of the extended paired track and fixes their order.

It matches passages as system/scripts/build_paired_set.py does, for every pair of books from two of
CSCI, ISYS, and MGMT, and for ACCT paired with each of them. It also drops near duplicates, chunks with
a quality flag, and the chunks of the original paired track. It uses only pairs with a cosine of at
least 0.66. The stages (--tahap, --ronde) write the topic pairs and their order.
"""

import argparse
import ast
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import extended_paired_common as U  # noqa: E402
import revision_broad_sample as T1  # noqa: E402


def original_paired_chunks() -> tuple:
    ids, hashes = set(), set()
    for r in U.read_jsonl(U.OLD_PAIRED):
        for key in ("gold_anchor", "counterpart_anchor"):
            content = r.get(key) or []
            if isinstance(content, str):
                content = ast.literal_eval(content)
            for a in content:
                ids.add(a.get("canonical_id"))
                hashes.add(a.get("text_sha256"))
    ids.discard(None)
    hashes.discard(None)
    return ids, hashes


def chunk_drop_reason(c: str, m: dict, old_id: set, old_hash: set):
    if m.get("chunk_level") != "child":
        return "not_a_child_chunk"
    if m.get("quality_flags"):
        return "quality_flag"
    if (m.get("token_count") or 0) < U.MIN_TOKEN:
        return "under_80_tokens"
    if c in old_id or m.get("text_sha256") in old_hash:
        return "paired_track_chunk"
    return None


def prepare_chunks():
    ids, text, meta = T1.load_corpus()
    X = T1.load_embeddings(ids)
    old_id, old_hash = original_paired_chunks()
    program = set(U.FOCUS) | {U.SPARE}
    book, discarded = defaultdict(list), Counter()
    for i, c in enumerate(ids):
        m = meta[c]
        if m.get("department") not in program:
            continue
        reason = chunk_drop_reason(c, m, old_id, old_hash)
        if reason:
            discarded[reason] += 1
        else:
            book[(m["department"], m["book_slug"])].append(i)
    return SimpleNamespace(ids=ids, text=text, meta=meta, X=X, book=dict(book),
                           discarded=dict(sorted(discarded.items())),
                           paired_track_anchors=len(old_id))


def jaccard(d, i: int, j: int, cache: dict) -> float:
    for k in (i, j):
        if k not in cache:
            cache[k] = set(d.text[k].lower().split())
    a, b = cache[i], cache[j]
    return len(a & b) / max(len(a | b), 1)


def program_pair_key(pa: str, pb: str) -> str:
    return "-".join(sorted((pa, pb)))


def same_work(ba: str, bb: str) -> bool:
    return frozenset((ba, bb)) in U.SAME_WORK_EDITIONS


def book_pair_matches(d, a: tuple, b: tuple, cache: dict) -> tuple:
    A, B = d.book[a], d.book[b]
    if (len(A), a[1]) > (len(B), b[1]):
        a, b, A, B = b, a, B, A
    S = d.X[A] @ d.X[B].T
    bj, bs = S.argmax(1), S.max(1)
    cands = sorted(((float(bs[i]), d.ids[A[i]], d.ids[B[int(bj[i])]], A[i], B[int(bj[i])])
                    for i in range(len(A)) if U.SIM_LOW <= bs[i] <= U.SIM_HIGH),
                   key=lambda t: (-t[0], t[1], t[2]))
    seen, out, dupes = set(), [], 0
    for s, ca, cb, ia, ib in cands:
        if cb in seen:
            continue
        jac = jaccard(d, ia, ib, cache)
        if s > U.DUP_COS or jac > U.DUP_JACCARD:
            dupes += 1
            continue
        seen.add(cb)
        out.append({"program_pair": program_pair_key(a[0], b[0]), "cosine_raw": s,
                    "cosine": round(s, 4), "jaccard": round(jac, 4),
                    "a": {"program": a[0], "book": a[1], "chunk": ca, "i": ia},
                    "b": {"program": b[0], "book": b[1], "chunk": cb, "i": ib}})
    return out, dupes


def all_matches(d) -> tuple:
    per_program, per_book, cache = defaultdict(list), {}, {}
    for a, b in itertools.combinations(sorted(d.book), 2):
        if a[0] == b[0] or same_work(a[1], b[1]):
            continue
        tp, dupes = book_pair_matches(d, a, b, cache)
        if not tp:
            continue
        per_program[tp[0]["program_pair"]].extend(tp)
        per_book[f"{a[0]}:{a[1]} | {b[0]}:{b[1]}"] = {
            "pairs_062_093": len(tp), "used_066_or_more": sum(t["cosine_raw"] >= U.MIN_SIM for t in tp),
            "near_duplicates_dropped": dupes}
    return per_program, per_book


def rotate_pairs(per_program: dict, rotation: tuple, taken: set) -> tuple:
    queue = {k: sorted((t for t in per_program.get(k, []) if t["cosine_raw"] >= U.MIN_SIM),
                       key=lambda t: (-t["cosine_raw"], t["a"]["chunk"], t["b"]["chunk"]))
             for k in rotation}
    pos = {k: 0 for k in rotation}
    out, skipped = [], Counter()
    while any(pos[k] < len(queue[k]) for k in rotation):
        for k in rotation:
            while pos[k] < len(queue[k]):
                t = queue[k][pos[k]]
                pos[k] += 1
                if t["a"]["chunk"] in taken or t["b"]["chunk"] in taken:
                    skipped[k] += 1
                    continue
                taken.update((t["a"]["chunk"], t["b"]["chunk"]))
                out.append(t)
                break
    return out, dict(skipped)


def passage_record(d, x: dict) -> dict:
    m = d.meta[x["chunk"]]
    return {"program": x["program"], "book": x["book"], "title": m.get("document_title"),
            "course_code": m.get("course_code"), "semester": m.get("semester"),
            "chunk": x["chunk"], "section": m.get("section_path"),
            "text": d.text[x["i"]][:U.MAX_TEXT]}


def record_order(d, ordering: list) -> list:
    rng = np.random.default_rng(U.SEED_DISPLAY)
    out = []
    for n, (clusters, t) in enumerate(ordering, start=1):
        pa, pb = passage_record(d, t["a"]), passage_record(d, t["b"])
        swap = bool(rng.integers(2))
        out.append({"tp_id": f"tp-{n:05d}", "urut": n, "kelompok": clusters,
                    "program_pair": t["program_pair"], "cosine": t["cosine"], "jaccard": t["jaccard"],
                    "shown_swapped": swap, "paragraf": [pb, pa] if swap else [pa, pb]})
    return out


def build_order(d) -> tuple:
    per_program, per_book = all_matches(d)
    taken = set()
    focus, skipped_f = rotate_pairs(per_program, U.FOCUS_ROTATION, taken)
    acct, skipped_a = rotate_pairs(per_program, U.ACCT_ROTATION, taken)
    records = record_order(d, [("fokus", t) for t in focus] + [("acct", t) for t in acct])
    count = {
        "chunks_dropped": d.discarded,
        "chunks_by_program_kept": dict(sorted(Counter(k[0] for k, v in d.book.items()
                                                      for _ in v).items())),
        "by_program_pair": {k: {"pairs_062_093": len(v),
                                "used_066_or_more": sum(t["cosine_raw"] >= U.MIN_SIM for t in v)}
                            for k, v in sorted(per_program.items())},
        "skipped_chunk_already_in_a_pair": {**skipped_f, **skipped_a},
        "in_order": {"fokus": len(focus), "acct": len(acct)},
        "in_order_by_program_pair": dict(sorted(Counter(r["program_pair"] for r in records).items())),
        "by_book_pair": per_book,
    }
    return records, count


def stage_check(d):
    records, count = build_order(d)
    st = U.preregistration_status()
    print(json.dumps({**{k: v for k, v in count.items() if k != "by_book_pair"},
                      "preregistration_ready": st["ready"],
                      "preregistration_missing_text": st["missing_text"],
                      "not_clean": sorted(n for n, v in {**st["scripts"], **st["r1_modules"]}.items()
                                          if v["tracked"] is False or not v["clean"])},
                     indent=1, ensure_ascii=False))
    print("\nbook pairs by used topic pairs:")
    for k, v in sorted(count["by_book_pair"].items(), key=lambda kv: -kv[1]["used_066_or_more"])[:30]:
        print(f"  {v['used_066_or_more']:5d} / {v['pairs_062_093']:5d}  dup {v['near_duplicates_dropped']:3d}  {k}")


def stage_order(d):
    st = U.require_preregistration()
    if U.PAIRS.exists() or U.PAIRS_MANIFEST.exists():
        raise SystemExit(f"ABORT: {U.PAIRS.name} exists. The order is written once.")
    records, count = build_order(d)
    U.RAW_P2.mkdir(parents=True, exist_ok=True)
    tmp = U.PAIRS.with_name(U.PAIRS.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "created": U.iso(),
        "script": "study/channel/scripts/10_rebuilt_and_extended/extended_paired_topic_pairs.py",
        "command": "extended_paired_topic_pairs.py --tahap urutan",
        "rules": {"programs_focus": U.FOCUS, "program_fallback": U.SPARE,
                  "rotation_focus": U.FOCUS_ROTATION, "rotation_fallback": U.ACCT_ROTATION,
                  "cosine_window": [U.SIM_LOW, U.SIM_HIGH], "cosine_used_from": U.MIN_SIM,
                  "near_duplicate": {"cosine_above": U.DUP_COS, "word_jaccard_above": U.DUP_JACCARD},
                  "min_tokens": U.MIN_TOKEN, "min_chars": 200, "text_cut": U.MAX_TEXT,
                  "editions_never_paired": [sorted(e) for e in U.SAME_WORK_EDITIONS],
                  "shown_order_seed": U.SEED_DISPLAY, "test_holdout": "never read"},
        "counts": count,
        "scripts": {n: {"sha256": v["sha256"], "commit": v["commit"]} for n, v in st["scripts"].items()},
        "inputs": {"corpus": U.sha(T1.CORPUS), "embedding_cache": U.sha(T1.CACHE),
                   "embedding_cache_ids": U.sha(T1.CACHE_ID), "paired_queries": U.sha(U.OLD_PAIRED)},
        "numpy_version": np.__version__,
        "output": {"path": "results/raw_p2/pasangan_topik_p2.jsonl", "sha256": U.sha(tmp),
                   "records": len(records)},
    }
    U.write_json(U.PAIRS_MANIFEST, manifest)
    tmp.replace(U.PAIRS)
    print(json.dumps({"in_order": count["in_order"],
                      "in_order_by_program_pair": count["in_order_by_program_pair"],
                      "sha256": manifest["output"]["sha256"]}, indent=1))


def load_order() -> list:
    if not (U.PAIRS.exists() and U.PAIRS_MANIFEST.exists()):
        raise SystemExit("ABORT: the topic-pair order is missing; run --tahap urutan first.")
    m = json.loads(U.PAIRS_MANIFEST.read_text())
    if U.sha(U.PAIRS) != m["output"]["sha256"]:
        raise SystemExit("ABORT: pasangan_topik_p2.jsonl does not match its manifest.")
    return U.read_jsonl(U.PAIRS)


def allowed_round(n: int, ordering: list) -> dict:
    if n == 1:
        if U.ROUND_FILE.exists() and json.loads(U.ROUND_FILE.read_text()).get("rounds"):
            raise SystemExit("ABORT: round 1 was already sent.")
        focus = [r for r in ordering if r["kelompok"] == "fokus"]
        return {"ronde": 1, "list": "fokus", "tp_from": focus[0]["urut"],
                "tp_to": focus[min(U.N_ROUNDS, len(focus)) - 1]["urut"]}
    if not U.ROUND_FILE.exists():
        raise SystemExit("ABORT: ronde_p2.json is missing; run extended_paired_assemble.py --tahap status.")
    upcoming = json.loads(U.ROUND_FILE.read_text()).get("next") or {}
    if upcoming.get("ronde") != n:
        raise SystemExit(f"ABORT: the stopping rule allows {upcoming}, not round {n}.")
    return upcoming


def stage_round(n: int):
    U.require_preregistration()
    ordering = load_order()
    permit = allowed_round(n, ordering)
    folder = U.BATCH / f"ronde_{n:02d}"
    if folder.exists():
        raise SystemExit(f"ABORT: {folder} exists. A round is written once.")
    pick = [r for r in ordering if permit["tp_from"] <= r["urut"] <= permit["tp_to"]
             and r["kelompok"] == permit["list"]]
    if not pick or len(pick) > U.N_ROUNDS:
        raise SystemExit(f"ABORT: round {n} would hold {len(pick)} topic pairs.")
    folder.mkdir(parents=True)
    fpath = []
    for b, begin in enumerate(range(0, len(pick), U.N_BATCH), start=1):
        content = {"ronde": n, "batch": b, "list": permit["list"],
               "topic_pairs": [{"tp_id": r["tp_id"], "program_pair": r["program_pair"],
                                "paragraf": [{"label": f"Chunk {i + 1}", "program": p["program"],
                                              "book_title": p["title"], "section": p["section"],
                                              "text": p["text"]}
                                             for i, p in enumerate(r["paragraf"])]}
                               for r in pick[begin:begin + U.N_BATCH]]}
        p = folder / f"tulis_{b:03d}.json"
        U.write_json(p, content)
        fpath.append({"file": U.rel(p), "sha256": U.sha(p),
                       "tp_ids": [t["tp_id"] for t in content["topic_pairs"]]})
    U.write_json(folder / "manifest_ronde.json", {
        "created": U.iso(), "ronde": n, "list": permit["list"], "tp_from": permit["tp_from"],
        "tp_to": permit["tp_to"], "topic_pairs": len(pick), "batches": fpath,
        "order_sha256": U.sha(U.PAIRS)})
    print(json.dumps({"ronde": n, "list": permit["list"], "topic_pairs": len(pick),
                      "batches": len(fpath), "folder": str(folder)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tahap", choices=["cek", "urutan", "ronde"], required=True)
    ap.add_argument("--ronde", type=int, default=None)
    args = ap.parse_args()
    if args.tahap == "ronde":
        if not args.ronde or args.ronde < 1:
            ap.error("--tahap ronde needs --ronde N (N >= 1)")
        stage_round(args.ronde)
        return
    d = prepare_chunks()
    if args.tahap == "cek":
        stage_check(d)
    else:
        stage_order(d)


if __name__ == "__main__":
    main()
