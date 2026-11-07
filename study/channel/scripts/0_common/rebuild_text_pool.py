"""Puts the passage text back into a pool that ships as chunk ids.

The p2 and r2 pools ship without book text. Give this script a corpus from rebuild_corpus.py and it
writes the text pool the scorers read. Each chunk is checked against its hash. A chunk that differs only in
whitespace still passes, and then the pool is not byte for byte the one the models saw. The last line of the
output says which case it is.

    python rebuild_text_pool.py --track p2 --corpus corpus.jsonl --out pools_p2.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "study/channel/results"
TRACKS = {"p2": ("raw_p2", "pools_p2"), "r2": ("raw_r2", "pools_xpair_r2")}


def normalized_sha(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", (text or "").strip()).encode("utf-8")).hexdigest()


def read_lines(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def book_of(chunk_id: str) -> str:
    return chunk_id.rsplit("-", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=sorted(TRACKS))
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    folder, stem = TRACKS[args.track]
    pools = RESULTS / folder / "pools"
    anchors = {r["id"]: r for r in read_lines(pools / f"{stem}_anchors.jsonl")}
    queries = {r["query_id"]: r["query"] for r in read_lines(pools / f"{stem}_queries.jsonl")}
    wanted = set(anchors)

    corpus = {}
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                cid = r.get("_id") or r.get("id")
                if cid in wanted:
                    corpus[cid] = r["text"]

    missing, changed, loose = [], [], 0
    for cid, a in anchors.items():
        text = corpus.get(cid)
        if text is None:
            missing.append(cid)
        elif hashlib.sha256(text.encode("utf-8")).hexdigest() == a["exact_sha256"]:
            continue
        elif normalized_sha(text) == a["text_sha256"]:
            loose += 1
        else:
            changed.append(cid)
    print(f"{len(anchors)} chunks in the pool: {len(anchors) - len(missing) - len(changed) - loose} exact, "
          f"{loose} equal after whitespace normalization, {len(changed)} different, {len(missing)} missing")
    if missing or changed:
        per_book = Counter(book_of(c) for c in missing + changed)
        for book, n in per_book.most_common():
            print(f"  {n:6d}  {book}")
        print("Text pool not written. Check the PDF edition of the books above and the Docling version.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in read_lines(pools / f"{stem}_ids.jsonl"):
            full = {}
            for key, value in row.items():
                if key == "cands":
                    value = [{"id": c, "text": corpus[c]} for c in value]
                full[key] = value
                if key == "pair_id":
                    full["query"] = queries[row["query_id"]]
            f.write(json.dumps(full, ensure_ascii=False) + "\n")

    manifest = json.loads((RESULTS / folder / f"{stem}.manifest.json").read_text(encoding="utf-8"))
    want = manifest["output"]["sha256"]
    got = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"written: {args.out}")
    print(f"sha256 {got[:16]}, manifest {want[:16]}: {'identical to the pool the models saw' if got == want else 'differs'}")
    return 0 if (got == want or loose) else 1


if __name__ == "__main__":
    sys.exit(main())
