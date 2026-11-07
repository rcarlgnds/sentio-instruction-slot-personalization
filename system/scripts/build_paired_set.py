"""Finds the topic pairs behind the original paired track.

The main benchmark, 1,412 queries when this set was built, has zero queries in which the same
question is asked by different students, so it cannot measure personalization. This set uses Basic
Statistics, which CSCI teaches from the Lane book and STAT from Introductory Statistics 2e. For a topic
both books cover, one neutral question gets two student profiles, and the right passage depends on the
student's program. The script pairs passages across the two books by embedding similarity alone and
never calls cps.rerank (DITS) or any gate. The questions are written later, in
study/channel/provenance/paired_set_generation_workflow.js.

Run it from system/ with EMBEDDING_DEVICE=cpu.
"""

import json
import os

import numpy as np

OUT_DIR = "data/paired_set"
BOOK_CSCI = "Introduction_to_Statistics_Lane_PublicDomain"
BOOK_STAT = "Introductory_Statistics_2e_CCBY_2023"
MIN_TOKENS = 80
SIM_LOW, SIM_HIGH = 0.62, 0.93
TOP_PAIRS = 400


def fetch_book_chunks(qdrant, doc_title, limit=6000):
    from qdrant_client import models as qm
    out, offset = [], None
    while True:
        pts, offset = qdrant.client.scroll(
            collection_name=qdrant.collection,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="document_title", match=qm.MatchValue(value=doc_title))]),
            limit=512, offset=offset, with_payload=True, with_vectors=["dense"])
        out.extend(pts)
        if offset is None or len(out) >= limit:
            break
    return out


def main():
    from app.storage.qdrant_client import QdrantAdapter
    os.makedirs(OUT_DIR, exist_ok=True)
    q = QdrantAdapter()

    def load(title, tag):
        pts = fetch_book_chunks(q, title)
        keep = []
        for p in pts:
            pl = p.payload or {}
            if (pl.get("chunk_level") or "") == "parent":
                continue
            if (pl.get("token_count") or 0) < MIN_TOKENS:
                continue
            vec = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
            if vec is None:
                continue
            keep.append({"id": pl.get("chunk_id") or str(p.id), "text": pl.get("raw_text", ""),
                         "section": pl.get("section_path", ""), "page": (pl.get("position") or {}).get("page"),
                         "vec": np.asarray(vec, dtype=np.float32), "book": tag})
        print(f"  {tag}: {len(pts)} chunks fetched, {len(keep)} passed the filter")
        return keep

    print("fetching the chunks of the two Basic Statistics books")
    a = load(BOOK_CSCI, "CSCI")
    b = load(BOOK_STAT, "STAT")
    if not a or not b:
        print("ONE OF THE BOOKS IS EMPTY, check document_title in Qdrant")
        return

    A = np.stack([c["vec"] for c in a])
    B = np.stack([c["vec"] for c in b])
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)

    print("computing cross-book similarity")
    best_j = np.zeros(len(a), dtype=np.int64)
    best_s = np.zeros(len(a), dtype=np.float32)
    for i in range(0, len(a), 256):
        sims = A[i:i + 256] @ B.T
        best_j[i:i + 256] = sims.argmax(axis=1)
        best_s[i:i + 256] = sims.max(axis=1)

    cand = [(i, int(best_j[i]), float(best_s[i])) for i in range(len(a))
            if SIM_LOW <= best_s[i] <= SIM_HIGH]
    cand.sort(key=lambda t: -t[2])

    seen_b, pairs = set(), []
    for i, j, s in cand:
        if j in seen_b:
            continue
        seen_b.add(j)
        pairs.append({
            "sim": round(s, 4),
            "csci": {k: a[i][k] for k in ("id", "text", "section", "page")},
            "stat": {k: b[j][k] for k in ("id", "text", "section", "page")},
        })
        if len(pairs) >= TOP_PAIRS:
            break

    with open(f"{OUT_DIR}/topic_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\ncross-book topic pairs: {len(pairs)}")
    if pairs:
        ss = [p["sim"] for p in pairs]
        print(f"  similarity: min {min(ss):.3f}  median {ss[len(ss)//2]:.3f}  max {max(ss):.3f}")
        print("\nthree example pairs:")
        for p in pairs[:3]:
            print(f"  sim={p['sim']}")
            print(f"    CSCI [{p['csci']['section'][:60]}] {p['csci']['text'][:110]}...")
            print(f"    STAT [{p['stat']['section'][:60]}] {p['stat']['text'][:110]}...")
    print(f"\nsaved to {OUT_DIR}/topic_pairs.jsonl")


if __name__ == "__main__":
    main()
