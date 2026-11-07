"""Rebuilds the chunk corpus from the 40 book PDFs, without a database or an embedder.

It runs the ingest steps that decide the chunk text (Docling parsing, boilerplate removal, chunking,
duplicate removal) and numbers the chunks the way the vector store did. The PDFs do not ship, and
study/_shared/book_sources.csv lists each book with its source, license, and hash.

    python rebuild_corpus.py --books <folder with the PDFs> --out corpus.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "system"))

from app.benchmark.release import canonical_id, slug_for  # noqa: E402
from app.ingestion.chunking.structure_aware import StructureAwareChunker  # noqa: E402
from app.ingestion.cleaning import dedup_chunks, strip_boilerplate  # noqa: E402
from app.ingestion.parsers.pdf_parser import PDFParser  # noqa: E402


def book_files(folder: Path) -> list:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def chunks_of(path: Path, parser, chunker) -> list:
    parsed = parser.parse(path.read_bytes(), path.name)
    parsed, _ = strip_boilerplate(parsed)
    chunks = chunker.chunk(parsed)
    chunks, _ = dedup_chunks(chunks)
    slug = slug_for(path.name)
    rows = []
    for index, chunk in enumerate(chunks):
        rows.append({"_id": canonical_id(slug, index), "text": chunk.raw_text,
                     "metadata": {"document_title": parsed.title, "page": (chunk.position or {}).get("page"),
                                  "chunk_level": str(getattr(chunk.chunk_level, "value", chunk.chunk_level))}})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--books", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", action="append", default=[], help="book slug; repeat to take several")
    args = ap.parse_args()

    files = book_files(args.books)
    if args.only:
        files = [p for p in files if slug_for(p.name) in set(args.only)]
    if not files:
        print(f"No PDF found under {args.books}")
        return 1

    parser, chunker = PDFParser(), StructureAwareChunker()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for n, path in enumerate(files, 1):
            t0 = time.time()
            rows = chunks_of(path, parser, chunker)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            print(f"[{n}/{len(files)}] {path.name}: {len(rows)} chunks in {time.time() - t0:.0f}s", flush=True)
    print(f"written: {args.out} ({total} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
