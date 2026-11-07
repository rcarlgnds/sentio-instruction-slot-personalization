import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[3])

AI = ROOT / "system"
sys.path.insert(0, str(AI))
os.chdir(AI)

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
OUT = RAW / "pools_paired.jsonl"


def main():
    from app.benchmark.eval.harness import EvalHarness
    from app.ingestion.embedder import HybridEmbedder
    from app.retrieval import cps
    from app.retrieval.prerequisite_graph import PrerequisiteGraph
    from app.routing.semantic_router import SemanticRouter
    from app.storage.llm_client import get_llm_client
    from app.storage.postgres_client import PostgresAdapter
    from app.storage.qdrant_client import QdrantAdapter

    rows = [json.loads(l) for l in (AI / "data/paired_set/paired_queries.jsonl").open() if l.strip()]
    embedder = HybridEmbedder(); embedder.load()
    router = SemanticRouter(encoder=embedder); router.load()
    pg = PostgresAdapter()
    cps.load_department_relations(pg)
    harness = EvalHarness(embedder=embedder, qdrant=QdrantAdapter(),
                          ollama=get_llm_client(),
                          prerequisite_graph=PrerequisiteGraph.load(pg),
                          router=router)
    n = 0
    with OUT.open("w") as f:
        for q in rows:
            pf = harness.prefetch(q, time.perf_counter)
            cands = [{"id": r.payload.get("chunk_id", ""),
                      "text": (r.payload.get("raw_text") or "")[:4000]}
                     for r in pf["results"]]
            f.write(json.dumps({
                "query_id": q["query_id"], "pair_id": q["pair_id"],
                "query": q["query_text"],
                "gold": q["gold_relevant_chunks"],
                "counterpart": q["counterpart_chunks"],
                "profile": q.get("student_profile") or {},
                "cands": cands,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"paired pool dump: {n} queries -> {OUT}")


if __name__ == "__main__":
    main()
