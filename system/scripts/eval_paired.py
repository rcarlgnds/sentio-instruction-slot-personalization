import json
import math
import statistics
import time

QUERIES = "data/paired_set/paired_queries.jsonl"
OUT = "data/paired_set/eval_paired.json"
ARMS = ["A", "S1", "S2", "B", "C", "D"]


def ndcg_at_k(order, gold, k=5):
    rels = [1.0 if c in gold else 0.0 for c in order[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def main():
    from app.benchmark.eval import rerank as rr
    from app.benchmark.eval.configs import CONFIG_BY_KEY
    from app.benchmark.eval.harness import EvalHarness
    from app.benchmark.eval.stats import paired_bootstrap_diff
    from app.ingestion.embedder import HybridEmbedder
    from app.retrieval import cps
    from app.retrieval.prerequisite_graph import PrerequisiteGraph
    from app.routing.semantic_router import SemanticRouter
    from app.storage.llm_client import get_llm_client
    from app.storage.postgres_client import PostgresAdapter
    from app.storage.qdrant_client import QdrantAdapter
    import copy

    rows = [json.loads(l) for l in open(QUERIES) if l.strip()]
    print(f"queries: {len(rows)} ({len(rows)//2} pairs)\n")

    embedder = HybridEmbedder()
    embedder.load()
    router = SemanticRouter(encoder=embedder)
    router.load()
    pg = PostgresAdapter()
    cps.load_department_relations(pg)
    graph = PrerequisiteGraph.load(pg)
    harness = EvalHarness(embedder=embedder, qdrant=QdrantAdapter(), ollama=get_llm_client(),
                          prerequisite_graph=graph, router=router)
    try:
        rr.force_load_cross_encoder()
    except Exception:
        pass

    pools = {}
    for i, q in enumerate(rows):
        pools[q["query_id"]] = harness.prefetch(q, time.perf_counter)
        if (i + 1) % 25 == 0:
            print(f"  prefetch {i+1}/{len(rows)}")

    res = {a: {"pairwise": [], "top1": [], "ndcg": [], "in_pool": []} for a in ARMS}
    for q in rows:
        pf = pools[q["query_id"]]
        if not pf["results"]:
            continue
        gold = set(q["gold_relevant_chunks"])
        other = set(q["counterpart_chunks"])
        for a in ARMS:
            pool = [copy.copy(r) for r in pf["results"]]
            ranked = rr.rerank(pool, pf["student"], CONFIG_BY_KEY[a], q["query_text"],
                               prerequisite_graph=graph)
            ids = [r.payload.get("chunk_id", "") for r in ranked]
            pos = {c: i for i, c in enumerate(ids)}
            g = min((pos[c] for c in gold if c in pos), default=None)
            o = min((pos[c] for c in other if c in pos), default=None)
            res[a]["in_pool"].append(1.0 if (g is not None and o is not None) else 0.0)
            if g is not None and o is not None:
                res[a]["pairwise"].append(1.0 if g < o else 0.0)
            res[a]["ndcg"].append(ndcg_at_k(ids, gold))
            res[a]["top1"].append(1.0 if ids and ids[0] in gold else 0.0)

    print("=== PAIRED SET RESULTS (personalization) ===")
    print(f"{'arm':4s} {'pairwise_acc':>13s} {'n_pair':>7s} {'top1_gold':>10s} {'nDCG@5':>8s} {'both_in':>10s}")
    for a in ARMS:
        r = res[a]
        pw = statistics.fmean(r["pairwise"]) if r["pairwise"] else float("nan")
        print(f"{a:4s} {pw:13.4f} {len(r['pairwise']):7d} {statistics.fmean(r['top1']):10.4f} "
              f"{statistics.fmean(r['ndcg']):8.4f} {statistics.fmean(r['in_pool']):10.4f}")

    print("\n=== test against random guessing at 0.5 (bootstrap against the blind arm A) ===")
    for a in ARMS:
        if a == "A" or not res[a]["pairwise"]:
            continue
        n = min(len(res[a]["pairwise"]), len(res["A"]["pairwise"]))
        d = paired_bootstrap_diff(res[a]["pairwise"][:n], res["A"]["pairwise"][:n])
        sig = "SIGNIFIKAN" if d["p_value"] is not None and d["p_value"] < 0.05 else ""
        print(f"  {a:3s} vs A: diff={d['diff']:+.4f} p={d['p_value']:.4f} {sig}")

    json.dump({a: {k: (statistics.fmean(v) if v else None) for k, v in res[a].items()}
               for a in ARMS}, open(OUT, "w"), indent=2)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
