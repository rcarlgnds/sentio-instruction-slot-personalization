import os

from app.retrieval import cps
from app.retrieval.cps import CPSWeights

_RERANKER = None
_GTE = None
GTE_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"


def _get_gte():
    global _GTE
    if _GTE is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GTE_MODEL, trust_remote_code=True)
        mod = AutoModelForSequenceClassification.from_pretrained(
            GTE_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
        mod.eval()
        _GTE = (tok, mod, torch)
    return _GTE


def _gte_rerank(results, query_text):
    if not results:
        return results
    tok, mod, torch = _get_gte()
    pairs = [(query_text, r.payload.get("raw_text", "")) for r in results]
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), 16):
            batch = pairs[i:i + 16]
            enc = tok([p[0] for p in batch], [p[1] for p in batch],
                      return_tensors="pt", padding=True, truncation=True, max_length=512)
            scores.extend(mod(**enc).logits.view(-1).float().tolist())
    for r, s in zip(results, scores):
        r.cps_score = float(s)
    return [r for _, _, r in sorted(
        ((s, (r.payload or {}).get("chunk_id", ""), r) for s, r in zip(scores, results)),
        key=lambda x: (-x[0], x[1]))]


def _get_cross_encoder():
    global _RERANKER
    if _RERANKER is None:
        from FlagEmbedding import FlagReranker
        _RERANKER = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    return _RERANKER


def force_load_cross_encoder():
    _get_cross_encoder()


def rerank(results, student, config, query_text="", prerequisite_graph=None, top_m: int = None):
    if top_m is None:
        top_m = int(os.environ.get("SENTIO_TOP_M", "10"))

    if config.reranker == "cosine":
        return sorted(results, key=lambda r: (-r.sim_score, (r.payload or {}).get("chunk_id", "")))

    if config.reranker == "cps":
        return cps.rerank(results, student, weights=CPSWeights(*config.weights),
                          prerequisite_graph=prerequisite_graph)

    if config.reranker == "cross_encoder":
        return _cross_encoder_rerank(results, query_text)

    if config.reranker == "gte":
        return _gte_rerank(results, query_text)

    if config.reranker == "cps_then_gte":
        cps_ranked = cps.rerank(results, student, weights=CPSWeights(*config.weights),
                                prerequisite_graph=prerequisite_graph)
        return _gte_rerank(cps_ranked[:top_m], query_text) + cps_ranked[top_m:]

    if config.reranker == "cps_then_reranker":
        cps_ranked = cps.rerank(results, student, weights=CPSWeights(*config.weights),
                                prerequisite_graph=prerequisite_graph)
        head = _cross_encoder_rerank(cps_ranked[:top_m], query_text)
        return head + cps_ranked[top_m:]

    raise ValueError(f"unknown reranker: {config.reranker}")


def _cross_encoder_rerank(results, query_text):
    if not results:
        return results
    reranker = _get_cross_encoder()
    pairs = [[query_text, r.payload.get("raw_text", "")] for r in results]
    scores = reranker.compute_score(pairs, normalize=True)
    try:
        scores = [float(s) for s in scores]
    except TypeError:
        scores = [float(scores)]
    ranked = sorted(zip(results, scores),
                    key=lambda t: (-t[1], (t[0].payload or {}).get("chunk_id", "")))
    return [r for r, _ in ranked]
