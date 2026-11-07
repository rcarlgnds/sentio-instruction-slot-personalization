import copy
from dataclasses import dataclass, asdict

from app.benchmark.eval import rerank as rr
from app.config import settings
from app.domain.models import StudentContext
from app.generation.socratic_mode import generate_socratic
from app.generation.strict_mode import generate_strict
from app.retrieval import cps
from app.retrieval.hybrid_search import build_payload_filter
from app.retrieval.normalization import CalibrationBounds, normalize_sims


@dataclass
class EvalRecord:
    query_id: str
    class_label: str
    config: str
    mode: str
    question: str
    answer: str
    contexts: list
    retrieved_chunk_ids: list
    gold_chunk_ids: list
    reference: str
    refused: bool
    refusal_reason: str
    retrieval_latency_s: float
    total_latency_s: float
    top_cps: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class EvalHarness:
    def __init__(self, embedder, qdrant, ollama, prerequisite_graph, router=None,
                 bounds: CalibrationBounds = None, top_k_ctx: int = 5, verify_fn=None):
        self.embedder = embedder
        self.qdrant = qdrant
        self.ollama = ollama
        self.graph = prerequisite_graph
        self.router = router
        self.bounds = bounds or CalibrationBounds.load(settings.calibration_path)
        self.top_k_ctx = top_k_ctx
        self.verify_fn = verify_fn

    def prefetch(self, query: dict, clock) -> dict:
        student = self._student(query)
        t0 = clock()
        embedding = self.embedder.embed_query(query["query_text"])
        forced = query.get("class_metadata", {}).get("force_mode")
        if forced:
            mode = forced
        else:
            mode = self._route(embedding["dense"], query["class_label"])
        results = self.qdrant.hybrid_search(
            dense_query=embedding["dense"],
            sparse_indices=embedding["sparse_indices"],
            sparse_values=embedding["sparse_values"],
            limit=settings.retrieval_pool_size,
            payload_filter=build_payload_filter(mode, student),
        )
        in_domain = normalize_sims(results, self.bounds) if results else False
        return {"student": student, "mode": mode, "results": results,
                "in_domain": in_domain, "retrieval_base_s": clock() - t0}

    def run_query(self, query: dict, config, clock, prefetched: dict = None,
                  retrieval_only: bool = False) -> EvalRecord:
        pf = prefetched or self.prefetch(query, clock)
        student, mode = pf["student"], pf["mode"]
        threshold = settings.strict_min_cps if mode == "strict" else settings.socratic_min_cps
        rec = EvalRecord(
            query_id=query["query_id"], class_label=query["class_label"], config=config.key,
            mode=mode, question=query["query_text"], answer="", contexts=[],
            retrieved_chunk_ids=[], gold_chunk_ids=query.get("gold_relevant_chunks", []),
            reference=query.get("expected_answer_summary", ""), refused=True,
            refusal_reason="", retrieval_latency_s=0.0, total_latency_s=0.0)

        t0 = clock()
        if not pf["results"]:
            rec.refusal_reason = "insufficient_context"
            rec.retrieval_latency_s = rec.total_latency_s = round(pf["retrieval_base_s"], 4)
            return rec
        if not pf["in_domain"]:
            rec.refusal_reason = "out_of_domain"
            rec.retrieval_latency_s = rec.total_latency_s = round(pf["retrieval_base_s"] + (clock() - t0), 4)
            return rec

        pool = [copy.copy(r) for r in pf["results"]]
        ranked = rr.rerank(pool, student, config, query["query_text"],
                           prerequisite_graph=self.graph)
        rec.retrieved_chunk_ids = [r.payload.get("chunk_id", "") for r in ranked]
        rec.top_cps = round(ranked[0].cps_score or 0.0, 4) if ranked else 0.0
        rec.contexts = [r.payload.get("raw_text", "") for r in ranked[:self.top_k_ctx]]
        rec.retrieval_latency_s = round(pf["retrieval_base_s"] + (clock() - t0), 4)

        produces_cps = config.reranker in ("cps", "cps_then_reranker")

        if (produces_cps and settings.cps_require_semantic_match
                and not cps.has_semantic_match(ranked, self.top_k_ctx)):
            rec.refusal_reason = "no_semantic_match"
            rec.total_latency_s = rec.retrieval_latency_s
            return rec

        if produces_cps:
            if settings.gate_ignores_semester:
                gw = cps.CPSWeights.default()
                gate_cps = max(
                    (cps.gate_score(r, student, gw, self.graph) for r in ranked),
                    default=0.0,
                )
            else:
                gate_cps = rec.top_cps
            if gate_cps < threshold:
                rec.refusal_reason = "insufficient_context"
                rec.total_latency_s = rec.retrieval_latency_s
                return rec
        if retrieval_only:
            rec.refused = False
            rec.total_latency_s = rec.retrieval_latency_s
            return rec

        chunks_for_gen = [{**r.payload, "cps_score": r.cps_score} for r in ranked]
        if mode == "strict":
            result = generate_strict(query["query_text"], chunks_for_gen, self.ollama, top_k=3)
        else:
            result = generate_socratic(query["query_text"], chunks_for_gen, self.ollama, top_k=10)
        answer = result.get("answer", "")
        verify_this_mode = mode == "strict" or settings.socratic_grounding_check
        if answer and self.verify_fn is not None and verify_this_mode:
            if not self.verify_fn(answer, result.get("context_used", ""), self.ollama, mode):
                rec.refusal_reason = "ungrounded_answer"
                rec.total_latency_s = round(clock() - t0 + pf["retrieval_base_s"], 4)
                return rec
        rec.answer = answer
        rec.refused = not answer
        rec.refusal_reason = "" if answer else "empty_generation"
        rec.total_latency_s = round(clock() - t0 + pf["retrieval_base_s"], 4)
        return rec

    def _route(self, dense, class_label: str) -> str:
        if self.router is not None:
            try:
                return self.router.route_embedded(dense)[0]
            except Exception:
                pass
        return "socratic" if class_label == "adversarial_routing" else "strict"

    def _student(self, query) -> StudentContext:
        p = query["student_profile"]
        return StudentContext(user_id="eval", department_id=p["department_id"],
                              current_semester=p["current_semester"],
                              enrolled_courses=p.get("enrolled_courses", []))
