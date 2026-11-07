import hashlib
import json
from pathlib import Path

BEHAVIOR_FLAGS = [
    "gate_ignores_mode",
    "gate_ignores_semester",
    "clarify_on_refusal",
    "slang_normalize_main",
    "citation_strip_untraceable",
    "lexical_evidence_gate",
    "socratic_verifier_risk_gate",
    "verifier_refuse_max_sim",
    "route_definitional_strict",
    "definitional_uses_socratic_verifier",
    "junk_chunk_filter",
    "crosslingual_rescue",
    "verifier_flag_mode",
    "socratic_top_k",
    "qdrant_exact_search",
    "qdrant_rrf_k",
    "vllm_enable_thinking",
    "vllm_temperature_override",
    "vllm_top_p",
    "vllm_top_k",
    "vllm_presence_penalty",
    "vllm_max_tokens_floor",
    "ood_query_expansion",
    "cps_require_semantic_match",
    "lexical_evidence_min_ratio",
    "lexical_evidence_min_terms",
    "retrieval_pool_size",
]

_GENERATION_DIR = Path(__file__).resolve().parent / "generation"


def prompt_fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted(_GENERATION_DIR.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def build_static_fingerprint(settings) -> dict:
    centroids_file = Path(settings.routing_centroids_path)
    centroids_hash = (
        hashlib.sha256(centroids_file.read_bytes()).hexdigest()[:12]
        if centroids_file.exists() else "builtin"
    )
    backend = settings.llm_backend.strip().lower()
    active_llm = settings.vllm_model if backend == "vllm" else settings.ollama_model
    return {
        "weights": [settings.cps_alpha, settings.cps_beta, settings.cps_gamma,
                    settings.cps_dept_match_default],
        "gates": [settings.strict_min_cps, settings.socratic_min_cps],
        "routing": [settings.routing_threshold, settings.routing_gap_margin],
        "routing_centroids": centroids_hash,
        "embedding": settings.embedding_model_name,
        "llm": f"{backend}:{active_llm}",
        "flags": {name: getattr(settings, name, None) for name in BEHAVIOR_FLAGS},
        "prompts": prompt_fingerprint(),
    }


def compute_config_version(static_fp: dict, bounds_key: tuple, relations_fp: str) -> str:
    fp = {**static_fp, "bounds": list(bounds_key), "dept_relations": relations_fp}
    return hashlib.sha256(
        json.dumps(fp, sort_keys=True).encode()
    ).hexdigest()[:12]
