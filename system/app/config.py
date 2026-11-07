from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "sentio-ai-service"
    service_version: str = "0.2.0"
    environment: str = Field(default="development")

    host: str = "0.0.0.0"
    port: int = 8000

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sentio_knowledge"
    qdrant_timeout: int = 30
    qdrant_exact_search: bool = True
    qdrant_rrf_k: int = 60

    llm_backend: str = "vllm"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_timeout: int = 120
    ollama_context_model: str = "qwen3:14b"

    vllm_url: str = "http://localhost:8001"
    vllm_model: str = "Qwen/Qwen3-14B-AWQ"
    vllm_timeout: int = 120
    vllm_context_model: str = "Qwen/Qwen3-14B-AWQ"
    vllm_api_key: str = ""
    vllm_vram_bytes: int = 14417920000

    junk_chunk_filter: bool = False

    vllm_enable_thinking: bool = False
    vllm_temperature_override: float | None = None
    vllm_top_p: float | None = None
    vllm_top_k: int | None = None
    vllm_presence_penalty: float | None = None
    vllm_max_tokens_floor: int | None = None

    gate_ignores_semester: bool = True

    gate_ignores_mode: bool = True

    advanced_material_ratio: float = 0.5

    cps_zero_sim_not_candidate: bool = True

    cps_require_semantic_match: bool = True

    socratic_grounding_check: bool = True

    anchor_check_enabled: bool = True

    socratic_verifier_risk_gate: bool = True
    socratic_verifier_risk_threshold: float = 0.25

    definitional_uses_socratic_verifier: bool = True

    lexical_evidence_gate: bool = True
    lexical_evidence_min_ratio: float = 0.7
    lexical_evidence_min_terms: int = 2

    citation_strip_untraceable: bool = True

    ood_query_expansion: bool = True

    topic_low_sim_warning: float = 0.30

    route_definitional_strict: bool = True

    socratic_top_k: int = 6

    slang_normalize_main: bool = True

    verifier_flag_mode: bool = False
    verifier_refuse_max_sim: float = 0.25

    verifier_flag_requires_evidence: bool = True

    clarify_on_refusal: bool = True
    clarify_reasons: tuple = ("insufficient_context", "no_semantic_match")

    crosslingual_rescue: bool = False
    crosslingual_slang_reembed: bool = True
    crosslingual_translate_rescue: bool = True
    crosslingual_translate_max_tokens: int = 80

    cps_dept_relations_ttl_seconds: int = 30

    redis_url: str = "redis://localhost:6379"
    redis_queue_name: str = "sentio:ingestion:queue"
    redis_dlq_name: str = "sentio:ingestion:dlq"
    redis_status_key_prefix: str = "sentio:doc:status:"
    redis_timeout: int = 60

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_bucket: str = "sentio-documents"

    database_url: str = "postgresql://localhost:5432/sentio"

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_fp16: bool = False
    embedding_batch_size: int = 8

    chunk_child_size: int = 256
    chunk_parent_size: int = 768
    chunk_overlap: int = 32
    min_chunk_chars: int = 50

    contextual_retrieval_enabled: bool = True
    contextual_max_doc_tokens: int = 8000
    contextual_timeout: int = 30
    contextual_temperature: float = 0.3
    contextual_max_response_tokens: int = 150

    cps_alpha: float = 0.5
    cps_beta: float = 0.3
    cps_gamma: float = 0.2
    cps_dept_match_default: float = 0.5
    cps_prox_semester_mode: str = "asymmetric"
    cps_prox_asym_cross_dept: bool = True
    retrieval_pool_size: int = 80


    routing_threshold: float = 0.60
    routing_fallback_mode: str = "strict"
    routing_gap_margin: float = 0.10
    routing_centroids_path: str = "routing_centroids.json"

    strict_min_cps: float = 0.55
    strict_min_sim: float = 0.45
    socratic_min_cps: float = 0.35
    socratic_min_sim: float = 0.25
    socratic_semester_window: int = 2
    calibration_path: str = "calibration.json"

    worker_poll_timeout: int = 5
    worker_max_retries: int = 3

    log_level: str = "INFO"
    log_json: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
