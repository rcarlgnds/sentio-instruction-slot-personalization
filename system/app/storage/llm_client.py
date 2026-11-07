from app.config import settings
from app.storage.ollama_client import OllamaClient
from app.storage.vllm_client import VLLMClient
from app.utils.logging import get_logger

log = get_logger(__name__)

LLMClient = OllamaClient | VLLMClient

_BACKENDS = {
    "ollama": OllamaClient,
    "vllm": VLLMClient,
}


def get_llm_client(backend: str | None = None) -> LLMClient:
    name = (backend or settings.llm_backend).strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM_BACKEND: {name!r}. Choices: {sorted(_BACKENDS)}"
        )
    client = _BACKENDS[name]()
    log.info(f"[llm] backend={name} url={client.base_url} model={client.default_model}")
    return client


def active_backend() -> str:
    return settings.llm_backend.strip().lower()
