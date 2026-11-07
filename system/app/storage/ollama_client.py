import requests

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)

class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_url.rstrip("/")
        self.default_model = settings.ollama_model
        self.context_model = settings.ollama_context_model
        self.timeout = settings.ollama_timeout

    def generate_with_usage(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> tuple[str, dict]:
        payload = {
               "model": model or self.default_model,
               "prompt": prompt,
               "stream": False,
               "think": False,
               "options": {
                   "temperature": temperature,
                   "num_predict": max_tokens,
               },
           }

        if system:
            payload["system"] = system

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )

        resp.raise_for_status()
        data = resp.json()
        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(data.get("eval_count", 0) or 0),
        }
        return data["response"].strip(), usage

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        text, _ = self.generate_with_usage(
            prompt, model=model, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )
        return text

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.3) -> str:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )

        resp.raise_for_status()

        return resp.json()["message"]["content"]

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def loaded_models(self) -> list[dict]:
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=5)
            resp.raise_for_status()
            return resp.json().get("models", []) or []
        except Exception:
            return []
