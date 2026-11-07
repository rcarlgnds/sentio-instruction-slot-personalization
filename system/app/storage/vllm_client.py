import requests

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


class VLLMClient:
    def __init__(self):
        self.base_url = settings.vllm_url.rstrip("/")
        self.default_model = settings.vllm_model
        self.context_model = settings.vllm_context_model
        self.timeout = settings.vllm_timeout
        self.headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}

    def _chat_completion(
        self,
        messages: list[dict],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[str, dict]:
        if settings.vllm_temperature_override is not None:
            temperature = settings.vllm_temperature_override

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": settings.vllm_enable_thinking},
        }
        for key, value in (
            ("top_p", settings.vllm_top_p),
            ("top_k", settings.vllm_top_k),
            ("presence_penalty", settings.vllm_presence_penalty),
        ):
            if value is not None:
                payload[key] = value

        if max_tokens is not None:
            floor = settings.vllm_max_tokens_floor
            payload["max_tokens"] = max(max_tokens, floor) if floor else max_tokens

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"] or ""
        raw_usage = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
        }
        return text.strip(), usage

    def generate_with_usage(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> tuple[str, dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat_completion(messages, model, temperature, max_tokens)

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
        text, _ = self._chat_completion(messages, model, temperature, None)
        return text

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def loaded_models(self) -> list[dict]:
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models", headers=self.headers, timeout=5
            )
            resp.raise_for_status()
            entries = resp.json().get("data", []) or []
            return [
                {"name": e.get("id", ""), "size_vram": settings.vllm_vram_bytes}
                for e in entries
            ]
        except Exception:
            return []
