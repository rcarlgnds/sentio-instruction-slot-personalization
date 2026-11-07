import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from app.benchmark.rads_synthesis.models import DIFFICULTIES
from app.config import settings

REQUIRED_KEYS = {"query_text", "expected_answer_summary", "reasoning_chain",
                 "gold_relevant_chunks", "difficulty"}


def validate_output(data: dict, allowed_chunk_ids: set) -> list:
    problems = []
    if not isinstance(data, dict):
        return ["output is not a JSON object"]
    missing = REQUIRED_KEYS - set(data)
    if missing:
        problems.append(f"missing fields: {sorted(missing)}")
        return problems
    if not isinstance(data["query_text"], str) or len(data["query_text"].strip()) < 10:
        problems.append("query_text empty, too short, or not a string")
    if not isinstance(data["expected_answer_summary"], str) or len(data["expected_answer_summary"].strip()) < 10:
        problems.append("expected_answer_summary empty or not a string")
    chain = data["reasoning_chain"]
    if not isinstance(chain, list) or not chain or not all(isinstance(x, str) for x in chain):
        problems.append("reasoning_chain is not a non-empty list of strings")
    if data["difficulty"] not in DIFFICULTIES:
        problems.append(f"invalid difficulty: {data['difficulty']!r}")
    gold = data["gold_relevant_chunks"]
    if not isinstance(gold, list) or not gold or not all(isinstance(x, str) for x in gold):
        problems.append("gold_relevant_chunks empty or not a list of strings")
    elif not set(gold) <= allowed_chunk_ids:
        problems.append("gold_relevant_chunks contains a chunk_id outside the supplied set")
    return problems


def parse_strict_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object in the output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:i + 1])
    raise ValueError("unbalanced JSON object")


class OllamaBackend:
    name = "ollama-dryrun"

    def __init__(self, model: str = None, max_retries: int = 2):
        self.model = model or settings.ollama_model
        self.max_retries = max_retries

    def generate(self, prompt: str, allowed_chunk_ids: set) -> tuple:
        last_problems = []
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call(prompt)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last_problems = [f"network or transport failure: {e}"]
                time.sleep(min(5 * (attempt + 1), 20))
                continue
            try:
                data = parse_strict_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last_problems = [f"JSON parse failure: {e}"]
                continue
            problems = validate_output(data, allowed_chunk_ids)
            if not problems:
                return data, self._meta()
            last_problems = problems
        return None, {"rejected": last_problems, **self._meta()}

    def _call(self, prompt: str) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1024},
                "think": False}
        req = urllib.request.Request(f"{settings.ollama_url}/api/generate",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600)).get("response", "")

    def _meta(self) -> dict:
        return {"model": self.model, "backend": self.name,
                "generated_at": datetime.now(timezone.utc).isoformat()}


class BatchFileBackend:
    name = "fable5-ultracode-batch"
    model = "claude-fable-5"

    @staticmethod
    def emit_batch(items: list, path: str):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def ingest_results(cls, path: str) -> dict:
        out = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = {"model": row.get("model", cls.model), "backend": cls.name,
                        "generated_at": row.get("generated_at", "")}
                data = row.get("output")
                if data is None:
                    out[row["batch_id"]] = (None, {"rejected": ["empty output"], **meta})
                else:
                    out[row["batch_id"]] = (data, meta)
        return out
