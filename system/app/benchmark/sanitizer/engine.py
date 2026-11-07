import re
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.benchmark.sanitizer.recognizers import custom_recognizers

DOC_EMAIL_DOMAIN = re.compile(r"@(?:[\w.-]+\.)?(?:example\.(?:com|org|net)|example|invalid|test|localhost)$", re.IGNORECASE)
INFRA_EMAIL_LOCAL = {"git", "ssh", "root"}

PLACEHOLDERS = {
    "ID_NIM": "<NIM>",
    "ID_NIK": "<NIK>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "ID_PHONE_NUMBER": "<PHONE>",
    "PERSON": "<PERSON>",
}

DEFAULT_ENTITIES = ["ID_NIM", "ID_NIK", "EMAIL_ADDRESS", "ID_PHONE_NUMBER"]


@dataclass
class Finding:
    entity_type: str
    start: int
    end: int
    score: float
    text: str


@dataclass
class SanitizeResult:
    clean_text: str
    findings: list = field(default_factory=list)

    @property
    def is_dirty(self) -> bool:
        return len(self.findings) > 0


class ChunkSanitizer:
    def __init__(self, with_person: bool = False, score_threshold: float = 0.5):
        self.score_threshold = score_threshold
        self.entities = list(DEFAULT_ENTITIES) + (["PERSON"] if with_person else [])
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        for rec in custom_recognizers():
            self._analyzer.registry.add_recognizer(rec)
        self._anonymizer = AnonymizerEngine()

    def _analyze_raw(self, text: str) -> list:
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        results = [r for r in results if not self._is_non_pii_email(text, r)]
        by_span = {}
        for r in results:
            key = (r.start, r.end)
            if key not in by_span or r.score > by_span[key].score:
                by_span[key] = r
        return sorted(by_span.values(), key=lambda r: r.start)

    @staticmethod
    def _is_non_pii_email(text: str, result) -> bool:
        if result.entity_type != "EMAIL_ADDRESS":
            return False
        value = text[result.start:result.end]
        local = value.split("@", 1)[0].lower()
        return local in INFRA_EMAIL_LOCAL or bool(DOC_EMAIL_DOMAIN.search(value))

    def analyze(self, text: str) -> list:
        return [
            Finding(r.entity_type, r.start, r.end, round(r.score, 3), text[r.start:r.end])
            for r in self._analyze_raw(text)
        ]

    def sanitize(self, text: str) -> SanitizeResult:
        raw_results = self._analyze_raw(text)
        if not raw_results:
            return SanitizeResult(clean_text=text)
        operators = {
            entity: OperatorConfig("replace", {"new_value": placeholder})
            for entity, placeholder in PLACEHOLDERS.items()
        }
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=raw_results, operators=operators)
        findings = [
            Finding(r.entity_type, r.start, r.end, round(r.score, 3), text[r.start:r.end])
            for r in raw_results
        ]
        return SanitizeResult(clean_text=anonymized.text, findings=findings)
