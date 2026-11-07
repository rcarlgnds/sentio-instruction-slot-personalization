import uuid
from dataclasses import dataclass, field, asdict

CLASS_LABELS = [
    "multi_disciplinary_ambiguity",
    "prerequisite_chain_depth",
    "temporal_scope_drift",
    "adversarial_routing",
]

DIFFICULTIES = ["easy", "medium", "hard"]


@dataclass
class SourceChunk:
    chunk_id: str
    course_id: str
    course_code: str
    course_name: str
    department_id: str
    department_name: str
    semester: int
    raw_text: str


@dataclass
class SelectionBundle:
    class_label: str
    chunks: list
    class_metadata: dict = field(default_factory=dict)


@dataclass
class GenerationMeta:
    model: str
    template_version: str
    backend: str
    seed: int
    generated_at: str


@dataclass
class BenchmarkQuery:
    query_id: str
    class_label: str
    difficulty: str
    query_text: str
    expected_answer_summary: str
    reasoning_chain: list
    gold_relevant_chunks: list
    student_profile: dict
    class_metadata: dict
    generation: dict
    qc: dict = field(default_factory=dict)
    split: str = ""
    linguistic_style: str = ""
    difficulty_score_raw: int = -1
    annotation_status: str = "synthetic_unvalidated"

    @classmethod
    def new(cls, **kwargs) -> "BenchmarkQuery":
        kwargs.setdefault("query_id", str(uuid.uuid4()))
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
