from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
from uuid import uuid4


@dataclass
class IngestionJob:
    job_id: str
    document_id: str
    object_key: str
    course_id: str
    department_id: str
    uploaded_by: str
    created_at: str

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionJob":
        return cls(**data)


@dataclass
class Section:
    title: str
    level: int
    text: str
    subsections: list["Section"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def walk(self):
        yield self
        for sub in self.subsections:
            yield from sub.walk()


@dataclass
class ParsedDocument:
    title: str
    sections: list[Section]
    full_text: str
    metadata: dict = field(default_factory=dict)


ChunkLevel = Literal["parent", "child"]


@dataclass
class Chunk:
    chunk_id: str
    raw_text: str
    text_for_embedding: str

    parent_chunk_id: Optional[str]
    chunk_level: ChunkLevel
    section_path: str
    position: dict
    token_count: int
    has_context_prefix: bool = False

    dense_vector: Optional[list[float]] = None
    sparse_indices: Optional[list[int]] = None
    sparse_values: Optional[list[float]] = None

    @classmethod
    def new(cls, **kwargs) -> "Chunk":
        return cls(chunk_id=str(uuid4()), **kwargs)


RoutingMode = Literal["strict", "socratic"]


@dataclass
class StudentContext:
    user_id: str
    department_id: str
    current_semester: int
    enrolled_courses: list[str]
    role: str = "student"


@dataclass
class RetrievalResult:
    chunk_id: str
    payload: dict
    sim_score: float
    cps_score: Optional[float] = None
    sim_norm: Optional[float] = None


@dataclass
class QueryResponse:
    mode: RoutingMode
    answer: str
    citations: list[dict]
    confidence: dict
    refusal: Optional[dict] = None


DocumentStatus = Literal["pending", "processing", "completed", "failed"]


@dataclass
class IngestionStatus:
    document_id: str
    status: DocumentStatus
    chunk_count: int = 0
    error: Optional[str] = None
    updated_at: Optional[str] = None
