from qdrant_client.http import models as qm
from app.domain.models import RoutingMode, StudentContext


def build_payload_filter(
    mode: RoutingMode,
    student: StudentContext,
    course_filter: str | None = None,
) -> qm.Filter:
    child_only = qm.FieldCondition(
        key="chunk_level",
        match=qm.MatchValue(value="child"),
    )

    if mode not in ("strict", "socratic"):
        raise ValueError(f"Unknown routing mode: {mode}")

    must = [child_only]
    if course_filter:
        must.append(qm.FieldCondition(
            key="course_id",
            match=qm.MatchValue(value=course_filter),
        ))
    return qm.Filter(must=must)
