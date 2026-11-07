"""Curriculum Proximity Scoring (CPS), the reference scorer the paper calls DITS.

Each candidate scores alpha * Sim + beta * Prox + gamma * Struct, with every term between 0 and 1. Sim is
the BGE-M3 similarity to the question, Prox rewards the student's own program and a semester at or below
theirs, and Struct falls with the distance in the prerequisite graph. explain_cps() returns the full
breakdown that the audit trail records.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.domain.models import RetrievalResult, StudentContext
from app.utils.logging import get_logger

log = get_logger(__name__)


_DEPT_RELATIONS: dict[tuple[str, str], float] = {}

_DEPT_PG_ADAPTER = None
_DEPT_LAST_REFRESH: float = 0.0


def _fetch_relations(pg_adapter) -> dict[tuple[str, str], float]:
    return pg_adapter.fetch_department_relations()


def load_department_relations(pg_adapter, enable_auto_reload: bool = False) -> None:
    global _DEPT_RELATIONS, _DEPT_PG_ADAPTER, _DEPT_LAST_REFRESH
    try:
        _DEPT_RELATIONS = _fetch_relations(pg_adapter)
        log.info(f"Loaded {len(_DEPT_RELATIONS)} department relation pairs")
    except Exception as e:
        log.warning(f"Failed to load department relations, falling back to empty dict: {e}")
        _DEPT_RELATIONS = {}
    _DEPT_PG_ADAPTER = pg_adapter if enable_auto_reload else None
    _DEPT_LAST_REFRESH = time.monotonic()


def load_department_relations_from_json(path) -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(f"department relations file not found: {p}")
    data = _json.loads(p.read_text(encoding="utf-8"))
    rel = {}
    for row in data["pairs"]:
        rel[(row["a"], row["b"])] = float(row["score"])
        rel[(row["b"], row["a"])] = float(row["score"])
    set_department_relations(rel)


def set_department_relations(relations: dict[tuple[str, str], float]) -> None:
    global _DEPT_RELATIONS, _DEPT_PG_ADAPTER
    _DEPT_RELATIONS = relations
    _DEPT_PG_ADAPTER = None


def _maybe_refresh_relations() -> None:
    global _DEPT_RELATIONS, _DEPT_LAST_REFRESH
    if _DEPT_PG_ADAPTER is None:
        return
    now = time.monotonic()
    if now - _DEPT_LAST_REFRESH < settings.cps_dept_relations_ttl_seconds:
        return
    _DEPT_LAST_REFRESH = now
    try:
        fresh = _fetch_relations(_DEPT_PG_ADAPTER)
    except Exception as e:
        log.warning(f"[dept_relations] refresh failed, keeping the old value: {e}")
        return
    if fresh != _DEPT_RELATIONS:
        log.info(
            f"[dept_relations] changed: {len(_DEPT_RELATIONS)} -> {len(fresh)} pairs, "
            f"applied without a restart"
        )
        _DEPT_RELATIONS = fresh


def has_semantic_match(results, top_k: int) -> bool:
    return any((r.sim_norm or 0.0) > 0.0 for r in results[:top_k])


def relations_fingerprint() -> str:
    if not _DEPT_RELATIONS:
        return "empty"
    payload = json.dumps(
        sorted((a, b, round(float(v), 4)) for (a, b), v in _DEPT_RELATIONS.items())
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def dept_match_score(chunk_dept: str, student_dept: str) -> float:
    if chunk_dept == student_dept:
        return 1.0
    _maybe_refresh_relations()
    pair = tuple(sorted([chunk_dept, student_dept]))
    return _DEPT_RELATIONS.get(pair, settings.cps_dept_match_default)


@dataclass
class CPSWeights:
    alpha: float
    beta: float
    gamma: float

    @classmethod
    def default(cls) -> "CPSWeights":
        return cls(settings.cps_alpha, settings.cps_beta, settings.cps_gamma)


def gate_score(
    result: RetrievalResult,
    student: StudentContext,
    weights: CPSWeights,
    prerequisite_graph: Optional[object] = None,
) -> float:
    e = explain_cps(result, student, weights, prerequisite_graph,
                    neutralize_semester=True)
    return e["cps"]


def is_ahead_of_student(result: RetrievalResult, student: StudentContext) -> int:
    chunk_semester = result.payload.get("semester") or student.current_semester
    return max(0, chunk_semester - student.current_semester)


def explain_cps(
    result: RetrievalResult,
    student: StudentContext,
    weights: CPSWeights,
    prerequisite_graph: Optional[object] = None,
    concept_graph: Optional[object] = None,
    neutralize_semester: bool = False,
) -> dict:
    sim = result.sim_norm if result.sim_norm is not None else 0.0

    dm = dept_match_score(
        result.payload.get("department_id", ""),
        student.department_id,
    )
    if getattr(student, "role", "student") == "student":
        chunk_semester = result.payload.get("semester") or student.current_semester
        delta = chunk_semester - student.current_semester
        sem_diff = abs(delta)
        asym = settings.cps_prox_semester_mode == "asymmetric" and (
            settings.cps_prox_asym_cross_dept or dm >= 1.0
        )
        if neutralize_semester:
            sem_prox = 1.0
        elif asym:
            sem_prox = 1.0 if delta <= 0 else 1.0 / (1.0 + delta)
        else:
            sem_prox = 1.0 / (1.0 + sem_diff)
    else:
        sem_diff = 0
        sem_prox = 1.0
    prox = dm * sem_prox

    struct_level = "course"
    if concept_graph is not None:
        dist = concept_graph.struct_distance(result, student)
        struct = 1.0 / (1.0 + dist) if dist is not None else 0.0
        struct_level = "concept"
    elif prerequisite_graph is None:
        dist = None
        struct = 0.0
    else:
        course_id = result.payload.get("course_id")
        dist = prerequisite_graph.shortest_path(course_id, student.enrolled_courses)
        struct = 1.0 / (1.0 + dist) if dist is not None else 0.0

    return {
        "raw_cosine": result.sim_score,
        "sim_norm": sim,
        "sem_diff": sem_diff,
        "dept_match": dm,
        "prox": prox,
        "graph_distance": dist,
        "struct": struct,
        "struct_level": struct_level,
        "cps": weights.alpha * sim + weights.beta * prox + weights.gamma * struct,
    }


def compute_cps_score(
    result: RetrievalResult,
    student: StudentContext,
    weights: CPSWeights,
    prerequisite_graph: Optional[object] = None,
    concept_graph: Optional[object] = None,
) -> float:
    return explain_cps(result, student, weights, prerequisite_graph, concept_graph)["cps"]


def rerank(
    results: list[RetrievalResult],
    student: StudentContext,
    weights: Optional[CPSWeights] = None,
    prerequisite_graph: Optional[object] = None,
    concept_graph: Optional[object] = None,
) -> list[RetrievalResult]:
    w = weights or CPSWeights.default()
    for r in results:
        if settings.cps_zero_sim_not_candidate and not (r.sim_norm or 0.0) > 0.0:
            r.cps_score = 0.0
            continue
        r.cps_score = compute_cps_score(r, student, w, prerequisite_graph, concept_graph)
    return sorted(results,
                  key=lambda r: (-(r.cps_score or 0.0),
                                 (r.payload or {}).get("chunk_id", "")))
