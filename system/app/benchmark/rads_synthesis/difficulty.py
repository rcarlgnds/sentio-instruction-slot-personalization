import re

LEVELS = ["easy", "medium", "hard"]
STYLES = ["formal", "informal"]
RETRIEVAL_BUMP_CPS = 0.60
HOP_BUMP = 3
DRIFT_BUMP = 4
DEPTS_BUMP = 3

_INFORMAL_MARKERS = re.compile(
    r"\b(nggak|gak|gimana|kenapa ya|sih|banget|kok|aku|kepake|pengin|kayak|"
    r"biar|bikin|dong|deh|kan)\b", re.IGNORECASE)


def _structural_bump(class_label: str, class_metadata: dict) -> bool:
    meta = class_metadata or {}
    if class_label == "prerequisite_chain_depth":
        return (meta.get("hop_distance") or 0) >= HOP_BUMP
    if class_label == "temporal_scope_drift":
        return abs(meta.get("drift") or 0) >= DRIFT_BUMP
    if class_label == "adversarial_routing":
        return meta.get("query_form", "long") == "short"
    if class_label == "multi_disciplinary_ambiguity":
        return len(meta.get("departments") or []) >= DEPTS_BUMP
    return False


def derive_score_raw(n_gold: int, class_label: str, class_metadata: dict,
                     top_cps: float | None) -> int:
    level = 0 if n_gold <= 1 else 1
    if _structural_bump(class_label, class_metadata):
        level += 1
    if top_cps is not None and top_cps < RETRIEVAL_BUMP_CPS:
        level += 1
    return level


def derive_difficulty(n_gold: int, class_label: str, class_metadata: dict,
                      top_cps: float | None) -> str:
    raw = derive_score_raw(n_gold, class_label, class_metadata, top_cps)
    return LEVELS[min(raw, 2)]


def derive_for_query(query) -> str:
    return derive_difficulty(
        n_gold=len(query.gold_relevant_chunks or []),
        class_label=query.class_label,
        class_metadata=query.class_metadata or {},
        top_cps=(query.qc or {}).get("top_cps"),
    )


def derive_raw_for_query(query) -> int:
    return derive_score_raw(
        n_gold=len(query.gold_relevant_chunks or []),
        class_label=query.class_label,
        class_metadata=query.class_metadata or {},
        top_cps=(query.qc or {}).get("top_cps"),
    )


def classify_style(query_text: str) -> str:
    return "informal" if _INFORMAL_MARKERS.search(query_text) else "formal"
