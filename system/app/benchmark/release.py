import hashlib
import re

QUALITY_PATTERNS = {
    "glyph_artifact": re.compile(r"GLYPH<"),
    "toc_dot_leader": re.compile(r"\.{8,}"),
    "oldstyle_glyphname": re.compile(r"/[a-z]+\.(Alt\.)?oldstyle|\.sc\b"),
    "spaced_letters": re.compile(r"\b\w \w \w \w \w\b"),
}
TEXT_PREFIX_LEN = 200
SHINGLE_SIZE = 5
TIER3_MIN_JACCARD = 0.5


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def slug_for(file_name: str) -> str:
    stem = re.sub(r"\.pdf$", "", (file_name or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def canonical_id(slug: str, chunk_index: int) -> str:
    return f"{slug}-{int(chunk_index):05d}"


def quality_flags(text: str) -> list:
    flags = [name for name, pat in QUALITY_PATTERNS.items() if pat.search(text or "")]
    if (text or "").count("|") > 30:
        flags.append("table_pipe_heavy")
    if len((text or "").strip()) < 80:
        flags.append("very_short")
    return sorted(flags)


def build_anchor(payload: dict, slug: str) -> dict:
    text = payload.get("raw_text") or ""
    position = payload.get("position") or {}
    return {
        "canonical_id": canonical_id(slug, payload.get("chunk_index") or 0),
        "document_title": payload.get("document_title") or "",
        "page": position.get("page"),
        "page_end": position.get("page_end"),
        "section_path": payload.get("section_path") or "",
        "text_prefix": normalize_text(text)[:TEXT_PREFIX_LEN],
        "text_sha256": text_sha256(text),
    }


def enrolled_for(dept_courses: list, department_id: str, current_semester: int,
                 existing: list | None = None) -> list:
    taken = {
        c["id"] for c in dept_courses
        if c.get("department_id") == department_id
        and (c.get("semester") or 99) <= (current_semester or 0)
    }
    taken.update(existing or [])
    return sorted(taken)


def _shingles(text: str) -> set:
    words = normalize_text(text).lower().split()
    if len(words) < SHINGLE_SIZE:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + SHINGLE_SIZE])
            for i in range(len(words) - SHINGLE_SIZE + 1)}


class AnchorResolver:
    def __init__(self, corpus_rows):
        self.by_sha = {}
        self.by_doc = {}
        for row in corpus_rows:
            meta = row.get("metadata") or {}
            text = row.get("text") or ""
            entry = {
                "id": row.get("_id"),
                "text": text,
                "norm": normalize_text(text),
                "sha": text_sha256(text),
                "doc": meta.get("document_title") or "",
                "page": meta.get("page"),
            }
            self.by_sha.setdefault(entry["sha"], []).append(entry)
            self.by_doc.setdefault(entry["doc"], []).append(entry)

    def resolve(self, anchor: dict) -> tuple:
        sha_hits = self.by_sha.get(anchor.get("text_sha256") or "", [])
        if len(sha_hits) == 1:
            return sha_hits[0]["id"], "exact_hash"
        if len(sha_hits) > 1:
            for hit in sha_hits:
                if hit["doc"] == anchor.get("document_title"):
                    return hit["id"], "exact_hash"
            return sha_hits[0]["id"], "exact_hash"

        candidates = self.by_doc.get(anchor.get("document_title") or "", [])
        prefix = anchor.get("text_prefix") or ""
        page = anchor.get("page")
        if prefix:
            for cand in candidates:
                page_ok = (page is None or cand["page"] is None
                           or abs((cand["page"] or 0) - page) <= 1)
                if page_ok and prefix in cand["norm"]:
                    return cand["id"], "doc_page_prefix"

        target = _shingles(prefix)
        best_id, best_score = None, 0.0
        for cand in candidates:
            cand_sh = _shingles(cand["text"][:4 * TEXT_PREFIX_LEN])
            if not target or not cand_sh:
                continue
            score = len(target & cand_sh) / len(target | cand_sh)
            if score > best_score:
                best_id, best_score = cand["id"], score
        if best_score >= TIER3_MIN_JACCARD:
            return best_id, "shingle_overlap"
        return None, "unresolved"
