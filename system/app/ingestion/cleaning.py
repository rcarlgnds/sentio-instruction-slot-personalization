import re
from collections import Counter

from app.domain.models import Chunk, ParsedDocument
from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_BOILERPLATE_LINE_CHARS = 120
MIN_ABS_FREQUENCY = 8
PAGE_FRACTION = 0.3

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
_PAGE_NUMBER_LINE = re.compile(r"^[\s\d|.•\-–-]+$")


def _line_key(line: str) -> str:
    return _WS.sub(" ", _DIGITS.sub("#", line)).strip().lower()


def _boilerplate_keys(parsed: ParsedDocument, threshold: int) -> set[str]:
    counts: Counter[str] = Counter()
    for section in parsed.sections:
        for line in section.text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > MAX_BOILERPLATE_LINE_CHARS:
                continue
            if stripped.startswith("|"):
                continue
            counts[_line_key(stripped)] += 1
    return {key for key, n in counts.items() if n >= threshold}


def strip_boilerplate(parsed: ParsedDocument) -> tuple[ParsedDocument, int]:
    page_count = int(parsed.metadata.get("page_count") or len(parsed.sections))
    threshold = max(MIN_ABS_FREQUENCY, int(page_count * PAGE_FRACTION))
    noisy = _boilerplate_keys(parsed, threshold)

    removed = 0
    kept_sections = []
    for section in parsed.sections:
        kept_lines = []
        for line in section.text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("|"):
                if _line_key(stripped) in noisy or _PAGE_NUMBER_LINE.match(stripped):
                    removed += 1
                    continue
            kept_lines.append(line)
        section.text = "\n".join(kept_lines).strip()
        if section.text:
            kept_sections.append(section)

    parsed.sections = kept_sections
    parsed.full_text = "\n\n".join(s.text for s in kept_sections)
    if removed:
        log.info(
            f"[CLEANING] strip_boilerplate: {removed} furniture lines dropped "
            f"(threshold {threshold}x, {len(noisy)} patterns)"
        )
    return parsed, removed


def dedup_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    seen: set[str] = set()
    kept: list[Chunk] = []
    dropped_parents: set[str] = set()
    dropped = 0

    for chunk in chunks:
        if chunk.parent_chunk_id and chunk.parent_chunk_id in dropped_parents:
            dropped += 1
            continue
        key = f"{chunk.chunk_level}:{_WS.sub(' ', chunk.raw_text).strip().lower()}"
        if key in seen:
            dropped += 1
            if chunk.chunk_level == "parent":
                dropped_parents.add(chunk.chunk_id)
            continue
        seen.add(key)
        kept.append(chunk)

    if dropped:
        log.info(f"[CLEANING] dedup_chunks: {dropped} duplicate chunks dropped")
    return kept, dropped
