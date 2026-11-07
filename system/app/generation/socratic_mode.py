import logging

from requests.exceptions import HTTPError

from app.generation.prompts import SOCRATIC_SYSTEM_RULES, SOCRATIC_USER_TEMPLATE

log = logging.getLogger(__name__)

MAX_ANSWER_TOKENS = 1536


def _shrinkage_plan(n: int) -> list[int]:
    order = [n]
    while order[-1] > 1:
        order.append(max(1, order[-1] // 2))
    return order


def _format_context_block(chunks: list[dict]) -> tuple[str, list[str]]:
    blocks = []
    raw_texts = []
    for i, payload in enumerate(chunks, start=1):
        text = payload.get("parent_text") or payload.get("raw_text", "")
        title = payload.get("document_title", "Unknown")
        section = payload.get("section_path", "")
        header = f"[Sumber {i}: {title} - {section}]"
        blocks.append(f"{header}\n{text}")
        raw_texts.append(text)
    return "\n\n---\n\n".join(blocks), raw_texts


def _build_citations(chunks: list[dict]) -> list[dict]:
    citations, seen = [], set()
    for payload in chunks:
        key = (payload.get("document_title", ""),
                 payload.get("section_path", ""),
                 payload.get("position", {}).get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "document_title": payload.get("document_title", ""),
            "section": payload.get("section_path", ""),
            "page": payload.get("position", {}).get("page"),
            "course": payload.get("course_name", ""),
            "cps_score": payload.get("cps_score", 0.0),
            "document_id": payload.get("document_id", ""),
        })
    return citations


def generate_socratic(
    query: str,
    retrieved_chunks: list[dict],
    ollama_client,
    top_k: int = 10,
) -> dict:
    selected = retrieved_chunks[:top_k]

    if not selected:
        return {
            "answer": "Saya belum menemukan materi kurikulum yang relevan dengan pertanyaan ini. Coba spesifikasikan mata kuliah atau topik lebih detail.",
            "citations": [],
            "context_used": [],
        }

    user_msg = SOCRATIC_USER_TEMPLATE.format(query=query)

    answer = usage = None
    for n_chunk in _shrinkage_plan(len(selected)):
        used = selected[:n_chunk]
        context_block, raw_texts = _format_context_block(used)
        try:
            answer, usage = ollama_client.generate_with_usage(
                prompt=user_msg,
                system=SOCRATIC_SYSTEM_RULES.format(context=context_block),
                temperature=0.5,
                max_tokens=MAX_ANSWER_TOKENS,
            )
        except HTTPError as e:
            if getattr(e.response, "status_code", None) != 400 or n_chunk <= 1:
                raise
            log.warning(f"[socratic] prompt rejected at {n_chunk} chunks, shrinking")
            continue
        selected = used
        break

    return {
        "answer": answer,
        "citations": _build_citations(selected),
        "context_used": raw_texts,
        "usage": usage,
    }
