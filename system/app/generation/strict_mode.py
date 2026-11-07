from app.generation.prompts import STRICT_SYSTEM_RULES, STRICT_USER_TEMPLATE

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

def _build_citation(chunks: list[dict]) -> tuple[str, list[str]]:
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

def generate_strict(
        query: str,
        retrieved_chunks: list[dict],
        ollama_client,
        top_k:int = 3,
) -> dict:
    selected = retrieved_chunks[:top_k]

    if not selected:
        return {
            "answer": "Saya tidak memiliki informasi tersebut dalam basis kurikulum",
            "citations": [],
            "context_used": [],
        }

    context_block, raw_texts = _format_context_block(selected)

    system_msg = STRICT_SYSTEM_RULES.format(context=context_block)
    user_msg = STRICT_USER_TEMPLATE.format(query=query)

    answer, usage = ollama_client.generate_with_usage(
        prompt=user_msg,
        system=system_msg,
        temperature=0.1,
        max_tokens=2048,
    )

    return {
        "answer": answer,
        "citations": _build_citation(selected),
        "context_used": raw_texts,
        "usage": usage,
    }
