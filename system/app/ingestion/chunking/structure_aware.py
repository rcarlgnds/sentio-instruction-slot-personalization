from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.domain.models import ParsedDocument, Section, Chunk
from app.ingestion.chunking.base import BaseChunker

from transformers import AutoTokenizer

class StructureAwareChunker(BaseChunker):
    def __init__(
        self,
        child_size: int = None,
        parent_size: int = None,
        overlap: int = None,
        min_chars: int = None,
    ):
        self.child_size = child_size or settings.chunk_child_size
        self.parent_size = parent_size or settings.chunk_parent_size
        self.overlap = overlap or settings.chunk_overlap
        self.min_chars = min_chars or settings.min_chunk_chars

        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_name)

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size * 4,
            chunk_overlap=self.overlap * 4,
            separators=["\n\n", "\n", ". ", " "],
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size * 4,
            chunk_overlap=self.overlap * 4,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks = self._chunk_sections(doc)

        title = (doc.title or "").strip()
        for c in chunks:
            crumb = " > ".join(p for p in (title, c.section_path) if p)
            if crumb:
                c.text_for_embedding = f"[{crumb}]\n{c.raw_text}"
        return chunks

    def _chunk_sections(self, doc: ParsedDocument) -> list[Chunk]:
        chunks = []

        for section in doc.sections:
            if not section.text or len(section.text.strip()) < self.min_chars:
                continue

            token_count = self._count_tokens(section.text)
            section_path = section.title
            position = self._build_position(section)

            if token_count <= self.child_size:
                child = Chunk.new(
                    raw_text=section.text,
                    text_for_embedding=section.text,
                    parent_chunk_id=None,
                    chunk_level="child",
                    section_path=section_path,
                    position=position,
                    token_count=token_count,
                )
                chunks.append(child)

            elif token_count <= self.parent_size:
                chunks.extend(self._create_parent_with_children(section.text, section_path, position))

            else:
                parent_texts = self.parent_splitter.split_text(section.text)
                for pt in parent_texts:
                    chunks.extend(self._create_parent_with_children(pt, section_path, position))

        return chunks

    def _build_position(self, section: Section) -> dict:
        pos = {"page": int(section.metadata.get("page", 0) or 0)}
        if "page_end" in section.metadata:
            pos["page_end"] = int(section.metadata["page_end"])
        return pos

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _build_section_path(self, section: Section, ancestors: list[Section]) -> str:
        return " > ".join([s.title for s in ancestors] + [section.title])

    def _create_parent_with_children(self, text: str, section_path: str, position: dict) -> list[Chunk]:
        chunks = []

        parent = Chunk.new(
            raw_text=text,
            text_for_embedding=text,
            parent_chunk_id=None,
            chunk_level="parent",
            section_path=section_path,
            position=dict(position),
            token_count=self._count_tokens(text),
        )
        chunks.append(parent)

        child_texts = self.child_splitter.split_text(text)
        for ct in child_texts:
            child = Chunk.new(
                raw_text=ct,
                text_for_embedding=ct,
                parent_chunk_id=parent.chunk_id,
                chunk_level="child",
                section_path=section_path,
                position=dict(position),
                token_count=self._count_tokens(ct),
            )
            chunks.append(child)

        return chunks
