from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from docling.datamodel.base_models import InputFormat
import torch
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.domain.models import ParsedDocument, Section
from app.ingestion.parsers.base import BaseParser, ParsingError
from app.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _SectionBuilder:
    title: str
    level: int
    page_start: int
    page_end: int
    paragraphs: list[str] = field(default_factory=list)

    def add_paragraph(self, text: str, page: int | None) -> None:
        t = text.strip()
        if not t:
            return
        self.paragraphs.append(t)
        if page is not None:
            self.page_end = max(self.page_end, page)

    def to_section(self) -> Section | None:
        if not self.paragraphs:
            return None
        return Section(
            title=self.title,
            level=self.level,
            text="\n\n".join(self.paragraphs),
            metadata={"page": self.page_start, "page_end": self.page_end},
        )


class PDFParser(BaseParser):
    SUPPORTED_EXTENSIONS = ("pdf",)

    _converter: DocumentConverter | None = None

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        if cls._converter is None:
            device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
            opts = PdfPipelineOptions()
            opts.do_ocr = False
            opts.do_table_structure = True
            opts.accelerator_options = AcceleratorOptions(device=device)
            cls._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                }
            )
            log.info(f"[PDF] Docling DocumentConverter initialized (ocr=off, tables=on, device={device.value})")
        return cls._converter

    def _convert_bytes(self, file_bytes: bytes):
        converter = self._get_converter()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return converter.convert(tmp.name)

    @staticmethod
    def _repair_pdf(file_bytes: bytes) -> bytes | None:
        try:
            import fitz

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            repaired = doc.tobytes(garbage=3, deflate=True)
            doc.close()
            return repaired
        except Exception as e:
            log.warning(f"[PDF] PyMuPDF repair failed: {e}")
            return None

    def _fallback_flat_parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        sections = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                sections.append(Section(
                    title=f"Halaman {i}",
                    level=1,
                    text=text,
                    metadata={"page": i, "page_end": i},
                ))
        doc.close()

        if not sections:
            raise ParsingError(f"Fallback flat parse: no text in {filename}")

        log.warning(
            f"[PDF] FALLBACK flat parse used for {filename} "
            f"({len(sections)} pages) - heading/table structure not extracted"
        )
        title = os.path.splitext(os.path.basename(filename))[0]
        return ParsedDocument(
            title=title,
            sections=sections,
            full_text="\n\n".join(s.text for s in sections),
            metadata={
                "source_format": "pdf",
                "page_count": len(sections),
                "parser_backend": "pymupdf-fallback",
                "section_count": len(sections),
            },
        )

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        result = None
        try:
            result = self._convert_bytes(file_bytes)
        except Exception as first_err:
            log.warning(f"[PDF] Docling failed for {filename}: {first_err}. Trying PDF repair...")
            repaired = self._repair_pdf(file_bytes)
            if repaired is not None:
                try:
                    result = self._convert_bytes(repaired)
                    log.info(f"[PDF] Repair succeeded - docling runs for {filename}")
                except Exception as second_err:
                    log.warning(f"[PDF] Docling still failed after repair: {second_err}")

        if result is None:
            return self._fallback_flat_parse(file_bytes, filename)

        dl_doc = result.document
        if dl_doc is None:
            raise ParsingError("Docling returned empty document")

        page_count = dl_doc.num_pages()
        doc_title = self._extract_title(dl_doc, filename)

        sections = self._walk_items(dl_doc)

        if not sections:
            raise ParsingError(f"No text extracted from {filename}")

        full_text = "\n\n".join(s.text for s in sections)

        log.info(
            f"[PDF] Docling parse done - {len(sections)} sections, "
            f"{page_count} pages, title='{doc_title[:60]}'"
        )

        return ParsedDocument(
            title=doc_title,
            sections=sections,
            full_text=full_text,
            metadata={
                "source_format": "pdf",
                "page_count": page_count,
                "parser_backend": "docling-2.40",
                "section_count": len(sections),
            },
        )

    @staticmethod
    def _extract_title(dl_doc: Any, filename: str) -> str:
        for item, _ in dl_doc.iterate_items():
            if type(item).__name__ == "SectionHeaderItem":
                t = (getattr(item, "text", "") or "").strip()
                if len(t) > 30:
                    return t
                break
        return filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    @staticmethod
    def _page_of(item: Any) -> int | None:
        prov = getattr(item, "prov", None)
        if not prov:
            return None
        p = getattr(prov[0], "page_no", None)
        return int(p) if p is not None else None

    def _walk_items(self, dl_doc: Any) -> list[Section]:
        path_stack: list[tuple[int, str]] = []
        builders: list[_SectionBuilder] = []
        current: _SectionBuilder | None = None
        preamble_started = False

        for item, _tree_level in dl_doc.iterate_items():
            kind = type(item).__name__
            page = self._page_of(item)

            if kind == "SectionHeaderItem":
                if current is not None:
                    builders.append(current)

                raw_title = (getattr(item, "text", "") or "").strip()
                if not raw_title:
                    current = None
                    continue

                heading_level = int(getattr(item, "level", 1) or 1)
                while path_stack and path_stack[-1][0] >= heading_level:
                    path_stack.pop()

                breadcrumb = " > ".join([t for _, t in path_stack] + [raw_title])
                path_stack.append((heading_level, raw_title))

                start_page = page if page is not None else 1
                current = _SectionBuilder(
                    title=breadcrumb,
                    level=heading_level,
                    page_start=start_page,
                    page_end=start_page,
                )
                continue

            body_text = self._render_body_item(item, kind, dl_doc)
            if not body_text:
                continue

            if current is None:
                if not preamble_started:
                    current = _SectionBuilder(
                        title="Preamble",
                        level=1,
                        page_start=page if page is not None else 1,
                        page_end=page if page is not None else 1,
                    )
                    preamble_started = True
            current.add_paragraph(body_text, page)

        if current is not None:
            builders.append(current)

        out: list[Section] = []
        for b in builders:
            sec = b.to_section()
            if sec is not None:
                out.append(sec)
        return out

    @staticmethod
    def _render_body_item(item: Any, kind: str, dl_doc: Any) -> str:
        if kind in ("PictureItem", "PageHeaderItem", "PageFooterItem"):
            return ""

        if kind == "TableItem":
            try:
                md = item.export_to_markdown(doc=dl_doc)
            except TypeError:
                md = item.export_to_markdown()
            except Exception as e:
                log.debug(f"[PDF] Table markdown export failed: {e}")
                return ""
            return md.strip()

        if kind == "ListItem":
            t = (getattr(item, "text", "") or "").strip()
            return f"- {t}" if t else ""

        t = getattr(item, "text", None)
        if isinstance(t, str):
            return t.strip()
        return ""
