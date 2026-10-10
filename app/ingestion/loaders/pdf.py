"""PDF loader built on pdfplumber.

Structure recovery:
  * tables  -> `page.find_tables()`, emitted as TABLE elements (Markdown + rows)
  * headings -> lines whose font size is clearly above the document's body size
  * lists   -> lines starting with bullets / enumerators
  * paragraphs -> consecutive body lines merged until a vertical gap
Text inside table areas is excluded from the running text to avoid duplication.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from app.ingestion.loaders.base import (
    BaseDocumentLoader,
    Element,
    ElementType,
    LoadedDocument,
    LoaderError,
    table_element,
)

_BULLET_RE = re.compile(r"^\s*(?:[•●▪◦\-*–]|\(?\d{1,2}[.)]|[a-z][.)])\s+")


def _inside(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


class PDFLoader(BaseDocumentLoader):
    source_type = "pdf"
    supported_extensions = (".pdf",)

    heading_size_ratio = 1.15
    max_heading_chars = 120

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        import pdfplumber

        path, raw = self._read_file(source)
        try:
            pdf = pdfplumber.open(path)
        except Exception as exc:
            raise LoaderError(f"Could not open PDF {path.name}: {exc}") from exc

        with pdf:
            pages: list[list[tuple[float, Element]]] = []
            line_sizes: list[float] = []
            page_lines: list[list[dict[str, Any]]] = []

            for page_no, page in enumerate(pdf.pages, start=1):
                tables = page.find_tables()
                bboxes = [t.bbox for t in tables]
                positioned: list[tuple[float, Element]] = []
                for table in tables:
                    data = [[(c or "").strip() for c in row] for row in table.extract() if row]
                    if len(data) >= 2:
                        positioned.append((table.bbox[1], table_element(data[0], data[1:], page_number=page_no)))

                text_page = page.filter(
                    lambda obj, b=bboxes: not any(_inside(obj, bb) for bb in b) if "x0" in obj else True
                )
                lines = text_page.extract_text_lines(return_chars=True, strip=True)
                for line in lines:
                    sizes = [c.get("size", 0) for c in line.get("chars", []) if c.get("text", "").strip()]
                    line["size"] = statistics.median(sizes) if sizes else 0
                    if sizes:
                        line_sizes.append(line["size"])
                page_lines.append(lines)
                pages.append(positioned)

            body_size = statistics.median(line_sizes) if line_sizes else 0
            elements: list[Element] = []
            for page_no, (lines, positioned) in enumerate(zip(page_lines, pages, strict=True), start=1):
                positioned = positioned + self._lines_to_elements(lines, body_size, page_no)
                positioned.sort(key=lambda p: p[0])
                elements.extend(e for _, e in positioned)

            meta = pdf.metadata or {}
            title = options.get("title") or meta.get("Title") or path.stem.replace("_", " ")
            return LoadedDocument(
                title=str(title),
                source_type=self.source_type,
                source_name=path.name,
                file_path=str(path),
                raw_content=raw,
                elements=elements,
                metadata={
                    "file_name": path.name,
                    "page_count": len(pdf.pages),
                    "author": meta.get("Author"),
                    "date": meta.get("CreationDate"),
                },
            )

    def _lines_to_elements(
        self, lines: list[dict[str, Any]], body_size: float, page_no: int
    ) -> list[tuple[float, Element]]:
        out: list[tuple[float, Element]] = []
        buffer: list[str] = []
        buffer_top: float | None = None
        last_bottom: float | None = None
        list_items: list[str] = []
        list_top: float | None = None

        def flush_paragraph() -> None:
            nonlocal buffer, buffer_top
            if buffer:
                out.append((buffer_top or 0, Element(ElementType.PARAGRAPH, " ".join(buffer), page_number=page_no)))
            buffer, buffer_top = [], None

        def flush_list() -> None:
            nonlocal list_items, list_top
            if list_items:
                out.append((list_top or 0, Element(ElementType.LIST, "\n".join(list_items), page_number=page_no)))
            list_items, list_top = [], None

        for line in lines:
            text = line.get("text", "").strip()
            if not text:
                continue
            gap = (line["top"] - last_bottom) if last_bottom is not None else 0
            line_height = max(line["bottom"] - line["top"], 1)
            last_bottom = line["bottom"]

            is_heading = (
                body_size > 0
                and line["size"] >= body_size * self.heading_size_ratio
                and len(text) <= self.max_heading_chars
            )
            if is_heading:
                flush_paragraph()
                flush_list()
                level = 1 if line["size"] >= body_size * 1.5 else 2
                out.append((line["top"], Element(ElementType.HEADING, text, level=level, page_number=page_no)))
                continue

            if _BULLET_RE.match(text):
                flush_paragraph()
                if list_top is None:
                    list_top = line["top"]
                list_items.append("- " + _BULLET_RE.sub("", text, count=1))
                continue

            if list_items and gap < line_height * 0.8 and not buffer:
                list_items[-1] += " " + text  # wrapped list item
                continue
            flush_list()

            if buffer and gap > line_height * 1.2:
                flush_paragraph()
            if buffer_top is None:
                buffer_top = line["top"]
            buffer.append(text)

        flush_paragraph()
        flush_list()
        return out
