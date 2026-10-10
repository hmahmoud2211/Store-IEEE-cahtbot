"""DOCX loader: walks the document body in order, preserving headings, lists and tables."""

from __future__ import annotations

import re
from typing import Any

from app.ingestion.loaders.base import (
    BaseDocumentLoader,
    Element,
    ElementType,
    LoadedDocument,
    LoaderError,
    table_element,
)

_HEADING_RE = re.compile(r"^heading\s*(\d)", re.IGNORECASE)


class DOCXLoader(BaseDocumentLoader):
    source_type = "docx"
    supported_extensions = (".docx",)

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        path, raw = self._read_file(source)
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise LoaderError(f"Could not open DOCX {path.name}: {exc}") from exc

        elements: list[Element] = []
        list_buffer: list[str] = []

        def flush_list() -> None:
            if list_buffer:
                elements.append(Element(ElementType.LIST, "\n".join(list_buffer)))
                list_buffer.clear()

        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                para = Paragraph(child, document)
                text = para.text.strip()
                if not text:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                heading = _HEADING_RE.match(style)
                is_list = "list" in style.lower() or child.pPr is not None and child.pPr.numPr is not None
                if heading or style.lower() == "title":
                    flush_list()
                    level = int(heading.group(1)) if heading else 1
                    elements.append(Element(ElementType.HEADING, text, level=level))
                elif is_list:
                    list_buffer.append(f"- {text}")
                else:
                    flush_list()
                    elements.append(Element(ElementType.PARAGRAPH, text))
            elif tag == "tbl":
                flush_list()
                table = Table(child, document)
                data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                data = [row for row in data if any(row)]
                if len(data) >= 2:
                    elements.append(table_element(data[0], data[1:]))
                elif data:
                    elements.append(Element(ElementType.PARAGRAPH, " | ".join(data[0])))
        flush_list()

        props = document.core_properties
        return LoadedDocument(
            title=options.get("title") or props.title or path.stem.replace("_", " "),
            source_type=self.source_type,
            source_name=path.name,
            file_path=str(path),
            raw_content=raw,
            elements=elements,
            metadata={
                "file_name": path.name,
                "author": props.author or None,
                "date": props.created.isoformat() if props.created else None,
            },
        )
