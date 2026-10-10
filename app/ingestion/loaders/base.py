"""Loader abstraction.

Every loader turns a source (file path, URL, DB query, API endpoint…) into a
`LoadedDocument`: a list of typed structural `Element`s (headings, paragraphs,
lists, tables, records). The rest of the pipeline only depends on this shape,
never on a concrete loader.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    RECORD = "record"  # one structured row (CSV/Excel/DB/API)


@dataclass
class Element:
    type: ElementType
    text: str
    level: int | None = None  # heading level (1 = top)
    page_number: int | None = None
    section: str | None = None
    rows: list[dict[str, str]] | None = None  # structured data for TABLE / RECORD
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    title: str
    source_type: str
    source_name: str
    elements: list[Element]
    source_url: str | None = None
    file_path: str | None = None
    raw_content: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """SHA-256 of the raw source bytes, falling back to the extracted text."""
        if self.raw_content is not None:
            payload = self.raw_content
        else:
            payload = "\n".join(f"{e.type.value}:{e.text}" for e in self.elements).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements if e.text)


class LoaderError(Exception):
    pass


class BaseDocumentLoader(ABC):
    """Common interface for all document loaders."""

    source_type: ClassVar[str]
    supported_extensions: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def load(self, source: Any, **options: Any) -> LoadedDocument:
        """Load `source` and return a structured document."""

    # Helpers -----------------------------------------------------------------
    @staticmethod
    def _read_file(source: str | Path) -> tuple[Path, bytes]:
        path = Path(source).expanduser()
        if not path.exists() or not path.is_file():
            raise LoaderError(f"File not found: {path}")
        return path, path.read_bytes()


def rows_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Render a table as Markdown so it survives chunking as a readable table."""

    def esc(cell: Any) -> str:
        return str(cell if cell is not None else "").replace("|", "\\|").replace("\n", " ").strip()

    lines = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def table_element(
    header: list[str], rows: list[list[str]], *, page_number: int | None = None, section: str | None = None
) -> Element:
    header = [h.strip() if h else f"column_{i + 1}" for i, h in enumerate(header)]
    dict_rows = [
        {header[i]: (row[i] if i < len(row) and row[i] is not None else "") for i in range(len(header))} for row in rows
    ]
    return Element(
        type=ElementType.TABLE,
        text=rows_to_markdown(header, rows),
        page_number=page_number,
        section=section,
        rows=dict_rows,
        metadata={"columns": header, "row_count": len(rows)},
    )


def record_text(row: dict[str, Any]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))


def records_to_elements(
    rows: list[dict[str, Any]], *, section: str | None = None, start_index: int = 0
) -> list[Element]:
    elements: list[Element] = []
    for i, row in enumerate(rows):
        clean = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
        if not any(clean.values()):
            continue
        elements.append(
            Element(
                type=ElementType.RECORD,
                text=record_text(clean),
                section=section,
                rows=[clean],
                metadata={"row_index": start_index + i},
            )
        )
    return elements
