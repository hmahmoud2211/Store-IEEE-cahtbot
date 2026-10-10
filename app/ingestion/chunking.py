"""Structure-aware chunking.

* RECORD elements (CSV/Excel/DB/API rows, component records) -> exactly one
  chunk each, no overlap: one logical item per chunk.
* Prose is grouped by heading path: paragraphs, lists and tables of the same
  section are packed together up to CHUNK_SIZE tokens.
* A new heading always starts a new chunk; the heading path is prefixed to
  each chunk and kept in `section`.
* Tables are never split mid-row; oversized tables are split by rows with
  the header repeated.
* Overlap (CHUNK_OVERLAP tokens, sentence-aligned) is only applied between
  consecutive prose chunks of the same section.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.ingestion.loaders.base import Element, ElementType, LoadedDocument, rows_to_markdown

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


@lru_cache
def _tiktoken_encoder():  # pragma: no cover - depends on environment
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    encoder = _tiktoken_encoder()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    # Fallback heuristic (~1.3 tokens per word for technical English).
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class ChunkDraft:
    content: str
    section: str | None = None
    page_number: int | None = None
    token_count: int = 0
    element_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        token_counter: Callable[[str], int] = count_tokens,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.count = token_counter

    # ------------------------------------------------------------------ API
    def chunk(self, document: LoadedDocument) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        heading_stack: list[tuple[int, str]] = []
        buffer: list[Element] = []
        buffer_tokens = 0
        prev_prose_tail: str | None = None  # overlap carried into the next prose chunk

        def section_path() -> str | None:
            return " > ".join(text for _, text in heading_stack) or None

        def flush(carry_overlap: bool) -> None:
            nonlocal buffer, buffer_tokens, prev_prose_tail
            if not buffer:
                return
            body = "\n\n".join(e.text for e in buffer)
            if prev_prose_tail:
                body = f"{prev_prose_tail}\n\n{body}"
            chunks.append(self._make_chunk(body, section_path(), buffer))
            prose = [e for e in buffer if e.type in (ElementType.PARAGRAPH, ElementType.LIST)]
            prev_prose_tail = (
                self._tail(prose[-1].text) if carry_overlap and prose and buffer[-1] is prose[-1] else None
            )
            buffer, buffer_tokens = [], 0

        for element in document.elements:
            if element.type is ElementType.HEADING:
                flush(carry_overlap=False)
                prev_prose_tail = None
                level = element.level or 1
                heading_stack = [(lvl, txt) for lvl, txt in heading_stack if lvl < level]
                heading_stack.append((level, element.text))
                continue

            if element.type is ElementType.RECORD:
                flush(carry_overlap=False)
                prev_prose_tail = None
                section = element.section or section_path()
                chunks.append(self._make_chunk(element.text, section, [element], prefix_section=False))
                continue

            tokens = self.count(element.text)

            if element.type is ElementType.TABLE and tokens > self.chunk_size:
                flush(carry_overlap=False)
                prev_prose_tail = None
                for part in self._split_table(element):
                    chunks.append(self._make_chunk(part.text, section_path(), [part]))
                continue

            if element.type in (ElementType.PARAGRAPH, ElementType.LIST) and tokens > self.chunk_size:
                flush(carry_overlap=True)
                for piece in self._split_prose(element.text):
                    piece_el = Element(element.type, piece, page_number=element.page_number, metadata=element.metadata)
                    chunks.append(self._make_chunk(piece, section_path(), [piece_el]))
                prev_prose_tail = None
                continue

            if buffer and buffer_tokens + tokens > self.chunk_size:
                flush(carry_overlap=element.type is not ElementType.TABLE)
            buffer.append(element)
            buffer_tokens += tokens

        flush(carry_overlap=False)
        return chunks

    # -------------------------------------------------------------- helpers
    def _make_chunk(
        self, body: str, section: str | None, elements: list[Element], prefix_section: bool = True
    ) -> ChunkDraft:
        content = f"{section}\n\n{body}" if section and prefix_section else body
        meta: dict[str, Any] = {}
        for el in elements:
            meta.update(el.metadata)
        return ChunkDraft(
            content=content,
            section=section,
            page_number=next((e.page_number for e in elements if e.page_number is not None), None),
            token_count=self.count(content),
            element_types=sorted({e.type.value for e in elements}),
            metadata=meta,
        )

    def _tail(self, text: str) -> str | None:
        """Last sentences of `text` totalling <= chunk_overlap tokens."""
        if self.chunk_overlap <= 0:
            return None
        sentences = _SENTENCE_RE.split(text)
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            n = self.count(sentence)
            if total + n > self.chunk_overlap:
                break
            tail.insert(0, sentence)
            total += n
        return " ".join(tail) or None

    def _split_prose(self, text: str) -> list[str]:
        """Sentence-aligned windows of ~chunk_size tokens with ~chunk_overlap overlap."""
        sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            n = self.count(sentence)
            if n > self.chunk_size:  # pathological sentence: hard-split by words
                words = sentence.split()
                step = max(1, int(len(words) * self.chunk_size / n))
                for i in range(0, len(words), step):
                    pieces.append(" ".join(words[i : i + step]))
                continue
            if current and current_tokens + n > self.chunk_size:
                pieces.append(" ".join(current))
                overlap: list[str] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    t = self.count(prev)
                    if overlap_tokens + t > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    overlap_tokens += t
                current, current_tokens = overlap, overlap_tokens
            current.append(sentence)
            current_tokens += n
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _split_table(self, element: Element) -> list[Element]:
        rows = element.rows or []
        if not rows:
            return [element]
        header = list(rows[0].keys())
        header_tokens = self.count(rows_to_markdown(header, []))
        parts: list[Element] = []
        group: list[dict[str, str]] = []
        group_tokens = header_tokens
        for row in rows:
            row_tokens = self.count(" | ".join(row.values())) + 2
            if group and group_tokens + row_tokens > self.chunk_size:
                parts.append(self._table_part(element, header, group, len(parts)))
                group, group_tokens = [], header_tokens
            group.append(row)
            group_tokens += row_tokens
        if group:
            parts.append(self._table_part(element, header, group, len(parts)))
        return parts

    @staticmethod
    def _table_part(element: Element, header: list[str], rows: list[dict[str, str]], index: int) -> Element:
        return Element(
            type=ElementType.TABLE,
            text=rows_to_markdown(header, [[r.get(h, "") for h in header] for r in rows]),
            page_number=element.page_number,
            rows=rows,
            metadata={**element.metadata, "table_part": index, "row_count": len(rows)},
        )
