"""Plain text / Markdown loader with Markdown structure recovery."""

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

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown(text: str) -> list[Element]:
    elements: list[Element] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        i = 0
        while i < len(lines):
            line = lines[i]
            heading = _MD_HEADING_RE.match(line)
            if heading:
                elements.append(Element(ElementType.HEADING, heading.group(2).strip(), level=len(heading.group(1))))
                i += 1
                continue
            if "|" in line and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
                header = _split_row(line)
                rows = []
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    rows.append(_split_row(lines[i]))
                    i += 1
                elements.append(table_element(header, rows))
                continue
            if _LIST_RE.match(line):
                items = []
                while i < len(lines) and (_LIST_RE.match(lines[i]) or lines[i].startswith("  ")):
                    item = lines[i]
                    if _LIST_RE.match(item):
                        items.append("- " + _LIST_RE.sub("", item, count=1).strip())
                    elif items:
                        items[-1] += " " + item.strip()
                    i += 1
                elements.append(Element(ElementType.LIST, "\n".join(items)))
                continue
            para = []
            while (
                i < len(lines)
                and not _MD_HEADING_RE.match(lines[i])
                and not _LIST_RE.match(lines[i])
                and not (i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEP_RE.match(lines[i + 1]))
            ):
                para.append(lines[i].strip())
                i += 1
            if para:
                elements.append(Element(ElementType.PARAGRAPH, " ".join(para)))
    return elements


class TextLoader(BaseDocumentLoader):
    source_type = "text"
    supported_extensions = (".txt", ".md", ".markdown")

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        if options.get("content") is not None:
            raw = str(options["content"]).encode("utf-8")
            name = str(source or options.get("source_name") or "inline.txt")
            file_path = None
        else:
            path, raw = self._read_file(source)
            name, file_path = path.name, str(path)
        try:
            text = raw.decode(options.get("encoding", "utf-8"))
        except UnicodeDecodeError as exc:
            raise LoaderError(f"Could not decode {name}: {exc}") from exc

        elements = parse_markdown(text)
        title = options.get("title")
        if not title:
            first_heading = next((e.text for e in elements if e.type is ElementType.HEADING), None)
            title = first_heading or name.rsplit(".", 1)[0].replace("_", " ")
        return LoadedDocument(
            title=title,
            source_type=self.source_type,
            source_name=name,
            file_path=file_path,
            raw_content=raw,
            elements=elements,
            metadata={"file_name": name},
        )
