"""Conservative text cleaning.

Rules:
  * Unicode NFKC (fullwidth chars, NBSP…) but keep en-dash ranges such as "4–10 EGP".
  * Collapse runs of spaces; keep line breaks inside tables and lists.
  * Drop control characters and obvious web boilerplate paragraphs.
  * Never alter technical tokens (74LS85, HC-SR04, 5V, 32.768 kHz).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from app.ingestion.loaders.base import Element, ElementType, LoadedDocument

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍﻿]")
_INLINE_SPACE_RE = re.compile(r"[ \t\f\v   ]+")
_BOILERPLATE_RE = re.compile(
    r"^(?:accept(?: all)? cookies|we use cookies|all rights reserved|©|copyright \d{4}|subscribe to our newsletter|"
    r"sign up for our newsletter|follow us on|share this|skip to (?:main )?content|back to top|"
    r"log ?in|sign ?in|register|add to (?:cart|wishlist)|cart is empty)\b",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, *, keep_newlines: bool = True) -> str:
    text = normalize_unicode(text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    if not keep_newlines:
        return " ".join(line for line in lines if line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_boilerplate(text: str) -> bool:
    return len(text) < 160 and bool(_BOILERPLATE_RE.match(text.strip()))


def clean_element(element: Element) -> Element | None:
    keep_newlines = element.type in (ElementType.TABLE, ElementType.LIST)
    text = clean_text(element.text, keep_newlines=keep_newlines or element.type is ElementType.PARAGRAPH)
    if not text:
        return None
    if element.type in (ElementType.PARAGRAPH, ElementType.LIST) and is_boilerplate(text):
        return None
    rows = None
    if element.rows is not None:
        rows = [
            {clean_text(k, keep_newlines=False): clean_text(v, keep_newlines=False) for k, v in r.items()}
            for r in element.rows
        ]
    return replace(element, text=text, rows=rows)


def clean_document(document: LoadedDocument) -> LoadedDocument:
    cleaned: list[Element] = []
    seen_paragraphs: set[str] = set()
    for element in document.elements:
        result = clean_element(element)
        if result is None:
            continue
        # Drop repeated paragraphs (page headers/footers repeated on every PDF page).
        if result.type is ElementType.PARAGRAPH and len(result.text) < 200:
            key = result.text.lower()
            if key in seen_paragraphs and document.source_type in ("pdf", "web"):
                continue
            seen_paragraphs.add(key)
        cleaned.append(result)
    return replace(document, title=clean_text(document.title, keep_newlines=False), elements=cleaned)
