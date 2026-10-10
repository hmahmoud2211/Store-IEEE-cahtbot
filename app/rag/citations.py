"""Citation validation and numeric grounding checks for generated answers.

* `[n]` markers that point to non-existent sources are removed and reported.
* Every price-like number in the answer (a number adjacent to a currency
  token) must appear in the context or in system-computed results;
  otherwise it is flagged as ungrounded (a likely hallucination).
* The final "Sources:" footer is generated from the cited sources by code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from app.rag.context_builder import ContextBundle, ContextSource

_CITATION_RE = re.compile(r"\[(?:SOURCE\s*)?(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
_CURRENCY = r"(?:EGP|LE|L\.E\.?|pounds?|ج\.م|USD|\$|EUR|€)(?![A-Za-z])"  # "8 LEDs" is not "8 LE"
_NUM = r"\d+(?:[.,]\d+)?"
_PRICE_RANGE_RE = re.compile(rf"({_NUM})\s*(?:-|–|—|to)\s*({_NUM})\s*{_CURRENCY}", re.IGNORECASE)
_PRICE_SINGLE_RE = re.compile(rf"(?<![\d.,–-])({_NUM})\s*{_CURRENCY}", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(_NUM)
# Code and LaTeX math are not prices: "`delay(1000)`", "$1001$" (binary) or "$R_1$".
_NON_PROSE_RE = re.compile(r"```[\s\S]*?(?:```|$)|`[^`\n]+`|\$(?=\S)[^$\n]*?(?<=\S)\$")


@dataclass
class GroundingReport:
    cited_ids: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.ungrounded_numbers


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", "."))
    except InvalidOperation:
        return None


def extract_citation_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            n = int(part.strip())
            if n not in ids:
                ids.append(n)
    return ids


def strip_invalid_citations(text: str, valid_ids: set[int]) -> tuple[str, list[int]]:
    invalid: list[int] = []

    def repl(match: re.Match[str]) -> str:
        nums = [int(p.strip()) for p in match.group(1).split(",")]
        keep = [n for n in nums if n in valid_ids]
        invalid.extend(n for n in nums if n not in valid_ids)
        return "".join(f"[{n}]" for n in keep)

    cleaned = _CITATION_RE.sub(repl, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return cleaned, sorted(set(invalid))


def price_numbers(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    consumed: list[tuple[int, int]] = []
    for m in _PRICE_RANGE_RE.finditer(text):
        consumed.append(m.span())
        for g in (1, 2):
            v = _to_decimal(m.group(g))
            if v is not None:
                values.append(v)
    for m in _PRICE_SINGLE_RE.finditer(text):
        if any(start <= m.start() < end for start, end in consumed):
            continue
        v = _to_decimal(m.group(1))
        if v is not None:
            values.append(v)
    return values


def allowed_numbers(context_text: str) -> set[Decimal]:
    allowed: set[Decimal] = set()
    for raw in _ANY_NUMBER_RE.findall(context_text):
        v = _to_decimal(raw)
        if v is not None:
            allowed.add(v.normalize())
    return allowed


_LLM_SOURCES_SECTION_RE = re.compile(
    r"\n+\s*(?:#+\s*)?\**\s*(?:Sources|References|Citations)\s*:?\s*\**\s*\n[\s\S]*$", re.IGNORECASE
)


def check_answer(answer: str, bundle: ContextBundle) -> tuple[str, GroundingReport]:
    valid_ids = {s.source_id for s in bundle.sources}
    answer = _LLM_SOURCES_SECTION_RE.sub("", answer).strip()  # the system appends its own footer
    cleaned, invalid = strip_invalid_citations(answer, valid_ids)
    report = GroundingReport(
        cited_ids=[i for i in extract_citation_ids(cleaned) if i in valid_ids], invalid_citations=invalid
    )

    allowed = allowed_numbers(bundle.render())
    for value in price_numbers(_NON_PROSE_RE.sub(" ", cleaned)):
        if value.normalize() not in allowed:
            report.ungrounded_numbers.append(format(value.normalize(), "f"))
    return cleaned, report


def select_cited_sources(bundle: ContextBundle, cited_ids: list[int]) -> list[ContextSource]:
    """Sources to return: those cited in the answer; if none were cited, all context sources."""
    if cited_ids:
        by_id = {s.source_id: s for s in bundle.sources}
        return [by_id[i] for i in sorted(cited_ids) if i in by_id]
    return list(bundle.sources)


def format_sources_footer(sources: list[ContextSource]) -> str:
    if not sources:
        return ""
    lines = ["Sources:"]
    for s in sources:
        lines.append(f"[{s.source_id}] {s.label}")
    return "\n".join(lines)
