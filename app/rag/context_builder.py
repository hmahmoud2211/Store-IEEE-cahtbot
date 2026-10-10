"""Context construction: numbered, de-duplicated [SOURCE n] blocks.

Structured component facts (from SQL) and reranked document chunks are merged
into one numbered source list. If a retrieved chunk belongs to a component that
is already present as a structured fact, the chunk's text is attached to that
source instead of creating a duplicate.

Computed results (totals, comparisons, filter summaries) are rendered in a
separate COMPUTED block that references source numbers; they are produced by
code, never by the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ingestion.price_parser import format_amount
from app.rag.types import (
    CalculationResult,
    ComparisonResult,
    ComponentFact,
    PriceFilterResult,
    QueryType,
)
from app.retrieval.types import SearchHit

MAX_CHUNK_CHARS = 2000


@dataclass
class ContextSource:
    source_id: int
    kind: str  # "component" | "chunk"
    label: str
    body: str
    citation: dict[str, Any]
    component: ComponentFact | None = None
    hit: SearchHit | None = None


@dataclass
class ContextBundle:
    query_type: QueryType
    question: str
    sources: list[ContextSource] = field(default_factory=list)
    computed: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    calculation: CalculationResult | None = None
    comparison: ComparisonResult | None = None
    price_filter: PriceFilterResult | None = None

    @property
    def is_empty(self) -> bool:
        return not self.sources

    def source_id_for(self, component_id: str) -> int | None:
        for s in self.sources:
            if s.component and s.component.component_id == component_id:
                return s.source_id
        return None

    def render(self) -> str:
        blocks: list[str] = []
        for s in self.sources:
            blocks.append(f"[SOURCE {s.source_id}]\n{s.body}")
        if self.computed:
            blocks.append("[COMPUTED BY THE SYSTEM — exact, use these numbers verbatim]\n" + "\n".join(self.computed))
        if self.not_found:
            blocks.append(
                "[NOT FOUND IN KNOWLEDGE BASE]\n"
                + "\n".join(f"- {item}: no matching record in the knowledge base" for item in self.not_found)
            )
        if self.notes:
            blocks.append("[NOTES]\n" + "\n".join(f"- {n}" for n in self.notes))
        return "\n\n".join(blocks) if blocks else "(no relevant context was retrieved)"


def _document_label(
    title: str | None, source_name: str | None, page: int | None = None, section: str | None = None
) -> str:
    doc = title or source_name or "Unknown document"
    if source_name and title and source_name != title:
        doc = f"{title} ({source_name})"
    if page:
        doc += f" — page {page}"
    if section:
        doc += f" — section: {section}"
    return doc


class ContextBuilder:
    def __init__(self, max_chunk_chars: int = MAX_CHUNK_CHARS) -> None:
        self.max_chunk_chars = max_chunk_chars

    def build(
        self,
        *,
        query_type: QueryType,
        question: str,
        components: list[ComponentFact],
        hits: list[SearchHit],
        not_found: list[str] | None = None,
        calculation: CalculationResult | None = None,
        comparison: ComparisonResult | None = None,
        price_filter: PriceFilterResult | None = None,
        notes: list[str] | None = None,
    ) -> ContextBundle:
        bundle = ContextBundle(
            query_type=query_type,
            question=question,
            not_found=list(not_found or []),
            notes=list(notes or []),
            calculation=calculation,
            comparison=comparison,
            price_filter=price_filter,
        )

        by_component: dict[str, ContextSource] = {}
        for fact in components:
            if fact.component_id in by_component:
                continue
            src = self._component_source(len(bundle.sources) + 1, fact)
            bundle.sources.append(src)
            by_component[fact.component_id] = src

        seen_chunks: set[str] = set()
        for hit in hits:
            key = str(hit.chunk_id)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            comp_id = hit.component_id
            if comp_id and comp_id in by_component:
                src = by_component[comp_id]
                src.citation.setdefault("chunk_id", key)
                src.hit = src.hit or hit
                continue
            bundle.sources.append(self._chunk_source(len(bundle.sources) + 1, hit))

        bundle.computed = self._computed_lines(bundle)
        return bundle

    # ------------------------------------------------------------- sources
    def _component_source(self, source_id: int, fact: ComponentFact) -> ContextSource:
        primary = fact.sources[0] if fact.sources else None
        lines = [
            "Type: structured component record (database)",
            f"Component: {fact.name}",
            f"Category: {fact.category}",
            f"Approximate price: {fact.price_text}",
        ]
        if fact.price:
            lines.append(
                f"Price fields: min={format_amount(fact.price.min)}, max={format_amount(fact.price.max)}, "
                f'currency={fact.price.currency}, unit={fact.price.unit}, original text="{fact.price.original}"'
            )
        if fact.aliases:
            lines.append(f"Also known as: {', '.join(fact.aliases[:10])}")
        if fact.description:
            lines.append(f"Description (curated annotation): {fact.description}")
        if fact.typical_uses:
            lines.append(f"Typical uses (curated annotation): {'; '.join(fact.typical_uses)}")
        if fact.partial_match:
            lines.append("Filter note: the price RANGE only partially satisfies the requested price limit.")
        if fact.matched_by:
            lines.append(f"Matched by: {fact.matched_by}")
        if len(fact.sources) > 1:
            lines.append("Prices observed per source:")
            for s in fact.sources:
                lines.append(
                    f"  - {s.price.format() if s.price else 'n/a'} from {_document_label(s.document_title, s.source_name)}"
                    + (f" (supplier: {s.supplier})" if s.supplier else "")
                )
            if fact.sources_disagree:
                lines.append("WARNING: sources disagree on the price of this component.")
        document_label = _document_label(primary.document_title, primary.source_name) if primary else "Unknown document"
        lines.append(f"Source document: {document_label}")

        citation = {
            "kind": "component",
            "document": primary.source_name if primary else None,
            "document_title": primary.document_title if primary else None,
            "document_id": primary.document_id if primary else None,
            "source_type": primary.source_type if primary else None,
            "source_url": primary.source_url if primary else None,
            "component": fact.name,
            "component_id": fact.component_id,
            "page": None,
            "section": fact.category,
            "price_text": fact.price.original if fact.price else None,
        }
        label = f"{primary.document_title if primary and primary.document_title else document_label} — {fact.name}"
        return ContextSource(source_id, "component", label, "\n".join(lines), citation, component=fact)

    def _chunk_source(self, source_id: int, hit: SearchHit) -> ContextSource:
        content = hit.content
        if len(content) > self.max_chunk_chars:
            content = content[: self.max_chunk_chars].rsplit(" ", 1)[0] + " …"
        doc_label = _document_label(hit.document_title, hit.source_name, hit.page_number, hit.section)
        body = f"Type: document excerpt\nDocument: {doc_label}\nContent:\n{content}"
        citation = {
            "kind": "chunk",
            "document": hit.source_name,
            "document_title": hit.document_title,
            "document_id": str(hit.document_id),
            "source_type": hit.source_type,
            "source_url": hit.source_url,
            "component": hit.metadata.get("component"),
            "component_id": hit.metadata.get("component_id"),
            "page": hit.page_number,
            "section": hit.section,
            "price_text": hit.metadata.get("price_text"),
            "chunk_id": str(hit.chunk_id),
            "score": round(hit.best_score, 4),
        }
        if hit.page_number:
            label = f"{hit.source_name or hit.document_title} — page {hit.page_number}"
        else:
            suffix = hit.metadata.get("component") or hit.section
            label = f"{hit.document_title or hit.source_name}" + (f" — {suffix}" if suffix else "")
        return ContextSource(source_id, "chunk", label, body, citation, hit=hit)

    # ------------------------------------------------------------ computed
    def _computed_lines(self, bundle: ContextBundle) -> list[str]:
        lines: list[str] = []

        def ref(component_id: str) -> str:
            sid = bundle.source_id_for(component_id)
            return f" [{sid}]" if sid else ""

        if bundle.calculation:
            calc = bundle.calculation
            lines.append("Cost estimate (quantity × unit price range):")
            for item in calc.items:
                c = item.component
                note = ""
                if item.assumed:
                    note = f" (assumed for '{item.mention}'"
                    note += f"; alternatives: {', '.join(item.alternatives)})" if item.alternatives else ")"
                lines.append(
                    f"- {format_amount(item.quantity)} × {c.name} @ {c.price.format() if c.price else 'n/a'} "
                    f"= {item.line_total.format()}{ref(c.component_id)}{note}"
                )
            if calc.total:
                lines.append(f"Estimated total = {calc.total.format()}")
            if calc.mixed_currency:
                lines.append("Total not computed: items are priced in different currencies.")
            if calc.unresolved:
                lines.append(f"Not priced (not found): {', '.join(calc.unresolved)}")

        if bundle.comparison:
            lines.append("Comparison facts:")
            lines.extend(f"- {s}" for s in bundle.comparison.statements)

        if bundle.price_filter:
            pf = bundle.price_filter
            scope = f" in category {', '.join(pf.categories)}" if pf.categories else ""
            constraint = f" priced {pf.constraint}" if pf.constraint else ""
            order = {"asc": " (sorted cheapest first)", "desc": " (sorted most expensive first)"}.get(pf.sort or "", "")
            lines.append(f"SQL filter: components{scope}{constraint}{order}: {len(pf.matches)} full match(es).")
            for c in pf.matches:
                lines.append(f"- {c.name}: {c.price_text}{ref(c.component_id)}")
            if pf.partial:
                lines.append("Partial matches (price range straddles the limit):")
                for c in pf.partial:
                    lines.append(f"- {c.name}: {c.price_text}{ref(c.component_id)}")
            if pf.sort and pf.matches:
                best = pf.matches[0]
                word = "cheapest" if pf.sort == "asc" else "most expensive"
                lines.append(f"The {word} match is {best.name} at {best.price_text}{ref(best.component_id)}.")
                overlapping = [
                    c.name
                    for c in pf.matches[1:]
                    if c.price and best.price and c.price.min <= best.price.max and c.price.max >= best.price.min
                ]
                if overlapping:
                    lines.append(
                        f"Note: ranking uses the {'lower' if pf.sort == 'asc' else 'upper'} bound of each price "
                        "range; these ranges overlap with it: " + ", ".join(overlapping)
                    )
        return lines
