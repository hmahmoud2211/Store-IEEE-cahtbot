"""Deterministic, template-based answer generator.

Used when no LLM is configured (no API key / offline evaluation), as a fallback
when the LLM call fails, and when the grounding check rejects an LLM answer.
It can only restate facts present in the ContextBundle, so it cannot hallucinate.
"""

from __future__ import annotations

import time

from app.rag.context_builder import ContextBundle
from app.rag.llm.base import LLMProvider, LLMResponse
from app.rag.prompt_builder import STORE_ORDERS_EMAIL
from app.rag.types import INVENTORY_MATCH, NOT_FOUND_MESSAGE, QueryType

DISCLAIMER = "Prices are approximate and may change between suppliers and over time."


def _snippet(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def render_extractive_answer(bundle: ContextBundle) -> str:
    lines: list[str] = []
    qt = bundle.query_type
    comp_sources = [s for s in bundle.sources if s.component is not None]
    chunk_sources = [s for s in bundle.sources if s.component is None]

    if qt is QueryType.CALCULATION and bundle.calculation:
        calc = bundle.calculation
        if calc.items:
            lines.append("Approximate cost estimate based on the knowledge base prices:")
            for item in calc.items:
                c = item.component
                sid = bundle.source_id_for(c.component_id)
                assumed = f" (assumed for “{item.mention}”)" if item.assumed else ""
                lines.append(f"- {item.quantity.normalize():f} × {c.name}{assumed}: {item.line_total.format()} [{sid}]")
            if calc.total:
                lines.append(f"\nEstimated total = {calc.total.format()}")
        if calc.unresolved:
            lines.append(f"\nNot included (not found in the knowledge base): {', '.join(calc.unresolved)}.")

    elif qt is QueryType.PRICE_FILTER and bundle.price_filter:
        pf = bundle.price_filter
        scope = f"{', '.join(pf.categories).lower()} " if pf.categories else ""
        if pf.sort and pf.matches:
            best = pf.matches[0]
            word = "cheapest" if pf.sort == "asc" else "most expensive"
            sid = bundle.source_id_for(best.component_id)
            lines.append(f"The {word} {scope or 'component '}option is the {best.name} at {best.price_text} [{sid}].")
            if len(pf.matches) > 1:
                lines.append("\nOther options:")
        elif pf.matches:
            constraint = f" priced {pf.constraint}" if pf.constraint else ""
            lines.append(f"Matching {scope}components{constraint}:")
        rest = pf.matches[1:] if pf.sort else pf.matches
        for c in rest:
            lines.append(f"- {c.name}: {c.price_text} [{bundle.source_id_for(c.component_id)}]")
        if pf.partial:
            lines.append("\nPossibly within the limit (the price range straddles it):")
            for c in pf.partial:
                lines.append(f"- {c.name}: {c.price_text} [{bundle.source_id_for(c.component_id)}]")
        if not pf.matches and not pf.partial:
            lines.append(NOT_FOUND_MESSAGE)

    elif qt is QueryType.COMPARISON and bundle.comparison:
        lines.append("Comparison based on the knowledge base:")
        for src in comp_sources:
            c = src.component
            assert c is not None
            detail = f" — {c.description}" if c.description else ""
            lines.append(f"- {c.name} ({c.category}): {c.price_text}{detail} [{src.source_id}]")
        if bundle.comparison.statements:
            lines.append("")
            lines.extend(bundle.comparison.statements)

    elif qt in (QueryType.RECOMMENDATION, QueryType.PROJECT_GUIDE, QueryType.SEMANTIC_SEARCH):
        project = qt is not QueryType.SEMANTIC_SEARCH
        # Catalog parts added only so the LLM can judge availability are not "relevant" by themselves.
        matched = [s for s in comp_sources if not (project and s.component and s.component.matched_by == INVENTORY_MATCH)]
        if project:
            what = "a step-by-step build guide" if qt is QueryType.PROJECT_GUIDE else "a full project plan"
            lines.append(f"I could not produce a verified {what} for this project right now.")
        if matched:
            lines.append(
                "\nStore parts that match this project:" if project else "Relevant components found in the knowledge base:"
            )
            for src in matched:
                c = src.component
                assert c is not None
                uses = f" — typical uses: {'; '.join(c.typical_uses[:3])}" if c.typical_uses else ""
                if len(matched) <= 2 and c.description:
                    uses = f" — {c.description.rstrip('.')}." + uses
                lines.append(f"- {c.name} ({c.category}): {c.price_text}{uses} [{src.source_id}]")
        elif project:
            lines.append("I could not automatically match specific store parts to this project description.")
        for src in chunk_sources:
            lines.append(f"- From {src.label}: {_snippet(src.hit.content if src.hit else src.body)} [{src.source_id}]")
        if project:
            lines.append(
                f"\nPlease try again in a moment. You can also request parts the store does not list by emailing "
                f"{STORE_ORDERS_EMAIL}."
            )

    else:  # FACT_LOOKUP and fallbacks
        for src in comp_sources:
            c = src.component
            assert c is not None
            lines.append(f"The {c.name} costs approximately {c.price_text} [{src.source_id}].")
            if c.sources_disagree:
                prices = "; ".join(
                    f"{s.price.format()} ({s.document_title or s.source_name})" for s in c.sources if s.price
                )
                lines.append(f"Note: sources disagree on this price: {prices}.")
            if c.description:
                lines.append(f"{c.description} [{src.source_id}]")
        for src in chunk_sources:
            lines.append(f"From {src.label}: {_snippet(src.hit.content if src.hit else src.body)} [{src.source_id}]")

    if bundle.not_found:
        lines.append(
            f"\nI could not find {', '.join(bundle.not_found)} in the current knowledge base, so no price is given for "
            f"{'it' if len(bundle.not_found) == 1 else 'them'}."
        )
    if not lines:
        return NOT_FOUND_MESSAGE
    if any(s.component and s.component.price for s in bundle.sources):
        lines.append(f"\n{DISCLAIMER}")
    return "\n".join(lines).strip()


class ExtractiveLLM(LLMProvider):
    model_name = "extractive-template"

    def answer(self, bundle: ContextBundle) -> LLMResponse:
        start = time.perf_counter()
        text = render_extractive_answer(bundle)
        return LLMResponse(text=text, model=self.model_name, latency_ms=round((time.perf_counter() - start) * 1000, 2))

    def generate(self, system_prompt: str, user_prompt: str, *, temperature=None, max_tokens=None) -> LLMResponse:
        raise NotImplementedError("ExtractiveLLM answers from a ContextBundle; call answer(bundle)")
