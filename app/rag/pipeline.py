"""Online RAG pipeline.

question
  → QueryProcessor (identifiers, categories, price constraints, quantities, intent)
  → component resolution (aliases / identifiers)          [SQL-backed index]
  → QueryRouter → QueryPlan
  → structured stage (SQL filters, lookups, pricing math)  [when applicable]
  → hybrid retrieval (pgvector + FTS → RRF → metadata/ACL filter)  [when applicable]
  → reranker → thresholds → best 3–8 chunks
  → ContextBuilder → PromptBuilder → LLM
  → citation validation + numeric grounding → answer + structured sources
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.core.cache import Cache
from app.core.config import Settings
from app.core.logging import get_logger, text_fingerprint, timed
from app.core.security import Principal
from app.db.repositories.components import ComponentQuery, PriceBound
from app.db.session import session_scope
from app.embeddings.base import EmbeddingProvider
from app.rag.citations import check_answer, format_sources_footer, select_cited_sources
from app.rag.context_builder import ContextBuilder, ContextBundle
from app.rag.llm.base import LLMError, LLMProvider
from app.rag.llm.extractive import ExtractiveLLM
from app.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.rag.query_processor import ChatMessage, QueryProcessor
from app.rag.query_router import QueryPlan, QueryRouter
from app.rag.types import (
    INVENTORY_MATCH,
    NO_ANSWER_MESSAGE,
    CalculationResult,
    ComparisonResult,
    ComponentFact,
    PriceFilterResult,
    QueryAnalysis,
    QueryType,
    ResolvedComponent,
)
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.metadata_filter import RetrievalFilters
from app.retrieval.reranker import Reranker
from app.retrieval.types import SearchHit
from app.services.component_service import ComponentService, Resolution
from app.services.pricing_service import PricingService

log = get_logger(__name__)

# Project questions: the answer lists everything the project needs, found or not.
PROJECT_TYPES = (QueryType.RECOMMENDATION, QueryType.PROJECT_GUIDE)
# Project answers (BOM tables, build guides with code) plus model thinking do not fit
# the default output budget.
PROJECT_MIN_OUTPUT_TOKENS = 8192

CALCULATION_NEEDS_ITEMS = (
    "Please tell me which components (and quantities) to include, for example: "
    '"2 × NE555, 1 × 400-point breadboard, 5 × LED". I will then compute the approximate total from the '
    "knowledge base prices."
)


@dataclass
class UserFilters:
    categories: list[str] | None = None
    currency: str | None = None
    max_price: Decimal | None = None
    min_price: Decimal | None = None
    document_ids: list[uuid.UUID] | None = None
    source_types: list[str] | None = None
    supplier: str | None = None
    country: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    answer: str
    query_type: QueryType
    sources: list[dict[str, Any]]
    analysis: QueryAnalysis
    structured_results: list[ComponentFact] = field(default_factory=list)
    calculation: CalculationResult | None = None
    comparison: ComparisonResult | None = None
    price_filter: PriceFilterResult | None = None
    hits: list[SearchHit] = field(default_factory=list)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    context: ContextBundle | None = None
    no_answer: bool = False


@dataclass
class _Prepared:
    query_type: QueryType
    analysis: QueryAnalysis
    bundle: ContextBundle
    facts: list[ComponentFact]
    hits: list[SearchHit]
    calculation: CalculationResult | None
    comparison: ComparisonResult | None
    price_filter: PriceFilterResult | None
    meta: dict[str, Any]


class RAGPipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: EmbeddingProvider,
        reranker: Reranker,
        llm: LLMProvider | None,
        settings: Settings,
        cache: Cache | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.cache = cache
        self.processor = QueryProcessor(default_currency=settings.default_currency)
        self.router = QueryRouter()
        self.retriever = HybridRetriever(embedder, settings, cache)
        self.reranker = reranker
        self.llm = llm
        self.extractive = ExtractiveLLM()
        self.context_builder = ContextBuilder()
        self.pricing = PricingService()

    # ================================================================ public
    def answer(
        self,
        question: str,
        principal: Principal | None = None,
        *,
        filters: UserFilters | None = None,
        history: list[ChatMessage] | None = None,
        final_k: int | None = None,
        use_llm: bool = True,
    ) -> RAGAnswer:
        principal = principal or Principal.anonymous()
        timings: dict[str, float] = {}
        start = time.perf_counter()

        with timed(timings, "query_processing_ms"):
            analysis = self.processor.analyze(question, history)

        with session_scope(self.session_factory) as session:
            components = ComponentService(session, self.cache)

            with timed(timings, "component_resolution_ms"):
                resolution = components.resolve(analysis.normalized, analysis.identifiers)
                if analysis.inherited_from_history and not resolution.resolved:
                    past = components.resolve(self.processor.history_text(history))
                    resolution = Resolution(past.resolved, [])
                    if not past.resolved:
                        analysis.inherited_from_history = False

            plan = self.router.route(analysis, resolution)
            log.info(
                "query_routed",
                query_fp=text_fingerprint(question),
                query_len=len(question),
                query_type=plan.query_type.value,
                reasons=plan.reasons,
                resolved=[r.name for r in resolution.resolved],
                unresolved=resolution.unresolved,
            )

            prepared = self._prepare(
                session, question, analysis, resolution, plan, principal, filters, final_k, timings
            )

        # Generation happens after the DB session is released.
        result = prepared if isinstance(prepared, RAGAnswer) else self._finish(prepared, use_llm, timings)
        timings["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
        result.retrieval_metadata["timings_ms"] = timings
        log.info(
            "query_answered",
            query_fp=text_fingerprint(question),
            query_type=result.query_type.value,
            no_answer=result.no_answer,
            retrieved_document_ids=sorted({str(h.document_id) for h in result.hits}),
            retrieved_chunk_ids=[str(h.chunk_id) for h in result.hits],
            structured_ids=[c.component_id for c in result.structured_results],
            **{k: v for k, v in timings.items()},
        )
        return result

    def search(
        self,
        question: str,
        principal: Principal | None = None,
        filters: UserFilters | None = None,
        rerank: bool = True,
        top_k: int | None = None,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        """Retrieval only (no generation) — used by POST /search and evaluation."""
        principal = principal or Principal.anonymous()
        timings: dict[str, float] = {}
        analysis = self.processor.analyze(question)
        with session_scope(self.session_factory) as session:
            rf = self._retrieval_filters(filters, None)
            hybrid = self.retriever.retrieve(session, question, principal, rf, extra_terms=analysis.identifiers)
            timings.update(hybrid.timings)
            hits = hybrid.hits[: self.settings.rerank_candidates]
            if rerank:
                with timed(timings, "rerank_ms"):
                    hits = self.reranker.rerank(question, hits)
        hits = hits[: top_k or self.settings.final_context_k]
        meta = {
            "vector_candidates": hybrid.vector_candidates,
            "keyword_candidates": hybrid.keyword_candidates,
            "fused_candidates": hybrid.fused_candidates,
            "timings_ms": timings,
            "analysis": analysis.as_dict(),
        }
        return hits, meta

    # ============================================================= internals
    def _prepare(
        self,
        session: Session,
        question: str,
        analysis: QueryAnalysis,
        resolution: Resolution,
        plan: QueryPlan,
        principal: Principal,
        filters: UserFilters | None,
        final_k: int | None,
        timings: dict[str, float],
    ) -> RAGAnswer | _Prepared:
        components = ComponentService(session, self.cache)
        qt = plan.query_type
        facts: list[ComponentFact] = []
        calculation: CalculationResult | None = None
        comparison: ComparisonResult | None = None
        price_filter: PriceFilterResult | None = None
        not_found = list(resolution.unresolved)
        notes: list[str] = []
        meta: dict[str, Any] = {"query_type": qt.value, "route_reasons": plan.reasons, "analysis": analysis.as_dict()}

        # ------------------------------------------------ structured stage
        with timed(timings, "structured_ms"):
            if qt is QueryType.CALCULATION:
                calculation, facts = self._calculate(components, analysis, resolution)
                not_found = list(calculation.unresolved) if calculation else not_found
                if calculation is None:
                    return self._fixed_answer(qt, analysis, CALCULATION_NEEDS_ITEMS, meta, no_answer=True)
            elif qt is QueryType.COMPARISON:
                facts = components.facts(resolution.resolved)
                comparison = self.pricing.compare(facts) if len(facts) >= 2 else None
            elif qt is QueryType.PRICE_FILTER:
                price_filter, facts = self._price_filter(components, analysis, resolution, filters)
            elif plan.use_structured:
                facts = components.facts(resolution.resolved)
                if not facts and analysis.categories:
                    cat_query = ComponentQuery(
                        categories=analysis.categories,
                        sort="asc",
                        limit=analysis.limit or 25,
                        currency=filters.currency if filters else None,
                    )
                    facts, _ = components.search(cat_query)
                    if facts:
                        notes.append(
                            f"Structured listing of category {', '.join(analysis.categories)} from the database."
                        )

        # A named part that is not in the KB and nothing else matched: refuse early.
        if not_found and not facts and qt in (QueryType.FACT_LOOKUP, QueryType.COMPARISON, QueryType.CALCULATION):
            msg = (
                f"I could not find {', '.join(not_found)} in the current knowledge base, so I cannot give a reliable "
                f"answer (no price is available for {'it' if len(not_found) == 1 else 'them'})."
            )
            return self._fixed_answer(qt, analysis, msg, meta, no_answer=True)

        # ------------------------------------------------ retrieval stage
        hits: list[SearchHit] = []
        if plan.use_retrieval:
            hits = self._retrieve(session, question, analysis, facts, qt, principal, filters, final_k, timings, meta)

        # Retrieved component chunks become structured records (exact price fields
        # + provenance) in rerank order; their chunk text is merged into that source.
        if hits and qt in (*PROJECT_TYPES, QueryType.SEMANTIC_SEARCH):
            known = {f.component_id for f in facts}
            promote = [h.component_id for h in hits if h.component_id and h.component_id not in known]
            if promote:
                facts = facts + components.facts(
                    [ResolvedComponent(cid, "", "retrieval", "retrieval") for cid in dict.fromkeys(promote)]
                )

        # Project answers mark every needed part "In store" / "Not in store"; that is only
        # reliable if the model sees the whole catalog, not just what retrieval surfaced.
        if qt in PROJECT_TYPES and self.settings.project_inventory_limit:
            known = {f.component_id for f in facts}
            inventory, _ = components.search(
                ComponentQuery(
                    limit=self.settings.project_inventory_limit, currency=filters.currency if filters else None
                )
            )
            for fact in inventory:
                if fact.component_id not in known:
                    fact.matched_by = INVENTORY_MATCH
                    facts.append(fact)

        if not facts and not hits and not (price_filter and (price_filter.matches or price_filter.partial)):
            if price_filter is not None:
                cat = f" in category {', '.join(price_filter.categories)}" if price_filter.categories else ""
                msg = f"No components{cat} priced {price_filter.constraint} were found in the current knowledge base."
                return self._fixed_answer(qt, analysis, msg, meta, no_answer=True)
            
            # SPECIAL CASE: For recommendations, if we found absolutely nothing, we still want the LLM to try 
            # and suggest a theoretical bill of materials (BOM) using its general knowledge, 
            # which will then be grounded by our prompt rules to list missing parts.
            if qt in PROJECT_TYPES:
                # Pass through to generation instead of returning "no answer"
                pass 
            else:
                return self._fixed_answer(qt, analysis, NO_ANSWER_MESSAGE, meta, no_answer=True)

        # ------------------------------------------------ context + generation
        bundle = self.context_builder.build(
            query_type=qt,
            question=question,
            components=facts,
            hits=hits,
            not_found=not_found,
            calculation=calculation,
            comparison=comparison,
            price_filter=price_filter,
            notes=notes,
        )
        return _Prepared(qt, analysis, bundle, facts, hits, calculation, comparison, price_filter, meta)

    def _finish(self, p: _Prepared, use_llm: bool, timings: dict[str, float]) -> RAGAnswer:
        bundle, meta, hits, facts = p.bundle, p.meta, p.hits, p.facts
        answer_text, gen_meta = self._generate(bundle, p.analysis, use_llm, timings)
        meta.update(gen_meta)

        _, report = check_answer(answer_text, bundle)
        cited = select_cited_sources(bundle, report.cited_ids)
        if not report.cited_ids and p.query_type in PROJECT_TYPES:
            # Nothing cited: list only parts that matched the project, not the whole catalog.
            cited = [s for s in cited if not (s.component and s.component.matched_by == INVENTORY_MATCH)]
        footer = format_sources_footer(cited)
        full_answer = f"{answer_text}\n\n{footer}" if footer else answer_text

        meta.update(
            {
                "final_chunks": len(hits),
                "structured_results": len(facts),
                "context_sources": len(bundle.sources),
                "cited_sources": [s.source_id for s in cited],
            }
        )
        return RAGAnswer(
            answer=full_answer,
            query_type=p.query_type,
            sources=[{"source_id": s.source_id, "label": s.label, **s.citation} for s in cited],
            analysis=p.analysis,
            structured_results=facts,
            calculation=p.calculation,
            comparison=p.comparison,
            price_filter=p.price_filter,
            hits=hits,
            retrieval_metadata=meta,
            context=bundle,
        )

    # --------------------------------------------------------------- stages
    def _calculate(
        self, components: ComponentService, analysis: QueryAnalysis, resolution: Resolution
    ) -> tuple[CalculationResult | None, list[ComponentFact]]:
        if analysis.quantities:
            wanted = [(q.quantity, q.mention) for q in analysis.quantities]
        else:
            mentions = list(dict.fromkeys(r.matched_text for r in resolution.resolved))
            wanted = [(Decimal(1), m) for m in mentions]
        if not wanted:
            return None, []

        resolved_items = []
        unresolved: list[str] = list(resolution.unresolved) if not analysis.quantities else []
        for quantity, mention in wanted:
            best, alternatives = components.resolve_item(mention)
            if best is None:
                unresolved.append(mention)
                continue
            resolved_items.append(
                (best, quantity, mention, best.match_type == "fuzzy" or bool(alternatives), alternatives)
            )

        facts_by_id = {f.component_id: f for f in components.facts([r[0] for r in resolved_items])}
        items = [
            (facts_by_id[best.component_id], qty, mention, assumed, alts)
            for best, qty, mention, assumed, alts in resolved_items
            if best.component_id in facts_by_id
        ]
        calc = self.pricing.estimate(items, unresolved=unresolved)
        unique_facts = list({f.component_id: f for f, *_ in items}.values())
        return calc, unique_facts

    def _price_filter(
        self,
        components: ComponentService,
        analysis: QueryAnalysis,
        resolution: Resolution,
        filters: UserFilters | None,
    ) -> tuple[PriceFilterResult, list[ComponentFact]]:
        p = analysis.price
        max_bound = PriceBound(p.max_price, p.max_inclusive) if p.max_price is not None else None
        min_bound = PriceBound(p.min_price, p.min_inclusive) if p.min_price is not None else None
        if filters and filters.max_price is not None and max_bound is None:
            max_bound = PriceBound(filters.max_price, True)
        if filters and filters.min_price is not None and min_bound is None:
            min_bound = PriceBound(filters.min_price, True)
        categories = analysis.categories or (filters.categories if filters and filters.categories else None)
        restrict_ids = None
        if not categories and len(resolution.resolved) >= 2:
            restrict_ids = [uuid.UUID(r.component_id) for r in resolution.resolved]
        query = ComponentQuery(
            categories=categories,
            component_ids=restrict_ids,
            max_price=max_bound,
            min_price=min_bound,
            approx_price=p.approx_price,
            approx_tolerance=self.settings.price_approx_tolerance,
            currency=p.currency or (filters.currency if filters else None),
            sort=analysis.sort,
            limit=analysis.limit or 50,
        )
        matches, partial = components.search(query)
        result = PriceFilterResult(
            matches=matches,
            partial=partial,
            constraint=p.describe(),
            categories=list(categories or []),
            sort=analysis.sort,
        )
        return result, [*matches, *partial]

    def _retrieval_filters(self, filters: UserFilters | None, categories: list[str] | None) -> RetrievalFilters | None:
        rf = RetrievalFilters()
        if filters:
            rf.categories = filters.categories
            rf.currency = filters.currency
            rf.max_price = filters.max_price
            rf.min_price = filters.min_price
            rf.document_ids = filters.document_ids
            rf.source_types = filters.source_types
            rf.supplier = filters.supplier
            rf.country = filters.country
            rf.metadata = dict(filters.metadata or {})
        if categories:
            rf.categories = categories
        return None if rf.is_empty() else rf

    def _retrieve(
        self,
        session: Session,
        question: str,
        analysis: QueryAnalysis,
        facts: list[ComponentFact],
        qt: QueryType,
        principal: Principal,
        filters: UserFilters | None,
        final_k: int | None,
        timings: dict[str, float],
        meta: dict[str, Any],
    ) -> list[SearchHit]:
        s = self.settings
        rf = self._retrieval_filters(filters, None)
        extra_terms = list(analysis.identifiers)
        # "Give me the steps" carries no subject; search for the project named earlier instead.
        search_query = analysis.topic if analysis.topic_from_history and analysis.topic else question
        hybrid = self.retriever.retrieve(
            session, search_query, principal, rf, extra_terms=extra_terms, top_k=s.hybrid_top_k
        )
        timings.update(hybrid.timings)

        candidates = hybrid.hits[: s.rerank_candidates]
        # Cross-encoders judge "does this passage answer the question?". For
        # project/related-to questions no single component does, so rerank
        # against the topic phrase ("digital clock") instead of the question.
        rerank_query = (
            analysis.topic
            if analysis.topic and qt in (*PROJECT_TYPES, QueryType.SEMANTIC_SEARCH)
            else question
        )
        with timed(timings, "rerank_ms"):
            reranked = self.reranker.rerank(rerank_query, candidates)

        k = final_k or (s.recommendation_context_k if qt in PROJECT_TYPES else s.final_context_k)
        kept = [h for h in reranked if h.scores.get("rerank", 0.0) >= s.min_rerank_score]

        # For lookups/comparisons the structured facts already cover the named
        # components; only keep chunks about those components or non-component documents.
        if facts and qt in (QueryType.FACT_LOOKUP, QueryType.COMPARISON):
            fact_ids = {f.component_id for f in facts}
            kept = [h for h in kept if h.component_id is None or h.component_id in fact_ids]

        final = kept[:k]
        meta.update(
            {
                "vector_candidates": hybrid.vector_candidates,
                "keyword_candidates": hybrid.keyword_candidates,
                "retrieved_candidates": len(candidates),
                "reranked_above_threshold": len(kept),
                "retrieval_cache_hit": hybrid.cached,
                "reranker": self.reranker.name,
                "rerank_query": rerank_query if rerank_query != question else None,
                "top_rerank_score": round(reranked[0].scores.get("rerank", 0.0), 4) if reranked else None,
            }
        )
        return final

    def _generate(
        self, bundle: ContextBundle, analysis: QueryAnalysis, use_llm: bool, timings: dict[str, float]
    ) -> tuple[str, dict[str, Any]]:
        meta: dict[str, Any] = {"llm_model": None, "llm_fallback": False, "grounding_warning": None}
        if not use_llm or self.llm is None or isinstance(self.llm, ExtractiveLLM):
            with timed(timings, "llm_ms"):
                response = self.extractive.answer(bundle)
            meta["llm_model"] = response.model
            return response.text, meta

        prompt = build_user_prompt(bundle, project=analysis.topic)
        max_tokens = None
        if bundle.query_type in PROJECT_TYPES:
            max_tokens = max(self.settings.llm_max_output_tokens, PROJECT_MIN_OUTPUT_TOKENS)
        try:
            with timed(timings, "llm_ms"):
                response = self.llm.generate(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
            meta["llm_model"] = response.model
            meta["llm_usage"] = response.usage
        except LLMError as exc:
            log.error("llm_failed_using_extractive_fallback", error=str(exc))
            meta.update({"llm_fallback": True, "llm_error": exc.__class__.__name__})
            return self.extractive.answer(bundle).text, meta

        cleaned, report = check_answer(response.text, bundle)
        if report.invalid_citations:
            meta["invalid_citations_removed"] = report.invalid_citations
        if not report.grounded:
            meta["grounding_warning"] = f"ungrounded price numbers: {', '.join(report.ungrounded_numbers)}"
            log.warning("ungrounded_answer", numbers=report.ungrounded_numbers, policy=self.settings.grounding_policy)
            if self.settings.grounding_policy == "fallback":
                meta["llm_fallback"] = True
                return self.extractive.answer(bundle).text, meta
        return cleaned, meta

    @staticmethod
    def _fixed_answer(
        qt: QueryType, analysis: QueryAnalysis, message: str, meta: dict[str, Any], *, no_answer: bool
    ) -> RAGAnswer:
        meta.update({"final_chunks": 0, "no_answer": no_answer})
        return RAGAnswer(
            answer=message,
            query_type=qt,
            sources=[],
            analysis=analysis,
            retrieval_metadata=meta,
            no_answer=no_answer,
        )
