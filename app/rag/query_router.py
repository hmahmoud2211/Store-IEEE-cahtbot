"""Lightweight query router.

Decides, per question, which stages run:
  * structured SQL over `components` (exact prices, numeric filters, totals)
  * hybrid retrieval over `document_chunks` (meaning + identifiers)
  * both

Principle: never use embeddings for what SQL answers exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.types import QueryAnalysis, QueryType
from app.services.component_service import Resolution


@dataclass
class QueryPlan:
    query_type: QueryType
    use_structured: bool
    use_retrieval: bool
    reasons: list[str] = field(default_factory=list)


class QueryRouter:
    def route(self, analysis: QueryAnalysis, resolution: Resolution) -> QueryPlan:
        resolved = len(resolution.resolved)
        mentioned = resolved + len(resolution.unresolved)

        # A single named component without quantities is just a price lookup.
        single_lookup = resolved == 1 and not analysis.quantities and not analysis.inherited_from_history
        if analysis.is_calculation and not analysis.price.active and not single_lookup:
            return QueryPlan(QueryType.CALCULATION, True, False, ["quantities or cost-estimate wording detected"])

        if analysis.is_comparison and mentioned >= 2:
            return QueryPlan(QueryType.COMPARISON, True, True, [f"comparison intent with {mentioned} components"])

        if analysis.price.active or (analysis.sort and (analysis.categories or resolved >= 2)):
            reason = "numeric price constraint" if analysis.price.active else f"sort={analysis.sort}"
            return QueryPlan(QueryType.PRICE_FILTER, True, False, [reason])

        if analysis.is_build_guide and (analysis.topic or analysis.is_recommendation or not mentioned):
            # Components resolved from earlier turns (the parts list we already gave) feed the BOM.
            return QueryPlan(
                QueryType.PROJECT_GUIDE, bool(analysis.categories or resolved), True, ["step-by-step build request"]
            )

        if analysis.is_recommendation:
            return QueryPlan(
                QueryType.RECOMMENDATION, bool(analysis.categories), True, ["project/recommendation wording"]
            )

        if mentioned and not analysis.is_explanation and not analysis.is_list_request:
            return QueryPlan(QueryType.FACT_LOOKUP, True, True, ["specific component(s) mentioned"])

        if analysis.categories and analysis.is_price_question and not analysis.is_list_request:
            # "What is the price of a breadboard?" -> list the category from SQL.
            return QueryPlan(QueryType.FACT_LOOKUP, True, True, ["category price question"])

        if analysis.is_comparison and mentioned == 1:
            return QueryPlan(QueryType.FACT_LOOKUP, True, True, ["comparison with a single resolvable component"])

        return QueryPlan(
            QueryType.SEMANTIC_SEARCH,
            bool(analysis.categories or resolved),
            True,
            ["open semantic question"],
        )
