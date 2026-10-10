"""Data structures shared by the query-side RAG stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from app.ingestion.price_parser import PriceRange


class QueryType(str, Enum):
    FACT_LOOKUP = "FACT_LOOKUP"
    PRICE_FILTER = "PRICE_FILTER"
    COMPARISON = "COMPARISON"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    RECOMMENDATION = "RECOMMENDATION"
    PROJECT_GUIDE = "PROJECT_GUIDE"  # step-by-step build instructions for a project
    CALCULATION = "CALCULATION"


NO_ANSWER_MESSAGE = (
    "I could not find enough information in the current knowledge base to answer this question reliably."
)
NOT_FOUND_MESSAGE = "I could not find this information in the current knowledge base."
# ComponentFact.matched_by for catalog parts added to project context without a retrieval match.
INVENTORY_MATCH = "store inventory"


# ------------------------------------------------------------------ analysis
@dataclass
class PriceConstraint:
    max_price: Decimal | None = None
    max_inclusive: bool = False
    min_price: Decimal | None = None
    min_inclusive: bool = False
    approx_price: Decimal | None = None
    currency: str | None = None

    def describe(self) -> str:
        cur = f" {self.currency}" if self.currency else ""
        parts = []
        if self.max_price is not None:
            parts.append(f"{'at most' if self.max_inclusive else 'less than'} {self.max_price}{cur}")
        if self.min_price is not None:
            parts.append(f"{'at least' if self.min_inclusive else 'more than'} {self.min_price}{cur}")
        if self.approx_price is not None:
            parts.append(f"approximately {self.approx_price}{cur}")
        return " and ".join(parts)

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.max_price, self.min_price, self.approx_price))


@dataclass
class QuantityItem:
    quantity: Decimal
    mention: str


@dataclass
class QueryAnalysis:
    original: str
    normalized: str
    identifiers: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    price: PriceConstraint = field(default_factory=PriceConstraint)
    currency: str | None = None
    sort: str | None = None  # "asc" (cheapest) | "desc" (most expensive)
    quantities: list[QuantityItem] = field(default_factory=list)
    is_comparison: bool = False
    is_calculation: bool = False
    is_recommendation: bool = False
    is_build_guide: bool = False  # "step by step", "instructions", "walk me through"
    is_list_request: bool = False
    is_source_request: bool = False
    is_explanation: bool = False
    is_price_question: bool = False
    refers_to_previous: bool = False
    limit: int | None = None
    topic: str | None = None  # project / subject phrase, e.g. "digital clock"
    topic_from_history: bool = False  # topic taken from an earlier user turn ("give me the steps")
    inherited_from_history: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifiers": self.identifiers,
            "categories": self.categories,
            "price_constraint": self.price.describe() or None,
            "currency": self.currency,
            "sort": self.sort,
            "topic": self.topic,
            "topic_from_history": self.topic_from_history,
            "quantities": [{"quantity": str(q.quantity), "mention": q.mention} for q in self.quantities],
            "intents": [
                name
                for name, flag in (
                    ("comparison", self.is_comparison),
                    ("calculation", self.is_calculation),
                    ("recommendation", self.is_recommendation),
                    ("build_guide", self.is_build_guide),
                    ("list", self.is_list_request),
                    ("source_request", self.is_source_request),
                    ("explanation", self.is_explanation),
                    ("price_question", self.is_price_question),
                )
                if flag
            ],
            "inherited_from_history": self.inherited_from_history,
        }


# ---------------------------------------------------------- structured facts
@dataclass
class PriceSource:
    document_id: str | None
    document_title: str | None
    source_name: str | None
    source_type: str | None
    source_url: str | None
    price: PriceRange | None
    supplier: str | None = None


@dataclass
class ComponentFact:
    component_id: str
    name: str
    category: str
    price: PriceRange | None
    aliases: list[str] = field(default_factory=list)
    description: str | None = None
    typical_uses: list[str] = field(default_factory=list)
    sources: list[PriceSource] = field(default_factory=list)
    matched_by: str | None = None
    partial_match: bool = False  # price range only partially satisfies a filter

    @property
    def price_text(self) -> str:
        if self.price is None:
            return "price not listed"
        unit = f" per {self.price.unit}" if self.price.unit not in ("piece", "total") else ""
        return f"{self.price.format()}{unit}"

    @property
    def sources_disagree(self) -> bool:
        prices = {(s.price.min, s.price.max, s.price.currency) for s in self.sources if s.price}
        return len(prices) > 1


@dataclass
class ResolvedComponent:
    component_id: str
    name: str
    matched_text: str
    match_type: str  # alias | identifier | fuzzy
    score: float = 1.0


@dataclass
class LineItem:
    component: ComponentFact
    quantity: Decimal
    line_total: PriceRange
    mention: str
    assumed: bool = False
    alternatives: list[str] = field(default_factory=list)


@dataclass
class CalculationResult:
    items: list[LineItem]
    total: PriceRange | None
    unresolved: list[str] = field(default_factory=list)
    mixed_currency: bool = False


@dataclass
class ComparisonResult:
    components: list[ComponentFact]
    statements: list[str]


@dataclass
class PriceFilterResult:
    matches: list[ComponentFact]
    partial: list[ComponentFact]
    constraint: str
    categories: list[str]
    sort: str | None
