from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.component import ComponentOut


class QueryFilters(BaseModel):
    categories: list[str] | None = None
    currency: str | None = Field(default=None, examples=["EGP"])
    max_price: Decimal | None = None
    min_price: Decimal | None = None
    document_ids: list[uuid.UUID] | None = None
    source_types: list[str] | None = None
    supplier: str | None = None
    country: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["Which counter costs less than 25 EGP?"])
    filters: QueryFilters | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    final_k: int | None = Field(default=None, ge=1, le=8, description="Override FINAL_CONTEXT_K (3–8 recommended)")
    use_llm: bool = Field(default=True, description="False = deterministic extractive answer, no LLM call")
    include_debug: bool = False


class SourceOut(BaseModel):
    source_id: int
    label: str
    kind: str
    document: str | None = None
    document_title: str | None = None
    document_id: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    component: str | None = None
    component_id: str | None = None
    page: int | None = None
    section: str | None = None
    price_text: str | None = None
    chunk_id: str | None = None
    score: float | None = None


class LineItemOut(BaseModel):
    component: str
    component_id: str
    quantity: Decimal
    unit_price: str
    line_total_min: Decimal
    line_total_max: Decimal
    line_total: str
    currency: str | None
    assumed: bool
    alternatives: list[str]
    mention: str


class CalculationOut(BaseModel):
    items: list[LineItemOut]
    total_min: Decimal | None
    total_max: Decimal | None
    total: str | None
    currency: str | None
    unresolved: list[str]


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: list[SourceOut]
    structured_results: list[ComponentOut] = Field(default_factory=list)
    calculation: CalculationOut | None = None
    comparison: list[str] | None = None
    retrieval_metadata: dict[str, Any]
    debug: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    filters: QueryFilters | None = None
    top_k: int = Field(default=5, ge=1, le=30)
    rerank: bool = True


class SearchHitOut(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str | None
    source: str | None
    source_type: str | None
    content: str
    section: str | None
    page: int | None
    metadata: dict[str, Any]
    scores: dict[str, float]
    ranks: dict[str, int]


class SearchResponse(BaseModel):
    hits: list[SearchHitOut]
    retrieval_metadata: dict[str, Any]
