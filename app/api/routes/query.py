from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import auth, get_container
from app.container import AppContainer
from app.core.security import Principal, get_principal
from app.rag.pipeline import RAGAnswer, UserFilters
from app.rag.query_processor import ChatMessage
from app.schemas.component import ComponentOut
from app.schemas.query import (
    CalculationOut,
    LineItemOut,
    QueryFilters,
    QueryRequest,
    QueryResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
    SourceOut,
)

router = APIRouter(tags=["query"], dependencies=auth)


def to_user_filters(filters: QueryFilters | None) -> UserFilters | None:
    if filters is None:
        return None
    return UserFilters(**filters.model_dump())


def to_response(result: RAGAnswer, include_debug: bool = False) -> QueryResponse:
    calc = None
    if result.calculation:
        c = result.calculation
        calc = CalculationOut(
            items=[
                LineItemOut(
                    component=i.component.name,
                    component_id=i.component.component_id,
                    quantity=i.quantity,
                    unit_price=i.component.price.format() if i.component.price else "n/a",
                    line_total_min=i.line_total.min,
                    line_total_max=i.line_total.max,
                    line_total=i.line_total.format(),
                    currency=i.line_total.currency,
                    assumed=i.assumed,
                    alternatives=i.alternatives,
                    mention=i.mention,
                )
                for i in c.items
            ],
            total_min=c.total.min if c.total else None,
            total_max=c.total.max if c.total else None,
            total=c.total.format() if c.total else None,
            currency=c.total.currency if c.total else None,
            unresolved=c.unresolved,
        )
    debug = None
    if include_debug:
        debug = {
            "hits": [h.debug_dict() for h in result.hits],
            "context": result.context.render() if result.context else None,
        }
    return QueryResponse(
        answer=result.answer,
        query_type=result.query_type.value,
        sources=[SourceOut(**s) for s in result.sources],
        structured_results=[ComponentOut.from_fact(f) for f in result.structured_results],
        calculation=calc,
        comparison=result.comparison.statements if result.comparison else None,
        retrieval_metadata=result.retrieval_metadata,
        debug=debug,
    )


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    container: AppContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> QueryResponse:
    result = container.rag.answer(
        request.question,
        principal,
        filters=to_user_filters(request.filters),
        history=[ChatMessage(t.role, t.content) for t in request.history],
        final_k=request.final_k,
        use_llm=request.use_llm,
    )
    return to_response(result, request.include_debug)


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    container: AppContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> SearchResponse:
    hits, meta = container.rag.search(
        request.query, principal, to_user_filters(request.filters), rerank=request.rerank, top_k=request.top_k
    )
    return SearchResponse(
        hits=[
            SearchHitOut(
                chunk_id=str(h.chunk_id),
                document_id=str(h.document_id),
                document_title=h.document_title,
                source=h.source_name,
                source_type=h.source_type,
                content=h.content,
                section=h.section,
                page=h.page_number,
                metadata=h.metadata,
                scores=h.scores,
                ranks=h.ranks,
            )
            for h in hits
        ],
        retrieval_metadata=meta,
    )
