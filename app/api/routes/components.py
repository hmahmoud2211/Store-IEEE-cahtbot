from __future__ import annotations

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import auth, get_container, get_session
from app.container import AppContainer
from app.db.repositories.components import ComponentQuery, PriceBound
from app.ingestion.categorizer import parse_category
from app.schemas.component import ComponentList, ComponentOut
from app.services.component_service import ComponentService

router = APIRouter(prefix="/components", tags=["components"], dependencies=auth)


@router.get("", response_model=ComponentList)
def list_components(
    category: list[str] | None = Query(default=None),
    max_price: Decimal | None = Query(default=None, description="Components whose whole price range is <= max_price"),
    min_price: Decimal | None = Query(default=None, description="Components whose whole price range is >= min_price"),
    currency: str | None = None,
    q: str | None = Query(default=None, description="Substring match on name / aliases / part numbers"),
    sort: str | None = Query(default=None, pattern="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
    container: AppContainer = Depends(get_container),
) -> ComponentList:
    categories = None
    if category:
        categories = [(parse_category(c).value if parse_category(c) else c) for c in category]
    query = ComponentQuery(
        categories=categories,
        max_price=PriceBound(max_price, inclusive=True) if max_price is not None else None,
        min_price=PriceBound(min_price, inclusive=True) if min_price is not None else None,
        currency=currency,
        text=q,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    service = ComponentService(session, container.cache)
    matches, partial = service.search(query)
    return ComponentList(
        items=[ComponentOut.from_fact(f) for f in matches],
        total=service.repo.count(query),
        partial_matches=[ComponentOut.from_fact(f) for f in partial],
    )


@router.get("/{component_id}", response_model=ComponentOut)
def get_component(
    component_id: uuid.UUID,
    session: Session = Depends(get_session),
    container: AppContainer = Depends(get_container),
) -> ComponentOut:
    fact = ComponentService(session, container.cache).get(component_id)
    if fact is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Component not found")
    return ComponentOut.from_fact(fact)
