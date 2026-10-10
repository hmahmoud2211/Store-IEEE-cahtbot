from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from app.rag.types import ComponentFact


class PriceSourceOut(BaseModel):
    document_id: str | None
    document_title: str | None
    source_name: str | None
    source_type: str | None
    source_url: str | None
    price_min: Decimal | None
    price_max: Decimal | None
    currency: str | None
    price_text: str | None
    supplier: str | None


class ComponentOut(BaseModel):
    id: str
    name: str
    category: str
    aliases: list[str]
    price_min: Decimal | None
    price_max: Decimal | None
    currency: str | None
    price_unit: str | None
    price_text: str | None
    description: str | None
    typical_uses: list[str]
    partial_match: bool = False
    matched_by: str | None = None
    sources: list[PriceSourceOut]

    @classmethod
    def from_fact(cls, fact: ComponentFact) -> ComponentOut:
        p = fact.price
        return cls(
            id=fact.component_id,
            name=fact.name,
            category=fact.category,
            aliases=fact.aliases,
            price_min=p.min if p else None,
            price_max=p.max if p else None,
            currency=p.currency if p else None,
            price_unit=p.unit if p else None,
            price_text=p.original if p else None,
            description=fact.description,
            typical_uses=fact.typical_uses,
            partial_match=fact.partial_match,
            matched_by=fact.matched_by,
            sources=[
                PriceSourceOut(
                    document_id=s.document_id,
                    document_title=s.document_title,
                    source_name=s.source_name,
                    source_type=s.source_type,
                    source_url=s.source_url,
                    price_min=s.price.min if s.price else None,
                    price_max=s.price.max if s.price else None,
                    currency=s.price.currency if s.price else None,
                    price_text=s.price.original if s.price else None,
                    supplier=s.supplier,
                )
                for s in fact.sources
            ],
        )


class ComponentList(BaseModel):
    items: list[ComponentOut]
    total: int
    partial_matches: list[ComponentOut] = []
