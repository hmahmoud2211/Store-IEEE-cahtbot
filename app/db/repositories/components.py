from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.orm import Session, selectinload

from app.db.models import Component, ComponentPrice, Document


@dataclass
class PriceBound:
    """A numeric price constraint. `inclusive` controls < vs <= (or > vs >=)."""

    value: Decimal
    inclusive: bool = False


@dataclass
class ComponentQuery:
    categories: Sequence[str] | None = None
    component_ids: Sequence[uuid.UUID] | None = None
    max_price: PriceBound | None = None
    min_price: PriceBound | None = None
    approx_price: Decimal | None = None
    approx_tolerance: float = 0.10
    currency: str | None = None
    text: str | None = None
    sort: str | None = None  # "asc" | "desc" | None
    limit: int = 100
    offset: int = 0


class ComponentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ----------------------------------------------------------------- reads
    def get(self, component_id: uuid.UUID) -> Component | None:
        return self.session.scalar(
            select(Component)
            .where(Component.id == component_id)
            .options(selectinload(Component.prices).selectinload(ComponentPrice.source_document))
        )

    def get_by_normalized_name(self, normalized_name: str) -> Component | None:
        return self.session.scalar(select(Component).where(Component.normalized_name == normalized_name))

    def get_many(self, ids: Sequence[uuid.UUID]) -> list[Component]:
        if not ids:
            return []
        rows = self.session.scalars(
            select(Component)
            .where(Component.id.in_(list(ids)))
            .options(
                selectinload(Component.prices).selectinload(ComponentPrice.source_document),
                selectinload(Component.source_document),
            )
        ).all()
        order = {cid: i for i, cid in enumerate(ids)}
        return sorted(rows, key=lambda c: order.get(c.id, 0))

    def all_for_index(self) -> list[tuple[uuid.UUID, str, str, list[str], str | None, Decimal | None]]:
        """(id, name, normalized_name, aliases, search_keys, price_min) for building the alias index."""
        rows = self.session.execute(
            select(
                Component.id,
                Component.name,
                Component.normalized_name,
                Component.aliases,
                Component.search_keys,
                Component.price_min,
            )
        ).all()
        return [tuple(r) for r in rows]

    def _price_conditions(self, q: ComponentQuery) -> tuple[list, list]:
        """Return (definite_conditions, partial_conditions) for price constraints.

        Definite: the whole price range satisfies the constraint.
        Partial:  the range straddles the bound (e.g. 15–25 EGP vs "under 25").
        """
        definite, partial = [], []
        if q.max_price is not None:
            v = q.max_price.value
            if q.max_price.inclusive:
                definite.append(Component.price_max <= v)
                partial.append(and_(Component.price_min <= v, Component.price_max > v))
            else:
                definite.append(Component.price_max < v)
                partial.append(and_(Component.price_min < v, Component.price_max >= v))
        if q.min_price is not None:
            v = q.min_price.value
            if q.min_price.inclusive:
                definite.append(Component.price_min >= v)
                partial.append(and_(Component.price_max >= v, Component.price_min < v))
            else:
                definite.append(Component.price_min > v)
                partial.append(and_(Component.price_max > v, Component.price_min <= v))
        if q.approx_price is not None:
            tol = Decimal(str(q.approx_tolerance))
            low, high = q.approx_price * (1 - tol), q.approx_price * (1 + tol)
            definite.append(and_(Component.price_min <= high, Component.price_max >= low))
        return definite, partial

    def search(self, q: ComponentQuery) -> tuple[list[Component], list[Component]]:
        """Structured search. Returns (definite_matches, partial_matches)."""
        base = select(Component).options(
            selectinload(Component.prices).selectinload(ComponentPrice.source_document),
            selectinload(Component.source_document),
        )
        conditions = []
        if q.categories:
            conditions.append(func.lower(Component.category).in_([c.lower() for c in q.categories]))
        if q.component_ids:
            conditions.append(Component.id.in_(list(q.component_ids)))
        if q.currency:
            conditions.append(Component.currency == q.currency.upper())
        if q.text:
            like = f"%{q.text.lower()}%"
            conditions.append(
                or_(
                    Component.normalized_name.like(like),
                    func.array_to_string(Component.aliases, " ").ilike(like),
                    Component.search_keys.like(like),
                )
            )
        definite_price, partial_price = self._price_conditions(q)
        has_price = bool(definite_price)
        if has_price:
            conditions.append(Component.price_min.is_not(None))

        def ordered(stmt):
            if q.approx_price is not None and q.sort is None:  # closest to the target first
                mid = (Component.price_min + Component.price_max) / 2
                return stmt.order_by(func.abs(mid - q.approx_price), Component.price_min)
            if q.sort == "desc":
                return stmt.order_by(Component.price_max.desc().nulls_last(), Component.price_min.desc())
            if q.sort == "asc" or has_price:
                return stmt.order_by(Component.price_min.asc().nulls_last(), Component.price_max.asc())
            return stmt.order_by(Component.category, Component.name)

        definite = self.session.scalars(
            ordered(base.where(and_(*conditions, *definite_price))).limit(q.limit).offset(q.offset)
        ).all()
        partial: Sequence[Component] = []
        if partial_price:
            partial = self.session.scalars(
                ordered(base.where(and_(*conditions, or_(*partial_price)))).limit(q.limit)
            ).all()
        return list(definite), list(partial)

    def count(self, q: ComponentQuery) -> int:
        definite, _ = self.search(ComponentQuery(**{**q.__dict__, "limit": 100000, "offset": 0}))
        return len(definite)

    def fuzzy_search(self, phrase: str, limit: int = 5, threshold: float = 0.3) -> list[tuple[Component, float]]:
        """Trigram similarity over names, aliases and identifier keys (pg_trgm)."""
        score = func.greatest(
            func.similarity(Component.normalized_name, phrase),
            func.word_similarity(phrase, Component.normalized_name),
            func.word_similarity(phrase, func.array_to_string(Component.aliases, " ")),
        )
        rows = self.session.execute(
            select(Component, score.label("score"))
            .where(score >= threshold)
            .order_by(text("score DESC"), Component.price_min.asc().nulls_last())
            .limit(limit)
        ).all()
        return [(r[0], float(r[1])) for r in rows]

    def prices_with_sources(self, component_id: uuid.UUID) -> list[tuple[ComponentPrice, Document | None]]:
        rows = self.session.execute(
            select(ComponentPrice, Document)
            .outerjoin(Document, Document.id == ComponentPrice.source_document_id)
            .where(ComponentPrice.component_id == component_id)
            .order_by(ComponentPrice.observed_at.desc())
        ).all()
        return [(r[0], r[1]) for r in rows]

    # ---------------------------------------------------------------- writes
    def upsert(self, values: dict, source_document_id: uuid.UUID | None) -> Component:
        """Insert or update a component by normalized_name; record the per-source price."""
        component = self.get_by_normalized_name(values["normalized_name"])
        if component is None:
            component = Component(id=uuid.uuid4(), **values, source_document_id=source_document_id)
            self.session.add(component)
        else:
            merged_aliases = list(dict.fromkeys([*(component.aliases or []), *values.get("aliases", [])]))
            for key, value in values.items():
                if key == "aliases":
                    component.aliases = merged_aliases
                elif key == "typical_uses" and not value or key == "description" and not value:
                    continue
                elif key == "meta":
                    component.meta = {**(component.meta or {}), **value}
                else:
                    setattr(component, key, value)
            component.source_document_id = source_document_id
        self.session.flush()

        if values.get("price_min") is not None:
            existing = self.session.scalar(
                select(ComponentPrice).where(
                    ComponentPrice.component_id == component.id,
                    ComponentPrice.source_document_id == source_document_id,
                )
            )
            price_fields = dict(
                price_min=values.get("price_min"),
                price_max=values.get("price_max"),
                currency=values.get("currency"),
                price_unit=values.get("price_unit"),
                price_text=values.get("price_text"),
                supplier=(values.get("meta") or {}).get("supplier"),
            )
            if existing:
                for k, v in price_fields.items():
                    setattr(existing, k, v)
            else:
                self.session.add(
                    ComponentPrice(
                        id=uuid.uuid4(),
                        component_id=component.id,
                        source_document_id=source_document_id,
                        **price_fields,
                    )
                )
            self.session.flush()
        return component

    def delete_orphans(self) -> int:
        """Remove components that no longer have any price source or source document."""
        orphan_ids = self.session.scalars(
            select(Component.id).where(
                Component.source_document_id.is_(None),
                ~select(ComponentPrice.id).where(ComponentPrice.component_id == Component.id).exists(),
            )
        ).all()
        for cid in orphan_ids:
            obj = self.session.get(Component, cid)
            if obj:
                self.session.delete(obj)
        self.session.flush()
        return len(orphan_ids)
