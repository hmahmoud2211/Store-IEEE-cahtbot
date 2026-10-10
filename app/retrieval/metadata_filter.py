"""Metadata and permission filtering.

Filters are compiled into SQL WHERE fragments and applied INSIDE both the
vector and the keyword queries (pre-filtering), so unauthorized or irrelevant
chunks never become candidates. `post_filter` re-checks the same constraints
in Python after fusion (defense in depth) and applies score gating.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.core.security import Principal
from app.retrieval.types import SearchHit


@dataclass
class RetrievalFilters:
    categories: list[str] | None = None
    component_ids: list[str] | None = None
    document_ids: list[uuid.UUID] | None = None
    source_types: list[str] | None = None
    currency: str | None = None
    max_price: Decimal | None = None
    min_price: Decimal | None = None
    supplier: str | None = None
    country: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(
            [
                self.categories,
                self.component_ids,
                self.document_ids,
                self.source_types,
                self.currency,
                self.max_price is not None,
                self.min_price is not None,
                self.supplier,
                self.country,
                self.metadata,
            ]
        )

    def cache_key(self) -> dict[str, Any]:
        return {k: (sorted(map(str, v)) if isinstance(v, list) else v) for k, v in self.__dict__.items() if v}


def acl_sql(principal: Principal, alias: str = "c") -> tuple[str, dict[str, Any]]:
    """Visibility rules: public, same organization, or owned by the user. Admins see everything."""
    if principal.is_admin:
        return "TRUE", {}
    clauses = [f"{alias}.visibility = 'public'"]
    params: dict[str, Any] = {}
    if principal.organization_id:
        clauses.append(f"({alias}.visibility = 'organization' AND {alias}.organization_id = :acl_org)")
        params["acl_org"] = principal.organization_id
    if principal.user_id:
        clauses.append(f"{alias}.owner_user_id = :acl_user")
        params["acl_user"] = principal.user_id
    return "(" + " OR ".join(clauses) + ")", params


def filters_sql(filters: RetrievalFilters | None, principal: Principal, alias: str = "c") -> tuple[str, dict[str, Any]]:
    acl, params = acl_sql(principal, alias)
    clauses = [acl]
    if filters is None:
        return " AND ".join(clauses), params
    if filters.categories:
        clauses.append(f"lower({alias}.metadata->>'category') = ANY(:f_categories)")
        params["f_categories"] = [c.lower() for c in filters.categories]
    if filters.component_ids:
        clauses.append(f"({alias}.metadata->>'component_id') = ANY(:f_component_ids)")
        params["f_component_ids"] = [str(c) for c in filters.component_ids]
    if filters.document_ids:
        clauses.append(f"{alias}.document_id = ANY(:f_document_ids)")
        params["f_document_ids"] = list(filters.document_ids)
    if filters.source_types:
        clauses.append(f"({alias}.metadata->>'source_type') = ANY(:f_source_types)")
        params["f_source_types"] = list(filters.source_types)
    if filters.currency:
        clauses.append(f"({alias}.metadata->>'currency') = :f_currency")
        params["f_currency"] = filters.currency.upper()
    if filters.max_price is not None:
        clauses.append(f"({alias}.metadata->>'price_min')::numeric <= :f_max_price")
        params["f_max_price"] = filters.max_price
    if filters.min_price is not None:
        clauses.append(f"({alias}.metadata->>'price_max')::numeric >= :f_min_price")
        params["f_min_price"] = filters.min_price
    if filters.supplier:
        clauses.append(f"lower({alias}.metadata->>'supplier') = :f_supplier")
        params["f_supplier"] = filters.supplier.lower()
    if filters.country:
        clauses.append(f"lower({alias}.metadata->>'country') = :f_country")
        params["f_country"] = filters.country.lower()
    if filters.metadata:
        clauses.append(f"{alias}.metadata @> CAST(:f_metadata AS jsonb)")
        params["f_metadata"] = json.dumps(filters.metadata)
    return " AND ".join(clauses), params


def _allowed(hit: SearchHit, principal: Principal) -> bool:
    if principal.is_admin or hit.visibility in (None, "public"):
        return True
    if (
        hit.visibility == "organization"
        and principal.organization_id
        and hit.organization_id == principal.organization_id
    ):
        return True
    return bool(principal.user_id and hit.owner_user_id == principal.user_id)


def _matches(hit: SearchHit, f: RetrievalFilters) -> bool:
    m = hit.metadata
    if f.categories and str(m.get("category", "")).lower() not in {c.lower() for c in f.categories}:
        return False
    if f.component_ids and str(m.get("component_id")) not in {str(c) for c in f.component_ids}:
        return False
    if f.document_ids and hit.document_id not in set(f.document_ids):
        return False
    if f.currency and m.get("currency") != f.currency.upper():
        return False
    if f.max_price is not None and (m.get("price_min") is None or Decimal(str(m["price_min"])) > f.max_price):
        return False
    if f.min_price is not None and (m.get("price_max") is None or Decimal(str(m["price_max"])) < f.min_price):
        return False
    return all(m.get(k) == v for k, v in f.metadata.items())


def post_filter(
    hits: list[SearchHit],
    filters: RetrievalFilters | None,
    principal: Principal,
    min_vector_similarity: float,
) -> list[SearchHit]:
    """Re-validate permissions/filters and drop weak candidates.

    A candidate is kept when it matched lexically (keyword search) or when its
    dense similarity reaches `min_vector_similarity`.
    """
    kept: list[SearchHit] = []
    for hit in hits:
        if not _allowed(hit, principal):
            continue
        if filters is not None and not _matches(hit, filters):
            continue
        lexical = "keyword" in hit.ranks
        similarity = hit.scores.get("vector_similarity", 0.0)
        if not lexical and similarity < min_vector_similarity:
            continue
        kept.append(hit)
    return kept
