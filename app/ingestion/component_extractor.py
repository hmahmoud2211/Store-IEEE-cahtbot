"""Detect component/price rows in structured elements and turn them into
`ComponentRecord`s with normalized names, aliases, categories and numeric prices.

Works for CSV/Excel/DB/API RECORD elements as well as TABLE elements coming
from PDFs, DOCX files or web pages, as long as a name column and a price
column can be identified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.categorizer import Category, categorize, parse_category
from app.ingestion.component_normalizer import normalize_component
from app.ingestion.loaders.base import Element, ElementType
from app.ingestion.price_parser import PriceRange, format_amount, parse_price

_NAME_COLUMNS = ("component", "component name", "name", "item", "product", "part", "part number", "part name", "title")
_PRICE_COLUMNS = ("approx. price", "approx price", "approximate price", "price", "unit price", "cost", "price (egp)")
_CATEGORY_COLUMNS = ("category", "type", "group")
_ALIAS_COLUMNS = ("aliases", "alias", "also known as", "aka")
_DESCRIPTION_COLUMNS = ("description", "details", "notes", "specs", "specification")
_USES_COLUMNS = ("typical_uses", "typical uses", "uses", "applications")
_CURRENCY_COLUMNS = ("currency",)
_SUPPLIER_COLUMNS = ("supplier", "vendor", "store", "shop")


def _norm_col(col: str) -> str:
    return re.sub(r"[\s_]+", " ", col.strip().lower())


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_norm_col(c): c for c in columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    for cand in candidates:  # fuzzy: "price (approx.)", "component / part"
        for norm, original in normalized.items():
            if cand in norm:
                return original
    return None


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in re.split(r"[;|\n]", value) if v.strip()]


@dataclass
class ComponentRecord:
    name: str
    normalized_name: str
    category: Category
    aliases: list[str]
    search_keys: str
    price: PriceRange | None
    description: str | None = None
    typical_uses: list[str] = field(default_factory=list)
    supplier: str | None = None
    source_row: dict[str, str] = field(default_factory=dict)

    def chunk_text(self, country: str | None = None) -> str:
        """Natural-language text that is embedded and indexed for this component."""
        parts = [f"{self.name}.", f"Category: {self.category.value}."]
        if self.aliases:
            parts.append(f"Also known as: {', '.join(self.aliases[:12])}.")
        if self.price:
            where = f" in {country}" if country else ""
            if self.price.is_range:
                amount = f"{format_amount(self.price.min)} to {format_amount(self.price.max)}"
            else:
                amount = format_amount(self.price.min)
            currency = f" {self.price.currency}" if self.price.currency else ""
            parts.append(
                f'Approximate price{where}: {amount}{currency} per {self.price.unit} (listed as "{self.price.original}").'
            )
        if self.description:
            parts.append(f"Description: {self.description.rstrip('.')}.")
        if self.typical_uses:
            parts.append(f"Typical uses: {'; '.join(self.typical_uses)}.")
        if self.supplier:
            parts.append(f"Supplier: {self.supplier}.")
        return " ".join(parts)

    def as_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "component": self.name,
            "normalized_name": self.normalized_name,
            "category": self.category.value,
            "aliases": self.aliases,
        }
        if self.price:
            meta.update(self.price.as_dict())
        if self.supplier:
            meta["supplier"] = self.supplier
        return meta


class ComponentExtractor:
    def __init__(self, default_currency: str | None = "EGP") -> None:
        self.default_currency = default_currency

    def detect_columns(self, columns: list[str]) -> dict[str, str | None] | None:
        name_col = _find_column(columns, _NAME_COLUMNS)
        price_col = _find_column(columns, _PRICE_COLUMNS)
        if not name_col or not price_col or name_col == price_col:
            return None
        return {
            "name": name_col,
            "price": price_col,
            "category": _find_column(columns, _CATEGORY_COLUMNS),
            "aliases": _find_column(columns, _ALIAS_COLUMNS),
            "description": _find_column(columns, _DESCRIPTION_COLUMNS),
            "uses": _find_column(columns, _USES_COLUMNS),
            "currency": _find_column(columns, _CURRENCY_COLUMNS),
            "supplier": _find_column(columns, _SUPPLIER_COLUMNS),
        }

    def from_row(
        self, row: dict[str, str], columns: dict[str, str | None], supplier: str | None = None
    ) -> ComponentRecord | None:
        name = (row.get(columns["name"] or "") or "").strip()
        price_text = (row.get(columns["price"] or "") or "").strip()
        if not name:
            return None

        def col(key: str) -> str | None:
            c = columns.get(key)
            value = row.get(c) if c else None
            return value.strip() if value else None

        currency_hint = col("currency")
        price = parse_price(price_text, default_currency=None)
        if price and price.currency is None:
            from dataclasses import replace

            price = replace(price, currency=(currency_hint or self.default_currency))

        description = col("description")
        normalized = normalize_component(name, extra_aliases=_split_list(col("aliases")))
        category = parse_category(col("category")) or categorize(name, description)
        return ComponentRecord(
            name=normalized.name,
            normalized_name=normalized.normalized_name,
            category=category,
            aliases=normalized.aliases,
            search_keys=normalized.search_keys,
            price=price,
            description=description,
            typical_uses=_split_list(col("uses")),
            supplier=col("supplier") or supplier,
            source_row=dict(row),
        )

    def extract(self, elements: list[Element], supplier: str | None = None) -> dict[int, list[ComponentRecord]]:
        """Return {element_index: [ComponentRecord, ...]} for every component-bearing element."""
        found: dict[int, list[ComponentRecord]] = {}
        for idx, element in enumerate(elements):
            if element.type not in (ElementType.RECORD, ElementType.TABLE) or not element.rows:
                continue
            columns = self.detect_columns(list(element.rows[0].keys()))
            if not columns:
                continue
            records = [r for r in (self.from_row(row, columns, supplier) for row in element.rows) if r]
            if records:
                found[idx] = records
        return found
