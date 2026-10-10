"""Structured component access: alias resolution, SQL filtering, fact assembly.

Resolution order for a question:
  1. Alias/name phrase matching (longest phrase first) against an in-memory
     index built from the components table ("555 timer", "IR obstacle sensor").
  2. Identifier matching on compact equivalence keys ("7485", "HC SR04" -> hcsr04).
  3. (Item names in calculations only) pg_trgm fuzzy matching.
Identifiers that resolve to nothing are reported as NOT FOUND — never guessed.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.cache import Cache, make_key
from app.db.models import Component
from app.db.repositories.components import ComponentQuery, ComponentRepository
from app.ingestion.component_normalizer import (
    compact_key,
    extract_identifiers,
    identifier_variants,
    phrase_key,
)
from app.ingestion.price_parser import PriceRange
from app.rag.types import ComponentFact, PriceSource, ResolvedComponent


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass
class Resolution:
    resolved: list[ResolvedComponent]
    unresolved: list[str]


@dataclass
class _Entry:
    name: str
    normalized_name: str
    price_min: Decimal | None


@dataclass
class AliasIndex:
    phrases: dict[str, set[str]] = field(default_factory=dict)
    compact: dict[str, set[str]] = field(default_factory=dict)
    entries: dict[str, _Entry] = field(default_factory=dict)
    max_words: int = 1

    @classmethod
    def build(cls, rows: Iterable[tuple]) -> AliasIndex:
        index = cls()
        for cid, name, normalized, aliases, search_keys, price_min in rows:
            cid = str(cid)
            index.entries[cid] = _Entry(name, normalized, price_min)
            for text in [name, normalized, *(aliases or [])]:
                key = phrase_key(text)
                if not key:
                    continue
                index._add_phrase(key, cid)
                tokens = key.split()
                if tokens[-1].isalpha() and not tokens[-1].endswith("s"):
                    index._add_phrase(" ".join([*tokens[:-1], tokens[-1] + "s"]), cid)
            for k in (search_keys or "").split():
                index.compact.setdefault(k, set()).add(cid)
        return index

    def _add_phrase(self, key: str, cid: str) -> None:
        self.phrases.setdefault(key, set()).add(cid)
        self.max_words = max(self.max_words, len(key.split()))

    def match_phrases(self, text: str) -> list[tuple[str, set[str]]]:
        """Greedy longest-match scan. Returns [(matched_phrase, component_ids)] in text order."""
        tokens = phrase_key(text).split()
        consumed = [False] * len(tokens)
        found: list[tuple[int, str, set[str]]] = []
        for n in range(min(self.max_words, len(tokens)), 0, -1):
            for i in range(0, len(tokens) - n + 1):
                if any(consumed[i : i + n]):
                    continue
                phrase = " ".join(tokens[i : i + n])
                ids = self.phrases.get(phrase)
                if ids is None and n == 1:
                    ids = self.phrases.get(_singular(phrase))
                if ids:
                    found.append((i, phrase, ids))
                    for j in range(i, i + n):
                        consumed[j] = True
        found.sort(key=lambda f: f[0])
        return [(phrase, ids) for _, phrase, ids in found]

    def match_identifier(self, identifier: str) -> set[str]:
        ids: set[str] = set()
        for variant in identifier_variants(identifier):
            ids |= self.compact.get(variant, set())
        return ids

    def resolve(self, text: str, identifiers: list[str] | None = None) -> Resolution:
        """Resolve component mentions in `text`: alias phrases first, then part numbers."""
        resolved: dict[str, ResolvedComponent] = {}
        covered_keys: set[str] = set()
        for phrase, ids in self.match_phrases(text):
            for ident in extract_identifiers(phrase):
                covered_keys |= identifier_variants(ident)
            covered_keys.add(compact_key(phrase))
            for cid in sorted(ids, key=lambda c: self.entries[c].name):
                resolved.setdefault(cid, ResolvedComponent(cid, self.entries[cid].name, phrase, "alias", 1.0))

        unresolved: list[str] = []
        for ident in identifiers if identifiers is not None else extract_identifiers(text):
            if identifier_variants(ident) & covered_keys:
                continue
            ids = self.match_identifier(ident)
            if not ids:
                unresolved.append(ident)
                continue
            for cid in sorted(ids, key=lambda c: self.entries[c].name):
                resolved.setdefault(cid, ResolvedComponent(cid, self.entries[cid].name, ident, "identifier", 0.9))
        return Resolution(list(resolved.values()), unresolved)

    def name_overlap(self, mention: str, cid: str) -> float:
        entry = self.entries[cid]
        name_tokens = {_singular(t) for t in phrase_key(entry.name).split()}
        mention_tokens = {_singular(t) for t in phrase_key(mention).split()}
        if not name_tokens:
            return 0.0
        return len(name_tokens & mention_tokens) / len(name_tokens)


def component_to_fact(component: Component, matched_by: str | None = None, partial: bool = False) -> ComponentFact:
    price = None
    if component.price_min is not None and component.price_max is not None:
        price = PriceRange(
            min=Decimal(component.price_min),
            max=Decimal(component.price_max),
            currency=component.currency,
            unit=component.price_unit or "piece",
            original=component.price_text or "",
        )
    sources: list[PriceSource] = []
    for p in sorted(component.prices or [], key=lambda p: p.observed_at, reverse=True):
        doc = p.source_document
        p_range = None
        if p.price_min is not None and p.price_max is not None:
            p_range = PriceRange(
                Decimal(p.price_min), Decimal(p.price_max), p.currency, p.price_unit or "piece", p.price_text or ""
            )
        sources.append(
            PriceSource(
                document_id=str(doc.id) if doc else None,
                document_title=doc.title if doc else None,
                source_name=doc.source_name if doc else None,
                source_type=doc.source_type if doc else None,
                source_url=doc.source_url if doc else None,
                price=p_range,
                supplier=p.supplier,
            )
        )
    if not sources and component.source_document is not None:
        doc = component.source_document
        sources.append(PriceSource(str(doc.id), doc.title, doc.source_name, doc.source_type, doc.source_url, price))
    return ComponentFact(
        component_id=str(component.id),
        name=component.name,
        category=component.category,
        price=price,
        aliases=list(component.aliases or []),
        description=component.description,
        typical_uses=list(component.typical_uses or []),
        sources=sources,
        matched_by=matched_by,
        partial_match=partial,
    )


class ComponentService:
    def __init__(self, session: Session, cache: Cache | None = None) -> None:
        self.session = session
        self.repo = ComponentRepository(session)
        self.cache = cache

    # ----------------------------------------------------------- alias index
    def alias_index(self) -> AliasIndex:
        key = make_key("alias_index", "v1")
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        index = AliasIndex.build(self.repo.all_for_index())
        if self.cache is not None:
            self.cache.set(key, index)
        return index

    # ------------------------------------------------------------ resolution
    def resolve(self, text: str, identifiers: list[str] | None = None) -> Resolution:
        return self.alias_index().resolve(text, identifiers)

    def resolve_item(self, mention: str) -> tuple[ResolvedComponent | None, list[str]]:
        """Resolve one calculation line item to a single component (+ alternatives)."""
        index = self.alias_index()
        candidates = self.resolve(mention).resolved
        if not candidates:
            for comp, score in self.repo.fuzzy_search(phrase_key(mention), limit=5, threshold=0.45):
                cid = str(comp.id)
                if cid in index.entries:
                    candidates.append(ResolvedComponent(cid, comp.name, mention, "fuzzy", score))
        if not candidates:
            return None, []

        def rank(rc: ResolvedComponent) -> tuple:
            entry = index.entries[rc.component_id]
            price = entry.price_min if entry.price_min is not None else Decimal("1e12")
            return (-rc.score, -index.name_overlap(mention, rc.component_id), price)

        ordered = sorted(candidates, key=rank)
        best = ordered[0]
        alternatives = [c.name for c in ordered[1:4]]
        return best, alternatives

    # ----------------------------------------------------------------- facts
    def facts(self, resolved: list[ResolvedComponent]) -> list[ComponentFact]:
        ids = [uuid.UUID(r.component_id) for r in resolved]
        by_id = {str(c.id): c for c in self.repo.get_many(ids)}
        facts: list[ComponentFact] = []
        for r in resolved:
            comp = by_id.get(r.component_id)
            if comp is not None:
                how = "hybrid retrieval" if r.match_type == "retrieval" else f"{r.match_type} '{r.matched_text}'"
                facts.append(component_to_fact(comp, matched_by=how))
        return facts

    def search(self, query: ComponentQuery) -> tuple[list[ComponentFact], list[ComponentFact]]:
        definite, partial = self.repo.search(query)
        return (
            [component_to_fact(c) for c in definite],
            [component_to_fact(c, partial=True) for c in partial],
        )

    def get(self, component_id: uuid.UUID) -> ComponentFact | None:
        comp = self.repo.get(component_id)
        return component_to_fact(comp) if comp else None


def clean_mention(mention: str) -> str:
    return re.sub(r"\s+", " ", mention).strip()
