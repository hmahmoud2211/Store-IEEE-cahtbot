"""Lexical retrieval with PostgreSQL full-text search (tsvector / tsquery / GIN).

Each chunk's `content_tsv` combines three weighted vectors:
  A  'simple'  over search_keys  (compact identifiers: hcsr04, 7485, ne555…)
  B  'simple'  over content      (exact tokens: 74ls85, hc-sr04, ds3231…)
  C  'english' over content      (stemmed words: counters -> counter)

The query side ORs significant terms instead of AND-ing every word of a natural
question, and gives an extra boost to chunks that match an extracted part number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import Principal
from app.ingestion.component_normalizer import extract_identifiers, identifier_variants
from app.retrieval.metadata_filter import RetrievalFilters, filters_sql
from app.retrieval.types import CHUNK_SELECT_COLUMNS, SearchHit, hit_from_row

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "into",
        "about",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "much",
        "many",
        "me",
        "my",
        "i",
        "you",
        "your",
        "we",
        "our",
        "us",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "need",
        "needs",
        "needed",
        "want",
        "please",
        "give",
        "show",
        "tell",
        "list",
        "find",
        "get",
        "all",
        "any",
        "some",
        "each",
        "every",
        "other",
        "more",
        "less",
        "than",
        "then",
        "there",
        "here",
        "also",
        "just",
        "only",
        "very",
        "most",
        "least",
        "cheaper",
        "cheapest",
        "expensive",
        "price",
        "prices",
        "priced",
        "cost",
        "costs",
        "costing",
        "egp",
        "le",
        "pounds",
        "approximately",
        "approx",
        "around",
        "about",
        "roughly",
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference",
        "between",
        "related",
        "use",
        "using",
        "used",
        "build",
        "make",
        "making",
        "create",
        "design",
        "circuit",
        "component",
        "components",
        "part",
        "parts",
        "item",
        "items",
        "thing",
        "things",
        "one",
        "ones",
        "available",
        "sell",
        "sold",
        "buy",
        "kb",
        "knowledge",
        "base",
        "source",
        "sources",
        "show",
        "cite",
    ]
)
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class KeywordQuery:
    simple: str  # tsquery string for the 'simple' config
    english: str  # tsquery string for the 'english' config
    identifiers: str  # tsquery string of identifier keys only (boost)

    @property
    def empty(self) -> bool:
        return not (self.simple or self.english)


def build_keyword_query(question: str, extra_terms: list[str] | None = None) -> KeywordQuery:
    identifier_keys: set[str] = set()
    for ident in extract_identifiers(question):
        identifier_keys |= identifier_variants(ident)
    for term in extra_terms or []:
        for ident in extract_identifiers(term):
            identifier_keys |= identifier_variants(ident)

    words = [w for w in _WORD_RE.findall(question.lower()) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]
    for term in extra_terms or []:
        words.extend(w for w in _WORD_RE.findall(term.lower()) if w not in STOPWORDS)
    words = list(dict.fromkeys(words))

    simple_terms = list(dict.fromkeys([*sorted(identifier_keys), *words]))
    english_terms = [w for w in words if not w.isdigit()]
    return KeywordQuery(
        simple=" | ".join(simple_terms),
        english=" | ".join(english_terms),
        identifiers=" | ".join(sorted(identifier_keys)),
    )


class KeywordSearcher:
    def search(
        self,
        session: Session,
        question: str,
        k: int,
        principal: Principal,
        filters: RetrievalFilters | None = None,
        extra_terms: list[str] | None = None,
    ) -> list[SearchHit]:
        kq = build_keyword_query(question, extra_terms)
        if kq.empty:
            return []

        parts = []
        if kq.simple:
            parts.append("to_tsquery('simple', :q_simple)")
        if kq.english:
            parts.append("to_tsquery('english', :q_english)")
        tsquery = " || ".join(parts)
        boost = (
            "CASE WHEN c.content_tsv @@ to_tsquery('simple', :q_ident) THEN 1.0 ELSE 0 END" if kq.identifiers else "0"
        )

        where, params = filters_sql(filters, principal)
        sql = text(
            f"""
            WITH q AS (SELECT ({tsquery}) AS query)
            SELECT {CHUNK_SELECT_COLUMNS},
                   ts_rank_cd(c.content_tsv, q.query, 32) + {boost} AS rank
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            CROSS JOIN q
            WHERE c.content_tsv @@ q.query AND d.status = 'active' AND {where}
            ORDER BY rank DESC, c.id
            LIMIT :k
            """
        )
        bind = {**params, "k": k}
        if kq.simple:
            bind["q_simple"] = kq.simple
        if kq.english:
            bind["q_english"] = kq.english
        if kq.identifiers:
            bind["q_ident"] = kq.identifiers

        rows = session.execute(sql, bind).all()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            hit = hit_from_row(row)
            hit.scores["keyword_rank"] = float(row._mapping["rank"])
            hit.ranks["keyword"] = rank
            hits.append(hit)
        return hits
