"""Hybrid retrieval: dense (pgvector) + lexical (FTS) fused with Reciprocal Rank Fusion.

    RRF(d) = Σ_i  w_i / (k + rank_i(d))

RRF only uses ranks, so the incomparable score scales of cosine similarity and
ts_rank_cd never need to be normalized against each other.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.cache import Cache, make_key
from app.core.config import Settings
from app.core.logging import timed
from app.core.security import Principal
from app.embeddings.base import EmbeddingProvider
from app.retrieval.keyword_search import KeywordSearcher
from app.retrieval.metadata_filter import RetrievalFilters, post_filter
from app.retrieval.types import SearchHit
from app.retrieval.vector_search import VectorSearcher


def reciprocal_rank_fusion(
    result_lists: dict[str, list[SearchHit]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[SearchHit]:
    weights = weights or {}
    fused: dict[object, SearchHit] = {}
    scores: dict[object, float] = {}
    for name, hits in result_lists.items():
        w = weights.get(name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            key = hit.chunk_id
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            if key in fused:
                # Merge scores/ranks from the other retriever into a single hit.
                fused[key].scores.update(hit.scores)
                fused[key].ranks.update(hit.ranks)
            else:
                fused[key] = hit
            fused[key].ranks.setdefault(name, rank)
    ordered = sorted(fused.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in ordered:
        hit.scores["rrf"] = scores[hit.chunk_id]
    return ordered[:top_n] if top_n else ordered


@dataclass
class HybridResult:
    hits: list[SearchHit]
    vector_candidates: int = 0
    keyword_candidates: int = 0
    fused_candidates: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    cached: bool = False


class HybridRetriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        settings: Settings,
        cache: Cache | None = None,
        vector_searcher: VectorSearcher | None = None,
        keyword_searcher: KeywordSearcher | None = None,
    ) -> None:
        self.embedder = embedder
        self.settings = settings
        self.cache = cache
        self.vector = vector_searcher or VectorSearcher(settings.hnsw_ef_search, settings.hnsw_iterative_scan)
        self.keyword = keyword_searcher or KeywordSearcher()

    def retrieve(
        self,
        session: Session,
        query: str,
        principal: Principal,
        filters: RetrievalFilters | None = None,
        *,
        extra_terms: list[str] | None = None,
        top_k: int | None = None,
    ) -> HybridResult:
        s = self.settings
        top_k = top_k or s.hybrid_top_k
        cache_key = None
        if self.cache is not None:
            cache_key = make_key(
                "hybrid",
                query,
                extra_terms,
                filters.cache_key() if filters else None,
                principal.user_id,
                principal.organization_id,
                principal.is_admin,
                top_k,
                self.embedder.model_name,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached = copy.deepcopy(cached)  # callers mutate hits (rerank scores)
                cached.cached = True
                return cached

        timings: dict[str, float] = {}
        with timed(timings, "query_embedding_ms"):
            query_vector = self.embedder.embed_query(query)
        with timed(timings, "vector_search_ms"):
            vector_hits = self.vector.search(session, query_vector, s.vector_top_k, principal, filters)
        with timed(timings, "keyword_search_ms"):
            keyword_hits = self.keyword.search(session, query, s.keyword_top_k, principal, filters, extra_terms)
        with timed(timings, "fusion_ms"):
            fused = reciprocal_rank_fusion(
                {"vector": vector_hits, "keyword": keyword_hits},
                k=s.rrf_k,
                weights={"vector": s.vector_weight, "keyword": s.keyword_weight},
            )
            filtered = post_filter(fused, filters, principal, s.min_retrieval_score)[:top_k]

        result = HybridResult(
            hits=filtered,
            vector_candidates=len(vector_hits),
            keyword_candidates=len(keyword_hits),
            fused_candidates=len(fused),
            timings=timings,
        )
        if self.cache is not None and cache_key:
            self.cache.set(cache_key, copy.deepcopy(result))
        return result
