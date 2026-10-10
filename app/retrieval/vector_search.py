"""Dense retrieval with pgvector (cosine distance, HNSW index)."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import Principal
from app.retrieval.metadata_filter import RetrievalFilters, filters_sql
from app.retrieval.types import CHUNK_SELECT_COLUMNS, SearchHit, hit_from_row


def to_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.7f}" for v in vector) + "]"


@lru_cache(maxsize=8)
def _supports_iterative_scan(version: str) -> bool:
    try:
        major, minor = (int(x) for x in version.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (0, 8)


class VectorSearcher:
    def __init__(self, ef_search: int = 64, iterative_scan: str = "relaxed_order") -> None:
        self.ef_search = int(ef_search)
        self.iterative_scan = iterative_scan
        self._pgvector_version: str | None = None

    def _configure(self, session: Session) -> None:
        # SET LOCAL lasts for the current transaction only.
        session.execute(text(f"SET LOCAL hnsw.ef_search = {self.ef_search}"))
        if self.iterative_scan != "off":
            if self._pgvector_version is None:
                self._pgvector_version = (
                    session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
                    or "0"
                )
            if _supports_iterative_scan(self._pgvector_version):
                # Keeps scanning the HNSW graph when filters remove candidates.
                session.execute(text(f"SET LOCAL hnsw.iterative_scan = {self.iterative_scan}"))

    def search(
        self,
        session: Session,
        query_embedding: list[float],
        k: int,
        principal: Principal,
        filters: RetrievalFilters | None = None,
    ) -> list[SearchHit]:
        self._configure(session)
        where, params = filters_sql(filters, principal)
        sql = text(
            f"""
            SELECT {CHUNK_SELECT_COLUMNS},
                   1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL AND d.status = 'active' AND {where}
            ORDER BY c.embedding <=> CAST(:qvec AS vector)
            LIMIT :k
            """
        )
        rows = session.execute(sql, {**params, "qvec": to_pgvector(query_embedding), "k": k}).all()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            hit = hit_from_row(row)
            hit.scores["vector_similarity"] = float(row._mapping["similarity"])
            hit.ranks["vector"] = rank
            hits.append(hit)
        return hits
