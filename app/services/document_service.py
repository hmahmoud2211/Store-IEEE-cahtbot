"""Document management: listing, deletion (with component cleanup) and re-indexing."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.core.cache import Cache
from app.core.logging import get_logger
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.components import ComponentRepository
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.jobs import JobRepository
from app.db.session import session_scope
from app.embeddings.base import EmbeddingProvider

log = get_logger(__name__)


class DocumentService:
    def __init__(self, session_factory: sessionmaker[Session], cache: Cache | None = None) -> None:
        self.session_factory = session_factory
        self.cache = cache

    def delete(self, document_id: uuid.UUID) -> bool:
        with session_scope(self.session_factory) as session:
            docs = DocumentRepository(session)
            doc = docs.get(document_id)
            if doc is None:
                return False
            docs.delete(doc)  # chunks + per-source prices cascade in the DB
            removed = ComponentRepository(session).delete_orphans()
            log.info("document_deleted", document_id=str(document_id), orphan_components_removed=removed)
        if self.cache is not None:
            self.cache.clear()
        return True

    def start_reindex(self, embedder: EmbeddingProvider) -> uuid.UUID:
        with session_scope(self.session_factory) as session:
            job = JobRepository(session).create(job_type="reindex", status="pending", source="all", **embedder.info)
            return job.id

    def reindex(self, embedder: EmbeddingProvider, job_id: uuid.UUID, batch_size: int = 128) -> int:
        """Re-embed every chunk with the currently configured embedding model."""
        total = 0
        try:
            with session_scope(self.session_factory) as session:
                jobs = JobRepository(session)
                job = jobs.get(job_id)
                if job:
                    job.status = "running"
            # Read batches in one session, write in short transactions.
            with session_scope(self.session_factory) as reader:
                for batch in ChunkRepository(reader).iter_all(batch_size):
                    vectors = embedder.embed_documents([content for _, content in batch])
                    with session_scope(self.session_factory) as writer:
                        repo = ChunkRepository(writer)
                        for (chunk_id, _), vector in zip(batch, vectors, strict=True):
                            repo.update_embedding(chunk_id, vector, embedder.model_name, embedder.version)
                    total += len(batch)
            with session_scope(self.session_factory) as session:
                jobs = JobRepository(session)
                job = jobs.get(job_id)
                if job:
                    jobs.finish(job, status="completed", chunks_created=total, stats={"chunks_reembedded": total})
        except Exception as exc:
            log.exception("reindex_failed", job_id=str(job_id))
            with session_scope(self.session_factory) as session:
                jobs = JobRepository(session)
                job = jobs.get(job_id)
                if job:
                    jobs.finish(job, status="failed", error=str(exc)[:2000], chunks_created=total)
            raise
        finally:
            if self.cache is not None:
                self.cache.clear()
        log.info("reindex_finished", job_id=str(job_id), chunks=total, model=embedder.model_name)
        return total
