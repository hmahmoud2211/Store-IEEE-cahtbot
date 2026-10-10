from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_many(self, chunks: Sequence[DocumentChunk]) -> None:
        self.session.add_all(chunks)
        self.session.flush()

    def list_for_document(self, document_id: uuid.UUID, limit: int = 100) -> list[DocumentChunk]:
        return list(
            self.session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(limit)
            )
        )

    def iter_all(self, batch_size: int = 256) -> Iterator[list[tuple[uuid.UUID, str]]]:
        """Yield (id, content) batches ordered by id, for re-indexing."""
        last_id: uuid.UUID | None = None
        while True:
            query = select(DocumentChunk.id, DocumentChunk.content).order_by(DocumentChunk.id).limit(batch_size)
            if last_id is not None:
                query = query.where(DocumentChunk.id > last_id)
            rows = self.session.execute(query).all()
            if not rows:
                return
            yield [(r.id, r.content) for r in rows]
            last_id = rows[-1].id

    def update_embedding(self, chunk_id: uuid.UUID, embedding: list[float], model: str, version: str) -> None:
        self.session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
            .values(embedding=embedding, embedding_model=model, embedding_version=version)
        )
