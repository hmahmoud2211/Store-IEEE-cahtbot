from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_hash(self, content_hash: str) -> Document | None:
        return self.session.scalar(select(Document).where(Document.content_hash == content_hash))

    def get_by_source(self, source_type: str, source_name: str) -> Document | None:
        return self.session.scalar(
            select(Document)
            .where(Document.source_type == source_type, Document.source_name == source_name)
            .order_by(Document.updated_at.desc())
            .limit(1)
        )

    def list(self, *, limit: int = 50, offset: int = 0, source_type: str | None = None) -> tuple[list[Document], int]:
        query = select(Document)
        count_query = select(func.count()).select_from(Document)
        if source_type:
            query = query.where(Document.source_type == source_type)
            count_query = count_query.where(Document.source_type == source_type)
        total = self.session.scalar(count_query) or 0
        rows = self.session.scalars(query.order_by(Document.created_at.desc()).limit(limit).offset(offset)).all()
        return list(rows), total

    def chunk_count(self, document_id: uuid.UUID) -> int:
        return (
            self.session.scalar(
                select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            or 0
        )

    def add(self, document: Document) -> Document:
        self.session.add(document)
        self.session.flush()
        return document

    def delete(self, document: Document) -> None:
        self.session.delete(document)
        self.session.flush()

    def delete_chunks(self, document_id: uuid.UUID) -> int:
        result = self.session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        return result.rowcount or 0
