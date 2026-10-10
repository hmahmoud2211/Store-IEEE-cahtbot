from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import auth, get_container, get_session
from app.container import AppContainer
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.documents import DocumentRepository
from app.schemas.document import ChunkOut, DocumentDetail, DocumentList, DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=auth)


@router.get("", response_model=DocumentList)
def list_documents(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source_type: str | None = None,
    session: Session = Depends(get_session),
) -> DocumentList:
    rows, total = DocumentRepository(session).list(limit=limit, offset=offset, source_type=source_type)
    return DocumentList(items=[DocumentOut.model_validate(r) for r in rows], total=total)


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: uuid.UUID,
    chunk_limit: int = Query(20, ge=0, le=500),
    session: Session = Depends(get_session),
) -> DocumentDetail:
    repo = DocumentRepository(session)
    doc = repo.get(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    chunks = ChunkRepository(session).list_for_document(document_id, limit=chunk_limit)
    base = DocumentOut.model_validate(doc).model_dump()
    return DocumentDetail(
        **base,
        chunk_count=repo.chunk_count(document_id),
        chunks=[ChunkOut.model_validate(c) for c in chunks],
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: uuid.UUID, container: AppContainer = Depends(get_container)) -> None:
    if not container.documents.delete(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
