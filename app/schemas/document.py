from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    source: str | dict[str, Any] = Field(
        description="File path, URL, or a dict spec for database/api sources",
        examples=["data/hardware_components.csv", "https://example.com/catalog"],
    )
    source_type: str | None = Field(default=None, examples=["csv", "pdf", "docx", "web", "api", "database"])
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    force: bool = False
    visibility: Literal["public", "organization", "private"] = "public"
    background: bool = False


class IngestResponse(BaseModel):
    job_id: uuid.UUID | None
    status: str
    document_id: uuid.UUID | None = None
    chunks_created: int = 0
    components_upserted: int = 0
    replaced_previous: bool = False
    message: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    title: str
    source_type: str
    source_name: str
    source_url: str | None
    file_path: str | None
    content_hash: str
    status: str
    visibility: str
    organization_id: str | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(validation_alias="meta")


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    chunk_index: int
    content: str
    section: str | None
    page_number: int | None
    token_count: int | None
    embedding_model: str | None
    metadata: dict[str, Any] = Field(validation_alias="meta")


class DocumentDetail(DocumentOut):
    chunk_count: int
    chunks: list[ChunkOut]


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    job_type: str
    status: str
    document_id: uuid.UUID | None
    source: str | None
    source_type: str | None
    embedding_model: str | None
    embedding_dimension: int | None
    embedding_version: str | None
    chunks_created: int
    error: str | None
    stats: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None


class ReindexResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    chunks_reembedded: int = 0
