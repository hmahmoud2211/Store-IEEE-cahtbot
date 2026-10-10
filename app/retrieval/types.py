from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    metadata: dict[str, Any]
    section: str | None = None
    page_number: int | None = None
    document_title: str | None = None
    source_name: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    visibility: str | None = None
    organization_id: str | None = None
    owner_user_id: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def component_id(self) -> str | None:
        return self.metadata.get("component_id")

    @property
    def best_score(self) -> float:
        for key in ("rerank", "rrf"):
            if key in self.scores:
                return self.scores[key]
        return 0.0

    def debug_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "component": self.metadata.get("component"),
            "section": self.section,
            "page": self.page_number,
            "scores": {k: round(v, 5) for k, v in self.scores.items()},
            "ranks": self.ranks,
        }


def hit_from_row(row: Any) -> SearchHit:
    m = row._mapping
    return SearchHit(
        chunk_id=m["id"],
        document_id=m["document_id"],
        content=m["content"],
        metadata=dict(m["metadata"] or {}),
        section=m["section"],
        page_number=m["page_number"],
        document_title=m["title"],
        source_name=m["source_name"],
        source_type=m["source_type"],
        source_url=m["source_url"],
        visibility=m["visibility"],
        organization_id=m["organization_id"],
        owner_user_id=m["owner_user_id"],
    )


CHUNK_SELECT_COLUMNS = """
    c.id, c.document_id, c.content, c.metadata, c.section, c.page_number,
    c.visibility, c.organization_id, c.owner_user_id,
    d.title, d.source_name, d.source_type, d.source_url
"""
