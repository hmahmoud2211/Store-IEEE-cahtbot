"""Chunk metadata construction.

Every chunk carries enough metadata to (a) cite it precisely, (b) filter on it
inside retrieval SQL and (c) enforce permissions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.ingestion.chunking import ChunkDraft
from app.ingestion.component_extractor import ComponentRecord
from app.ingestion.component_normalizer import extract_identifiers, identifier_variants
from app.ingestion.loaders.base import LoadedDocument

# Keys copied from document-level metadata onto every chunk when present.
_INHERITED_KEYS = (
    "author",
    "date",
    "supplier",
    "country",
    "tags",
    "language",
    "source_url",
    "file_name",
)


def chunk_search_keys(text: str, extra: str | None = None) -> str:
    keys: set[str] = set()
    for ident in extract_identifiers(text):
        keys |= identifier_variants(ident)
    if extra:
        keys |= set(extra.split())
    return " ".join(sorted(keys))


class MetadataEnricher:
    def __init__(self, default_country: str | None = None) -> None:
        self.default_country = default_country

    def build(
        self,
        *,
        document: LoadedDocument,
        document_id: str,
        draft: ChunkDraft,
        chunk_index: int,
        component: ComponentRecord | None = None,
        component_id: str | None = None,
        acl: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        doc_meta = document.metadata or {}
        meta: dict[str, Any] = {
            "source": document.source_name,
            "source_type": document.source_type,
            "document_id": document_id,
            "document_title": document.title,
            "chunk_index": chunk_index,
            "section": draft.section,
            "page": draft.page_number,
            "element_types": draft.element_types,
            "ingested_at": datetime.now(UTC).isoformat(),
        }
        for key in _INHERITED_KEYS:
            if doc_meta.get(key) not in (None, "", []):
                meta[key] = doc_meta[key]
        if document.source_url:
            meta["source_url"] = document.source_url
        if draft.metadata.get("row_index") is not None:
            meta["row_index"] = draft.metadata["row_index"]

        if component is not None:
            meta.update(component.as_metadata())
            meta["component_id"] = component_id
            if component.description or component.typical_uses:
                # Make provenance explicit: prices come from the price column,
                # descriptions/uses are curated annotations.
                meta["annotations"] = "curated"
            if component.price and component.price.currency == "EGP":
                meta.setdefault("country", self.default_country or "Egypt")

        if acl:
            meta.update({k: v for k, v in acl.items() if v is not None})
        return {k: v for k, v in meta.items() if v is not None}
