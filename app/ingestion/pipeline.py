"""Offline ingestion pipeline.

    source ─► Loader ─► clean ─► hash/dedup ─► component extraction ─► structure-aware chunking
           ─► metadata ─► embeddings ─► PostgreSQL (documents, document_chunks, components, prices)

The pipeline depends only on the loader *interface* (via the registry), the
embedding *interface* and repositories, never on concrete implementations.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.core.cache import Cache
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk
from app.db.repositories.components import ComponentRepository
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.jobs import JobRepository
from app.db.session import session_scope
from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import ChunkDraft, StructureAwareChunker
from app.ingestion.cleaning import clean_document
from app.ingestion.component_extractor import ComponentExtractor, ComponentRecord
from app.ingestion.loaders.base import Element, ElementType, LoadedDocument
from app.ingestion.loaders.registry import LoaderRegistry
from app.ingestion.metadata import MetadataEnricher, chunk_search_keys

log = get_logger(__name__)


@dataclass
class IngestionOptions:
    source_type: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    force: bool = False
    owner_user_id: str | None = None
    organization_id: str | None = None
    visibility: str = "public"
    loader_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    job_id: uuid.UUID | None
    document_id: uuid.UUID | None
    status: str  # completed | duplicate | failed
    chunks_created: int = 0
    components_upserted: int = 0
    replaced_previous: bool = False
    message: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PreparedChunk:
    draft: ChunkDraft
    component: ComponentRecord | None


class IngestionPipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: EmbeddingProvider,
        registry: LoaderRegistry,
        settings: Settings,
        cache: Cache | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.embedder = embedder
        self.registry = registry
        self.settings = settings
        self.cache = cache
        self.chunker = StructureAwareChunker(settings.chunk_size, settings.chunk_overlap)
        self.extractor = ComponentExtractor(default_currency=settings.default_currency)
        self.enricher = MetadataEnricher(default_country=settings.default_country)

    # ------------------------------------------------------------ public API
    def create_job(self, source: Any, source_type: str | None) -> uuid.UUID:
        with session_scope(self.session_factory) as session:
            job = JobRepository(session).create(
                source=self._describe_source(source), source_type=source_type, status="pending", **self.embedder.info
            )
            return job.id

    def ingest(
        self, source: Any, options: IngestionOptions | None = None, job_id: uuid.UUID | None = None
    ) -> IngestionResult:
        options = options or IngestionOptions()
        started = time.perf_counter()
        timings: dict[str, float] = {}

        job_id = job_id or self.create_job(source, options.source_type)
        self._mark_job(job_id, status="running")
        try:
            t = time.perf_counter()
            loader = self.registry.resolve(source, options.source_type)
            document = loader.load(source, title=options.title, **options.loader_options)
            timings["load_ms"] = _ms(t)

            document = self._apply_overrides(document, options)
            t = time.perf_counter()
            document = clean_document(document)
            timings["clean_ms"] = _ms(t)
            content_hash = document.content_hash()

            result = self._persist(document, content_hash, options, job_id, timings)
            result.stats["total_ms"] = _ms(started)
            log.info(
                "ingestion_finished",
                job_id=str(job_id),
                status=result.status,
                document_id=str(result.document_id) if result.document_id else None,
                source_type=document.source_type,
                chunks=result.chunks_created,
                components=result.components_upserted,
                **timings,
            )
            if self.cache is not None and result.status == "completed":
                self.cache.clear()  # retrieval results / alias index are now stale
            return result
        except Exception as exc:
            log.exception("ingestion_failed", job_id=str(job_id), error=str(exc))
            self._mark_job(job_id, status="failed", error=f"{exc.__class__.__name__}: {exc}"[:2000])
            return IngestionResult(job_id=job_id, document_id=None, status="failed", message=str(exc))

    # -------------------------------------------------------------- internals
    def _persist(
        self,
        document: LoadedDocument,
        content_hash: str,
        options: IngestionOptions,
        job_id: uuid.UUID,
        timings: dict[str, float],
    ) -> IngestionResult:
        with session_scope(self.session_factory) as session:
            docs = DocumentRepository(session)
            jobs = JobRepository(session)
            job = jobs.get(job_id)

            # 1) Exact duplicate -> skip (unless forced).
            existing = docs.get_by_hash(content_hash)
            if existing and not options.force:
                jobs.finish(job, status="duplicate", document_id=existing.id, stats={"reason": "content_hash_match"})
                return IngestionResult(
                    job_id=job_id,
                    document_id=existing.id,
                    status="duplicate",
                    message="Document already ingested (identical content hash).",
                )

            # 2) Same source with changed content -> re-index in place.
            previous = existing or docs.get_by_source(document.source_type, document.source_name)
            replaced = previous is not None
            if previous is not None:
                docs.delete(previous)  # cascades chunks + per-source prices
                session.flush()

            doc_row = docs.add(
                Document(
                    id=uuid.uuid4(),
                    title=document.title,
                    source_type=document.source_type,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    file_path=document.file_path,
                    content_hash=content_hash,
                    owner_user_id=options.owner_user_id,
                    organization_id=options.organization_id,
                    visibility=options.visibility,
                    meta={**document.metadata, **self.embedder.info},
                )
            )

            # 3) Structured component extraction (tables / records with prices).
            t = time.perf_counter()
            prepared = self._prepare_chunks(document)
            timings["chunk_ms"] = _ms(t)

            component_repo = ComponentRepository(session)
            component_ids: dict[str, uuid.UUID] = {}
            for item in prepared:
                if item.component is None:
                    continue
                rec = item.component
                comp = component_repo.upsert(self._component_values(rec, document), source_document_id=doc_row.id)
                component_ids[rec.normalized_name] = comp.id
            if replaced:
                component_repo.delete_orphans()  # components dropped from the new version

            # 4) Embeddings (batched).
            t = time.perf_counter()
            texts = [p.draft.content for p in prepared]
            vectors: list[list[float]] = []
            batch = max(1, self.settings.embedding_batch_size)
            for i in range(0, len(texts), batch):
                vectors.extend(self.embedder.embed_documents(texts[i : i + batch]))
            timings["embed_ms"] = _ms(t)

            # 5) Chunks with metadata, search keys and ACL fields.
            acl = {
                "owner_user_id": options.owner_user_id,
                "organization_id": options.organization_id,
                "visibility": options.visibility,
            }
            rows: list[DocumentChunk] = []
            for index, (item, vector) in enumerate(zip(prepared, vectors, strict=True)):
                comp_id = component_ids.get(item.component.normalized_name) if item.component else None
                meta = self.enricher.build(
                    document=document,
                    document_id=str(doc_row.id),
                    draft=item.draft,
                    chunk_index=index,
                    component=item.component,
                    component_id=str(comp_id) if comp_id else None,
                    acl=acl,
                )
                rows.append(
                    DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=doc_row.id,
                        chunk_index=index,
                        content=item.draft.content,
                        search_keys=chunk_search_keys(
                            item.draft.content, item.component.search_keys if item.component else None
                        ),
                        embedding=vector,
                        embedding_model=self.embedder.model_name,
                        embedding_version=self.embedder.version,
                        page_number=item.draft.page_number,
                        section=item.draft.section,
                        token_count=item.draft.token_count,
                        meta=meta,
                        **acl,
                    )
                )
            t = time.perf_counter()
            session.add_all(rows)
            session.flush()
            timings["store_ms"] = _ms(t)

            stats = {"elements": len(document.elements), "components": len(component_ids), **timings}
            jobs.finish(
                job,
                status="completed",
                document_id=doc_row.id,
                chunks_created=len(rows),
                stats=stats,
            )
            return IngestionResult(
                job_id=job_id,
                document_id=doc_row.id,
                status="completed",
                chunks_created=len(rows),
                components_upserted=len(component_ids),
                replaced_previous=replaced,
                stats=stats,
            )

    def _prepare_chunks(self, document: LoadedDocument) -> list[_PreparedChunk]:
        """Component rows become one natural-language record each; everything else is chunked by structure."""
        supplier = document.metadata.get("supplier")
        extracted = self.extractor.extract(document.elements, supplier=supplier)
        country = document.metadata.get("country") or self.settings.default_country

        prepared: list[_PreparedChunk] = []
        pending: list[Element] = []

        def flush_pending() -> None:
            if pending:
                sub = replace(document, elements=list(pending))
                prepared.extend(_PreparedChunk(d, None) for d in self.chunker.chunk(sub))
                pending.clear()

        current_section: str | None = None
        for idx, element in enumerate(document.elements):
            if element.type is ElementType.HEADING:
                current_section = element.text
            records = extracted.get(idx)
            if not records:
                pending.append(element)
                continue
            # Keep the original table (if any) as context, then one record per component.
            flush_pending()
            if element.type is ElementType.TABLE:
                sub = replace(document, elements=[element])
                prepared.extend(_PreparedChunk(d, None) for d in self.chunker.chunk(sub))
            for rec in records:
                text = rec.chunk_text(country=country if rec.price and rec.price.currency == "EGP" else None)
                draft = ChunkDraft(
                    content=text,
                    section=element.section or current_section or rec.category.value,
                    page_number=element.page_number,
                    token_count=self.chunker.count(text),
                    element_types=["record"],
                    metadata=dict(element.metadata),
                )
                prepared.append(_PreparedChunk(draft, rec))
        flush_pending()
        return prepared

    def _component_values(self, rec: ComponentRecord, document: LoadedDocument) -> dict[str, Any]:
        price = rec.price
        return {
            "name": rec.name,
            "normalized_name": rec.normalized_name,
            "category": rec.category.value,
            "aliases": rec.aliases,
            "search_keys": rec.search_keys,
            "price_min": price.min if price else None,
            "price_max": price.max if price else None,
            "currency": price.currency if price else None,
            "price_unit": price.unit if price else None,
            "price_text": price.original if price else None,
            "description": rec.description,
            "typical_uses": rec.typical_uses,
            "meta": {
                "original_row": rec.source_row,
                "source": document.source_name,
                "supplier": rec.supplier,
                "country": document.metadata.get("country") or self.settings.default_country,
            },
        }

    @staticmethod
    def _apply_overrides(document: LoadedDocument, options: IngestionOptions) -> LoadedDocument:
        meta = {**document.metadata, **options.metadata}
        title = options.title or meta.pop("title", None) or document.title
        return replace(document, title=title, metadata=meta)

    def _mark_job(self, job_id: uuid.UUID, *, status: str, error: str | None = None) -> None:
        with session_scope(self.session_factory) as session:
            jobs = JobRepository(session)
            job = jobs.get(job_id)
            if job is None:
                return
            if status in ("failed", "completed", "duplicate"):
                jobs.finish(job, status=status, error=error)
            else:
                job.status = status

    @staticmethod
    def _describe_source(source: Any) -> str:
        if isinstance(source, dict):  # never persist credentials/headers
            return str(source.get("name") or source.get("title") or source.get("url", "")).split("@")[-1][:500]
        return str(source)[:500]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
