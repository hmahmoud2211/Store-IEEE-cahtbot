from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import auth, get_container, get_session
from app.container import AppContainer
from app.core.security import Principal, get_principal
from app.db.repositories.jobs import JobRepository
from app.ingestion.pipeline import IngestionOptions, IngestionResult
from app.schemas.document import IngestRequest, IngestResponse, JobOut, ReindexResponse

router = APIRouter(tags=["ingestion"], dependencies=auth)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _to_response(result: IngestionResult) -> IngestResponse:
    return IngestResponse(
        job_id=result.job_id,
        status=result.status,
        document_id=result.document_id,
        chunks_created=result.chunks_created,
        components_upserted=result.components_upserted,
        replaced_previous=result.replaced_previous,
        message=result.message,
        stats=result.stats,
    )


def _options(
    req_type: str | None, title: str | None, metadata: dict, force: bool, visibility: str, principal: Principal
) -> IngestionOptions:
    return IngestionOptions(
        source_type=req_type,
        title=title,
        metadata=metadata,
        force=force,
        owner_user_id=principal.user_id,
        organization_id=principal.organization_id,
        visibility=visibility,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    background: BackgroundTasks,
    container: AppContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> IngestResponse:
    """Ingest a server-side file path, a URL, or a database/API source spec."""
    source = request.source
    if isinstance(source, str) and not source.lower().startswith(("http://", "https://")):
        path = Path(source).expanduser().resolve()
        allowed_roots = [container.settings.data_dir.resolve()]
        if not any(path.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Local files must live under the data directory; use /ingest/upload otherwise",
            )
        source = str(path)

    options = _options(
        request.source_type, request.title, request.metadata, request.force, request.visibility, principal
    )
    pipeline = container.ingestion
    if request.background:
        job_id = pipeline.create_job(source, request.source_type)
        background.add_task(pipeline.ingest, source, options, job_id)
        return IngestResponse(job_id=job_id, status="pending")
    result = pipeline.ingest(source, options)
    if result.status == "failed":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result.message or "Ingestion failed")
    return _to_response(result)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str | None = Form(default=None),
    title: str | None = Form(default=None),
    force: bool = Form(default=False),
    visibility: str = Form(default="public"),
    run_in_background: bool = Form(default=False),
    container: AppContainer = Depends(get_container),
    principal: Principal = Depends(get_principal),
) -> IngestResponse:
    """Upload a PDF/DOCX/CSV/Excel/TXT file and ingest it."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "upload").name)
    upload_dir = container.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    target.write_bytes(data)

    options = _options(
        source_type, title or Path(safe_name).stem, {"file_name": safe_name}, force, visibility, principal
    )
    options.loader_options = {}
    pipeline = container.ingestion
    if run_in_background:
        job_id = pipeline.create_job(str(target), source_type)
        background.add_task(pipeline.ingest, str(target), options, job_id)
        return IngestResponse(job_id=job_id, status="pending")
    result = pipeline.ingest(str(target), options)
    if result.status == "failed":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result.message or "Ingestion failed")
    return _to_response(result)


@router.post("/reindex", response_model=ReindexResponse, status_code=status.HTTP_202_ACCEPTED)
def reindex(background: BackgroundTasks, container: AppContainer = Depends(get_container)) -> ReindexResponse:
    """Re-embed all chunks with the currently configured embedding model (runs in background)."""
    job_id = container.documents.start_reindex(container.embedder)
    background.add_task(container.documents.reindex, container.embedder, job_id)
    return ReindexResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, session=Depends(get_session)) -> JobOut:
    job = JobRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return JobOut.model_validate(job)
