from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.api.deps import get_container
from app.container import AppContainer

router = APIRouter(tags=["health"])


@router.get("/health")
def health(container: AppContainer = Depends(get_container)) -> dict:
    s = container.settings
    checks: dict[str, object] = {}
    status = "ok"
    try:
        with container.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
            checks["pgvector_version"] = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            # For vector columns atttypmod stores the declared dimension.
            column_dim = conn.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'"
                )
            ).scalar()
            checks["embedding_column_dimension"] = column_dim
            stale = conn.execute(
                text(
                    "SELECT count(*) FROM document_chunks "
                    "WHERE embedding_model IS DISTINCT FROM :m OR embedding_version IS DISTINCT FROM :v"
                ),
                {"m": container.embedder.model_name, "v": container.embedder.version},
            ).scalar()
            checks["chunks_needing_reindex"] = stale
            counts = conn.execute(
                text(
                    "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM document_chunks), "
                    "(SELECT count(*) FROM components)"
                )
            ).one()
            checks["documents"], checks["chunks"], checks["components"] = counts
            if checks["pgvector_version"] is None:
                status = "degraded"
            if column_dim != s.embedding_dimension:
                status = "degraded"
                checks["dimension_mismatch"] = f"column={column_dim} config={s.embedding_dimension}"
    except Exception as exc:
        status = "error"
        checks["database"] = f"error: {exc.__class__.__name__}"
    return {
        "status": status,
        "checks": checks,
        "config": {
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model
            if s.gemini_key or s.llm_provider != "gemini"
            else "extractive (no GEMINI_API_KEY)",
            "embedding_model": s.embedding_model,
            "embedding_dimension": s.embedding_dimension,
            "reranker": f"{s.reranker_provider}:{s.reranker_model}",
        },
    }
