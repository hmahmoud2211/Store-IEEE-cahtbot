"""FastAPI application factory."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import components, documents, health, ingest, query
from app.container import AppContainer
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(container: AppContainer | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or AppContainer(settings)
        if settings.warmup_models:
            app.state.container.warmup()
        log.info("app_started", environment=settings.environment)
        yield
        app.state.container.close()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Retrieval-Augmented Generation over an electronics component knowledge base.",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex[:12]
        log.exception("unhandled_error", error_id=error_id, path=request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "error_id": error_id})

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(components.router, prefix=prefix)

    # Test console (single static page that calls the API on the same origin).
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app


app = create_app()
