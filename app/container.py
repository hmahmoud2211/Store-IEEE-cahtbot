"""Composition root: builds providers and pipelines from settings.

Swapping an embedding model, reranker or LLM is a configuration change here,
not a code change anywhere else.
"""

from __future__ import annotations

from functools import cached_property

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.cache import Cache, build_cache
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.session import build_engine, build_session_factory
from app.embeddings.base import EmbeddingProvider
from app.embeddings.provider import build_embedding_provider
from app.ingestion.loaders.registry import LoaderRegistry, default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.rag.llm.base import LLMProvider
from app.rag.llm.extractive import ExtractiveLLM
from app.rag.pipeline import RAGPipeline
from app.retrieval.reranker import Reranker, build_reranker
from app.services.document_service import DocumentService

log = get_logger(__name__)


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "gemini":
        if not settings.gemini_key:
            log.warning("gemini_api_key_missing_using_extractive_answers")
            return ExtractiveLLM()
        from app.rag.llm.gemini import GeminiLLM

        return GeminiLLM(
            api_key=settings.gemini_key,
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            thinking_budget=settings.llm_thinking_budget,
            thinking_level=settings.llm_thinking_level,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return ExtractiveLLM()


class AppContainer:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        llm: LLMProvider | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder_override = embedder
        self._reranker_override = reranker
        self._llm_override = llm
        self._cache_override = cache

    @cached_property
    def engine(self) -> Engine:
        return build_engine(self.settings)

    @cached_property
    def session_factory(self) -> sessionmaker[Session]:
        return build_session_factory(self.engine)

    @cached_property
    def cache(self) -> Cache:
        if self._cache_override is not None:
            return self._cache_override
        s = self.settings
        return build_cache(s.redis_url, s.cache_max_items, s.cache_ttl_seconds)

    @cached_property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder_override or build_embedding_provider(self.settings, self.cache)

    @cached_property
    def llm(self) -> LLMProvider:
        return self._llm_override or build_llm(self.settings)

    @cached_property
    def reranker(self) -> Reranker:
        return self._reranker_override or build_reranker(self.settings, self.llm)

    @cached_property
    def registry(self) -> LoaderRegistry:
        return default_registry()

    @cached_property
    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(self.session_factory, self.embedder, self.registry, self.settings, self.cache)

    @cached_property
    def rag(self) -> RAGPipeline:
        return RAGPipeline(self.session_factory, self.embedder, self.reranker, self.llm, self.settings, self.cache)

    @cached_property
    def documents(self) -> DocumentService:
        return DocumentService(self.session_factory, self.cache)

    def warmup(self) -> None:
        """Load local models eagerly so the first request is not slow."""
        self.embedder.embed_query("warmup")
        self.reranker.score("warmup", ["warmup"])

    def close(self) -> None:
        if "engine" in self.__dict__:
            self.engine.dispose()
