"""Application configuration loaded from environment variables / `.env`.

Every model provider and every retrieval knob is configurable here so that
swapping an embedding model, reranker or LLM never requires code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Hardware Component RAG"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    api_keys: str = Field(default="", description="Comma-separated API keys. Empty = auth disabled.")
    warmup_models: bool = False

    # ------------------------------------------------------------- database
    database_url: str = "postgresql+psycopg://localhost:5432/rag_hardware"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    # ------------------------------------------------------------------ LLM
    llm_provider: Literal["gemini", "extractive"] = "gemini"
    llm_model: str = "gemini-3.8-flash"
    gemini_api_key: SecretStr | None = None
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048
    llm_thinking_budget: int | None = None  # older models (token budget)
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None  # Gemini 3+ models
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # warn     -> keep LLM answer, flag ungrounded numbers in metadata
    # fallback -> replace answer with the deterministic extractive answer
    grounding_policy: Literal["warn", "fallback"] = "fallback"

    # ----------------------------------------------------------- embeddings
    embedding_provider: Literal["sentence_transformers", "gemini", "hashing"] = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_version: str = "1"
    embedding_batch_size: int = 64
    embedding_device: str | None = None
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "

    # ------------------------------------------------------------- reranker
    reranker_provider: Literal["cross_encoder", "llm", "passthrough"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_max_length: int = 512

    # ------------------------------------------------------------ retrieval
    vector_top_k: int = Field(default=20, ge=1, le=200)
    keyword_top_k: int = Field(default=20, ge=1, le=200)
    hybrid_top_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    final_context_k: int = Field(default=5, ge=1, le=8)
    recommendation_context_k: int = Field(default=8, ge=1, le=8)
    # Project questions also see the store catalog (up to this many parts) so
    # "In store" / "Not in store" is decided against the real inventory. 0 disables.
    project_inventory_limit: int = Field(default=60, ge=0, le=500)
    # A candidate survives if it matched lexically OR its cosine similarity >= this.
    min_retrieval_score: float = 0.55
    # Minimum reranker relevance probability (0..1) for a chunk to enter the context.
    min_rerank_score: float = 0.05
    hnsw_ef_search: int = 64
    hnsw_iterative_scan: Literal["off", "relaxed_order", "strict_order"] = "relaxed_order"
    price_approx_tolerance: float = 0.10

    # ------------------------------------------------------------- chunking
    chunk_size: int = Field(default=500, ge=50)
    chunk_overlap: int = Field(default=75, ge=0)

    # ---------------------------------------------------------------- cache
    cache_ttl_seconds: int = 600
    cache_max_items: int = 2048
    redis_url: str | None = None

    # -------------------------------------------------------------- domain
    default_currency: str = "EGP"
    default_country: str = "Egypt"
    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 500)
        if v >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def gemini_key(self) -> str | None:
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
