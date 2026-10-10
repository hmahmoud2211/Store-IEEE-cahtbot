"""Embedding provider implementations and factory.

* SentenceTransformerProvider – local model (default BAAI/bge-small-en-v1.5, 384-d)
* GeminiEmbeddingProvider     – Gemini API embeddings (configurable dimension)
* HashingEmbeddingProvider    – deterministic, dependency-free (tests / offline CI)
* CachedEmbeddingProvider     – wraps any provider with a query-embedding cache
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence

from app.core.cache import Cache, make_key
from app.core.config import Settings
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider

log = get_logger(__name__)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str,
        dimension: int,
        version: str = "1",
        batch_size: int = 64,
        device: str | None = None,
        query_instruction: str = "",
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.version = version
        self.batch_size = batch_size
        self.device = device
        self.query_instruction = query_instruction
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading_embedding_model", model=self.model_name)
                    model = SentenceTransformer(self.model_name, device=self.device)
                    actual = (
                        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
                    )()
                    if actual != self.dimension:
                        raise ValueError(
                            f"Embedding model {self.model_name} produces {actual}-d vectors but "
                            f"EMBEDDING_DIMENSION={self.dimension}. Update the config and migrate."
                        )
                    self._model = model
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [f"{self.query_instruction}{query}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vector.tolist()


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model_name: str, dimension: int, version: str = "1", batch_size: int = 64) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.dimension = dimension
        self.version = version
        self.batch_size = batch_size

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimension),
            )
            out.extend(_l2_normalize(list(e.values)) for e in result.embeddings)
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], "RETRIEVAL_QUERY")[0]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Feature-hashing embedding over words and character trigrams.

    Not semantic, but deterministic and lexical-similarity aware: good enough
    to exercise pgvector search mechanics in tests without downloading models.
    """

    def __init__(self, dimension: int = 384, version: str = "1") -> None:
        self.model_name = "hashing-ngram"
        self.dimension = dimension
        self.version = version

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"#{w}#"
            feats.extend(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for feat in self._features(text):
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (2.0 if feat.startswith("w:") else 1.0)
        return _l2_normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


class CachedEmbeddingProvider(EmbeddingProvider):
    """Caches query embeddings (documents are embedded once at ingestion)."""

    def __init__(self, inner: EmbeddingProvider, cache: Cache) -> None:
        self.inner = inner
        self.cache = cache
        self.model_name = inner.model_name
        self.dimension = inner.dimension
        self.version = inner.version

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        key = make_key("qemb", self.model_name, self.version, query)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vector = self.inner.embed_query(query)
        self.cache.set(key, vector)
        return vector


def build_embedding_provider(settings: Settings, cache: Cache | None = None) -> EmbeddingProvider:
    provider: EmbeddingProvider
    if settings.embedding_provider == "sentence_transformers":
        provider = SentenceTransformerProvider(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            version=settings.embedding_version,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            query_instruction=settings.embedding_query_instruction,
        )
    elif settings.embedding_provider == "gemini":
        if not settings.gemini_key:
            raise ValueError("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY")
        provider = GeminiEmbeddingProvider(
            api_key=settings.gemini_key,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            version=settings.embedding_version,
        )
    elif settings.embedding_provider == "hashing":
        provider = HashingEmbeddingProvider(dimension=settings.embedding_dimension, version=settings.embedding_version)
    else:  # pragma: no cover - guarded by Literal type
        raise ValueError(f"Unknown EMBEDDING_PROVIDER {settings.embedding_provider}")
    return CachedEmbeddingProvider(provider, cache) if cache is not None else provider
