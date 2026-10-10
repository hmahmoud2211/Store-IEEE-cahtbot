from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Interface every embedding backend implements.

    `model_name`, `dimension` and `version` are persisted with every chunk and
    ingestion job so stale embeddings can be detected and re-indexed.
    """

    model_name: str
    dimension: int
    version: str = "1"

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...

    @property
    def info(self) -> dict[str, object]:
        return {
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
            "embedding_version": self.version,
        }
