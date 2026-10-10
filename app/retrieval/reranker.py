"""Reranking stage: re-score ~20 fused candidates with a stronger model, keep the best 3–8.

Providers:
  * CrossEncoderReranker – dedicated cross-encoder (default BAAI/bge-reranker-base)
  * LLMReranker          – asks the LLM to grade passages (fallback / experimentation)
  * PassThroughReranker  – keeps fusion order (tests, or when latency matters most)

All providers write a relevance score in [0, 1] to `hit.scores["rerank"]`.
"""

from __future__ import annotations

import json
import math
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.types import SearchHit

if TYPE_CHECKING:
    from app.rag.llm.base import LLMProvider

log = get_logger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance scores in [0, 1], aligned with `passages`."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int | None = None) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.score(query, [h.content for h in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.scores["rerank"] = float(score)
        ranked = sorted(hits, key=lambda h: h.scores["rerank"], reverse=True)
        for i, hit in enumerate(ranked, start=1):
            hit.ranks["rerank"] = i
        return ranked[:top_k] if top_k else ranked


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str, max_length: int = 512, device: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    log.info("loading_reranker_model", model=self.model_name)
                    self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False)
        values = [float(v) for v in raw]
        # Some checkpoints return logits; map them to probabilities.
        if any(v < 0.0 or v > 1.0 for v in values):
            values = [_sigmoid(v) for v in values]
        return values


class LLMReranker(Reranker):
    name = "llm"

    _PROMPT = (
        "Grade how useful each passage is for answering the question, from 0 (irrelevant) to 10 "
        "(directly answers it). Respond ONLY with a JSON list of numbers, one per passage, in order.\n\n"
        "Question: {question}\n\n{passages}"
    )

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        listing = "\n\n".join(f"Passage {i + 1}:\n{p[:1200]}" for i, p in enumerate(passages))
        try:
            response = self.llm.generate(
                system_prompt="You are a strict relevance grader.",
                user_prompt=self._PROMPT.format(question=query, passages=listing),
            )
            match = re.search(r"\[[\s\S]*\]", response.text)
            grades = json.loads(match.group(0)) if match else []
            if len(grades) == len(passages):
                return [max(0.0, min(1.0, float(g) / 10.0)) for g in grades]
        except Exception as exc:  # degrade gracefully to fusion order
            log.warning("llm_rerank_failed", error=str(exc))
        n = len(passages)
        return [1.0 - i / max(n, 1) for i in range(n)]


class PassThroughReranker(Reranker):
    """Keeps fusion order; scores decay with rank so thresholds still behave sensibly."""

    name = "passthrough"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        n = len(passages)
        return [1.0 - i / (n + 1) for i in range(n)]


def build_reranker(settings: Settings, llm: LLMProvider | None = None) -> Reranker:
    if settings.reranker_provider == "cross_encoder":
        return CrossEncoderReranker(settings.reranker_model, settings.reranker_max_length, settings.embedding_device)
    if settings.reranker_provider == "llm":
        if llm is None:
            raise ValueError("RERANKER_PROVIDER=llm requires an LLM provider")
        return LLMReranker(llm)
    return PassThroughReranker()
