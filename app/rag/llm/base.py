from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


class LLMError(Exception):
    pass


class LLMProvider(ABC):
    """Interface for answer-generating LLMs. Implementations must be stateless per call."""

    model_name: str

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
