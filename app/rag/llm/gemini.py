"""Gemini LLM provider (google-genai SDK)."""

from __future__ import annotations

import time

from app.core.logging import get_logger
from app.rag.llm.base import LLMError, LLMProvider, LLMResponse

log = get_logger(__name__)


class GeminiLLM(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000))
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.max_retries = max_retries

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        types = self._types
        config_kwargs = dict(
            system_instruction=system_prompt,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self.max_output_tokens,
            # No tools are passed; disable automatic function calling explicitly.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if self.thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level.upper())
        elif self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        config = types.GenerateContentConfig(**config_kwargs)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model_name, contents=user_prompt, config=config
                )
                text = (response.text or "").strip()
                if not text:
                    raise LLMError("Gemini returned an empty response")
                usage = {}
                if getattr(response, "usage_metadata", None) is not None:
                    um = response.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, "prompt_token_count", None),
                        "output_tokens": getattr(um, "candidates_token_count", None),
                        "total_tokens": getattr(um, "total_token_count", None),
                    }
                finish = None
                if response.candidates:
                    finish = str(getattr(response.candidates[0], "finish_reason", None))
                return LLMResponse(
                    text=text,
                    model=self.model_name,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    usage=usage,
                    finish_reason=finish,
                )
            except Exception as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                log.warning("gemini_generate_failed", attempt=attempt + 1, error=exc.__class__.__name__, code=code)
                # 4xx (bad model name, invalid config, auth) will not succeed on retry; 429 might.
                if isinstance(code, int) and 400 <= code < 500 and code != 429:
                    break
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise LLMError(f"Gemini generation failed: {last_error}") from last_error
