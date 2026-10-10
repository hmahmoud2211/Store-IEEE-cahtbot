"""Generation guardrails with a fake LLM: citation cleanup, grounding fallback, error fallback."""

import pytest

from app.core.config import get_settings
from app.embeddings.provider import HashingEmbeddingProvider
from app.rag.context_builder import ContextBuilder
from app.rag.llm.base import LLMError, LLMProvider, LLMResponse
from app.rag.pipeline import RAGPipeline
from app.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.rag.query_processor import QueryProcessor
from app.rag.types import QueryType
from app.retrieval.reranker import PassThroughReranker
from tests.unit.test_generation_units import NE555


class FakeLLM(LLMProvider):
    model_name = "fake-llm"

    def __init__(self, text: str | None = None, fail: bool = False) -> None:
        self.text, self.fail, self.prompts = text, fail, []

    def generate(self, system_prompt, user_prompt, *, temperature=None, max_tokens=None):
        self.prompts.append((system_prompt, user_prompt))
        if self.fail:
            raise LLMError("boom")
        return LLMResponse(text=self.text or "", model=self.model_name)


def _pipeline(llm, policy="fallback"):
    settings = get_settings().model_copy(update={"grounding_policy": policy})
    return RAGPipeline(None, HashingEmbeddingProvider(64), PassThroughReranker(), llm, settings)


def _bundle():
    return ContextBuilder().build(
        query_type=QueryType.FACT_LOOKUP, question="Price of NE555?", components=[NE555], hits=[]
    )


ANALYSIS = QueryProcessor().analyze("Price of NE555?")


def test_grounded_llm_answer_is_kept_and_prompt_contains_context():
    llm = FakeLLM("The NE555 Timer IC costs approximately 4–10 EGP [1]. Prices may change.")
    text, meta = _pipeline(llm)._generate(_bundle(), ANALYSIS, True, {})
    assert text.startswith("The NE555 Timer IC costs approximately 4–10 EGP [1].")
    assert meta["llm_model"] == "fake-llm" and not meta["llm_fallback"]
    system, user = llm.prompts[0]
    assert system == SYSTEM_PROMPT
    assert "[SOURCE 1]" in user and "Approximate price: 4–10 EGP" in user


def test_invented_price_triggers_extractive_fallback():
    text, meta = _pipeline(FakeLLM("The NE555 costs 50 EGP [1]."))._generate(_bundle(), ANALYSIS, True, {})
    assert "50 EGP" not in text
    assert "4–10 EGP [1]" in text
    assert meta["llm_fallback"] and "50" in meta["grounding_warning"]


def test_warn_policy_keeps_answer_but_flags_it():
    text, meta = _pipeline(FakeLLM("The NE555 costs 50 EGP [1]."), policy="warn")._generate(
        _bundle(), ANALYSIS, True, {}
    )
    assert "50 EGP" in text
    assert meta["grounding_warning"]


def test_invalid_citations_removed():
    text, meta = _pipeline(FakeLLM("It costs 4–10 EGP [1][9]."))._generate(_bundle(), ANALYSIS, True, {})
    assert "[9]" not in text
    assert meta["invalid_citations_removed"] == [9]


def test_llm_error_falls_back_to_extractive():
    text, meta = _pipeline(FakeLLM(fail=True))._generate(_bundle(), ANALYSIS, True, {})
    assert "4–10 EGP [1]" in text
    assert meta["llm_fallback"] and meta["llm_error"] == "LLMError"


def test_prompt_contains_rules():
    prompt = build_user_prompt(_bundle())
    assert "Query type: FACT_LOOKUP" in prompt
    for rule in (
        "Never invent component prices",
        "I could not find this information in the current knowledge base.",
        "sources disagree",
    ):
        assert rule in SYSTEM_PROMPT


def test_project_guide_prompt_names_project_and_sections():
    bundle = ContextBuilder().build(
        query_type=QueryType.PROJECT_GUIDE, question="Give me the steps", components=[NE555], hits=[]
    )
    prompt = build_user_prompt(bundle, project="digital clock")
    assert "Project: digital clock" in prompt
    for section in ("Bill of materials", "Step-by-step assembly", "Troubleshooting", "store-orders@example.com"):
        assert section in prompt


def test_guide_wording_is_not_mistaken_for_prices():
    guide = "Connect 8 LEDs through the level shifter. The NE555 Timer IC costs approximately 4–10 EGP [1]."
    llm = FakeLLM(guide)
    bundle = ContextBuilder().build(
        query_type=QueryType.PROJECT_GUIDE, question="Give me the steps", components=[NE555], hits=[]
    )
    text, meta = _pipeline(llm)._generate(bundle, ANALYSIS, True, {})
    assert text == guide and not meta["llm_fallback"]


def test_code_and_math_are_not_mistaken_for_prices():
    guide = (
        "Count from $0000$ to $1001$ using $R_1$. Call `delay(1000)`.\n```\nint pin = 13$\n```\n"
        "The NE555 Timer IC costs approximately 4–10 EGP [1]."
    )
    bundle = ContextBuilder().build(
        query_type=QueryType.PROJECT_GUIDE, question="Give me the steps", components=[NE555], hits=[]
    )
    text, meta = _pipeline(FakeLLM(guide))._generate(bundle, ANALYSIS, True, {})
    assert text == guide and not meta["llm_fallback"]


def test_gemini_does_not_retry_client_errors(monkeypatch):
    from app.rag.llm.gemini import GeminiLLM

    class ClientError(Exception):
        code = 404

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            raise ClientError("model not found")

    llm = GeminiLLM(api_key="test", max_retries=2)
    llm._client = type("C", (), {"models": Models()})()
    with pytest.raises(LLMError):
        llm.generate("s", "u")
    assert len(calls) == 1
    assert calls[0]["config"].automatic_function_calling.disable is True
