"""Retrieval quality with the REAL local models (bge-small embeddings + bge cross-encoder).

Run with:  pytest -m models tests/retrieval
Requires the models to be downloadable/cached and the test database to be reachable.
"""

import pytest

from app.core.security import Principal

pytestmark = [pytest.mark.models, pytest.mark.integration]


@pytest.fixture(scope="module")
def real_container(migrated_db):
    from app.container import AppContainer
    from app.core.config import get_settings
    from tests.integration.conftest import seed, truncate_all

    settings = get_settings().model_copy(
        update={
            "embedding_provider": "sentence_transformers",
            "reranker_provider": "cross_encoder",
            "min_retrieval_score": 0.55,
            "llm_provider": "extractive",
        }
    )
    c = AppContainer(settings)
    truncate_all(c)
    assert seed(c).status == "completed"
    yield c
    truncate_all(c)  # leave the test DB for hashing-based tests to reseed
    c.close()


def top_components(container, query, k=5):
    hits, _ = container.rag.search(query, Principal.anonymous(), top_k=k)
    return [h.metadata.get("component") for h in hits]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("555 timer", "NE555 Timer IC"),
        ("7485", "74LS85 / 7485 Comparator"),
        ("HC SR04", "HC-SR04 Ultrasonic Sensor"),
        ("real time clock module", "DS3231 RTC Module"),
        ("decade counter", "CD4017 Counter"),
        ("ultrasonic distance measurement", "HC-SR04 Ultrasonic Sensor"),
    ],
)
def test_top1(real_container, query, expected):
    assert top_components(real_container, query, 3)[0] == expected


def test_seven_segment_related(real_container):
    found = set(top_components(real_container, "seven segment display", 5))
    assert {"Single 7-segment display", "Dual 7-segment display", "4-digit 7-segment display"} <= found


def test_digital_clock_recommendation(real_container):
    result = real_container.rag.answer("What components can I use to build a digital clock?", use_llm=False)
    names = {f.name for f in result.structured_results}
    assert {"4-digit 7-segment display", "DS3231 RTC Module"} <= names
    assert not result.no_answer


def test_off_topic_is_refused(real_container):
    result = real_container.rag.answer("How do I bake sourdough bread?", use_llm=False)
    assert result.no_answer
