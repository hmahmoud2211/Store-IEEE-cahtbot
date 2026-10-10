from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.conftest import SEED_CSV


def truncate_all(container) -> None:
    with container.engine.begin() as conn:
        conn.execute(text("TRUNCATE documents, components, component_prices, document_chunks, ingestion_jobs CASCADE"))


def seed(container, force: bool = False):
    from app.ingestion.pipeline import IngestionOptions

    return container.ingestion.ingest(
        str(SEED_CSV),
        IngestionOptions(
            source_type="csv", title="Hardware Components Price List", metadata={"country": "Egypt"}, force=force
        ),
    )


@pytest.fixture(scope="module")
def container(migrated_db):
    from app.container import AppContainer
    from app.core.config import get_settings

    c = AppContainer(get_settings())
    truncate_all(c)
    result = seed(c)
    assert result.status == "completed", result.message
    yield c
    c.close()


@pytest.fixture(scope="module")
def client(container):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(container)) as test_client:
        yield test_client


@pytest.fixture
def session(container):
    s = container.session_factory()
    try:
        yield s
    finally:
        s.rollback()
        s.close()
