"""Ingestion against a real PostgreSQL + pgvector database."""

from decimal import Decimal

import pytest
from sqlalchemy import func, select, text

from app.db.models import Component, ComponentPrice, Document, DocumentChunk, IngestionJob
from app.db.repositories.components import ComponentQuery, ComponentRepository, PriceBound
from app.ingestion.pipeline import IngestionOptions
from tests.conftest import SEED_CSV
from tests.integration.conftest import seed

pytestmark = pytest.mark.integration


def test_seed_creates_components_chunks_and_job(container, session):
    assert session.scalar(select(func.count()).select_from(Component)) == 32
    assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 32
    job = session.scalars(select(IngestionJob).where(IngestionJob.status == "completed")).first()
    assert job.embedding_model == container.embedder.model_name
    assert job.embedding_dimension == 384
    assert job.embedding_version == container.embedder.version


def test_component_row_is_structured(session):
    ne555 = session.scalar(select(Component).where(Component.normalized_name == "ne555 timer ic"))
    assert ne555.category == "Timer"
    assert (ne555.price_min, ne555.price_max, ne555.currency) == (Decimal("4.00"), Decimal("10.00"), "EGP")
    assert ne555.price_text == "4–10 EGP"  # original preserved
    assert "555 Timer" in ne555.aliases
    assert "ne555" in ne555.search_keys.split()


def test_chunk_has_embedding_tsvector_and_metadata(session):
    chunk = session.scalar(
        select(DocumentChunk).where(DocumentChunk.meta["component"].astext == "HC-SR04 Ultrasonic Sensor")
    )
    assert chunk.embedding is not None and len(chunk.embedding) == 384
    assert chunk.meta["category"] == "Sensor"
    assert chunk.meta["price_min"] == 40.0
    assert chunk.meta["source"] == "hardware_components.csv"
    assert chunk.meta["country"] == "Egypt"
    lexemes = session.execute(
        text("SELECT content_tsv::text FROM document_chunks WHERE id = :id"), {"id": chunk.id}
    ).scalar()
    assert "'hcsr04'" in lexemes and "'hc-sr04'" in lexemes


def test_duplicate_document_is_skipped(container):
    result = seed(container)
    assert result.status == "duplicate"


def test_force_reingest_replaces_previous(container, session):
    result = seed(container, force=True)
    assert result.status == "completed" and result.replaced_previous
    assert session.scalar(select(func.count()).select_from(Document)) == 1
    assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 32
    assert session.scalar(select(func.count()).select_from(Component)) == 32


def test_changed_document_is_reindexed(container, tmp_path):
    changed = tmp_path / "hardware_components.csv"  # same source name, different content
    changed.write_text(
        SEED_CSV.read_text(encoding="utf-8").replace("DS3231 RTC Module,135 EGP", "DS3231 RTC Module,140 EGP"),
        encoding="utf-8",
    )
    result = container.ingestion.ingest(
        str(changed), IngestionOptions(source_type="csv", title="Hardware Components Price List")
    )
    assert result.status == "completed" and result.replaced_previous
    with container.session_factory() as s:
        ds3231 = s.scalar(select(Component).where(Component.normalized_name == "ds3231 rtc module"))
        assert ds3231.price_min == Decimal("140.00")
    seed(container, force=True)  # restore original data for the other tests


def test_second_source_records_additional_price(container, tmp_path):
    md = tmp_path / "supplier_x.md"
    md.write_text(
        "# Supplier X price list\n\n| Component | Price |\n|---|---|\n| NE555 Timer IC | 6 EGP |\n", encoding="utf-8"
    )
    result = container.ingestion.ingest(str(md), IngestionOptions(metadata={"supplier": "Supplier X"}))
    assert result.status == "completed" and result.components_upserted == 1
    with container.session_factory() as s:
        ne555 = s.scalar(select(Component).where(Component.normalized_name == "ne555 timer ic"))
        prices = s.scalars(select(ComponentPrice).where(ComponentPrice.component_id == ne555.id)).all()
        assert len(prices) == 2
        assert {p.supplier for p in prices} == {None, "Supplier X"}
    container.documents.delete(result.document_id)
    seed(container, force=True)


def test_sql_price_filter_definite_and_partial(session):
    repo = ComponentRepository(session)
    definite, partial = repo.search(ComponentQuery(categories=["Counter"], max_price=PriceBound(Decimal(25))))
    assert [c.name for c in definite] == ["74LS161 Counter"]
    assert [c.name for c in partial] == ["CD4017 Counter"]
    definite, _ = repo.search(ComponentQuery(max_price=PriceBound(Decimal(15)), sort="asc"))
    assert definite[0].name == "Resistors"
    assert all(c.price_max < 15 for c in definite)


def test_fuzzy_component_search(session):
    names = [c.name for c, _ in ComponentRepository(session).fuzzy_search("breadboard", threshold=0.3)]
    assert "400-point breadboard" in names and "830-point breadboard" in names


def test_reindex_reembeds_all_chunks(container):
    job_id = container.documents.start_reindex(container.embedder)
    assert container.documents.reindex(container.embedder, job_id) == 32
    with container.session_factory() as s:
        assert s.get(IngestionJob, job_id).status == "completed"
