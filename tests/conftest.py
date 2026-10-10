"""Shared test configuration.

Unit tests need no database and no model downloads: they use the hashing
embedder, the pass-through reranker and the extractive (template) answerer.
Integration tests use TEST_DATABASE_URL (default: local rag_hardware_test DB)
and are skipped automatically when PostgreSQL is unreachable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("TEST_DATABASE_URL", "postgresql+psycopg://localhost:5432/rag_hardware_test")
os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")
os.environ.setdefault("RERANKER_PROVIDER", "passthrough")
os.environ.setdefault("LLM_PROVIDER", "extractive")
os.environ.setdefault("MIN_RETRIEVAL_SCORE", "0.1")
os.environ.setdefault("CACHE_TTL_SECONDS", "0")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ["ENVIRONMENT"] = "test"

SEED_CSV = ROOT / "data" / "hardware_components.csv"


@pytest.fixture(scope="session")
def seed_records():
    """ComponentRecords parsed from the seed CSV (no database)."""
    from app.ingestion.cleaning import clean_document
    from app.ingestion.component_extractor import ComponentExtractor
    from app.ingestion.loaders.csv import CSVLoader

    doc = clean_document(CSVLoader().load(SEED_CSV))
    extracted = ComponentExtractor().extract(doc.elements)
    return [rec for recs in extracted.values() for rec in recs]


@pytest.fixture(scope="session")
def alias_index(seed_records):
    """In-memory alias index built from the seed records, with stable fake ids."""
    from app.services.component_service import AliasIndex

    rows = []
    ids = {}
    for rec in seed_records:
        cid = uuid.uuid5(uuid.NAMESPACE_DNS, rec.normalized_name)
        ids[rec.name] = str(cid)
        rows.append(
            (cid, rec.name, rec.normalized_name, rec.aliases, rec.search_keys, rec.price.min if rec.price else None)
        )
    index = AliasIndex.build(rows)
    index.test_ids = ids  # name -> id, for assertions
    return index


def _db_available(url: str) -> bool:
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def migrated_db():
    url = os.environ["TEST_DATABASE_URL"]
    if not _db_available(url):
        pytest.skip(f"PostgreSQL not reachable at TEST_DATABASE_URL={url}")
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(cfg, "head")
    return url
