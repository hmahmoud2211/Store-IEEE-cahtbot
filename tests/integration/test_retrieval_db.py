"""Vector, keyword, hybrid retrieval and metadata/permission filtering in PostgreSQL."""

from decimal import Decimal

import pytest

from app.core.security import Principal
from app.ingestion.pipeline import IngestionOptions
from app.retrieval.keyword_search import KeywordSearcher
from app.retrieval.metadata_filter import RetrievalFilters
from app.retrieval.vector_search import VectorSearcher

pytestmark = pytest.mark.integration
ANON = Principal.anonymous()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("74LS85", "74LS85 / 7485 Comparator"),
        ("7485", "74LS85 / 7485 Comparator"),
        ("HC SR04", "HC-SR04 Ultrasonic Sensor"),
        ("hc-sr04 price", "HC-SR04 Ultrasonic Sensor"),
        ("DS3231", "DS3231 RTC Module"),
        ("CD4511", "CD4511 BCD-to-7-Segment Decoder"),
        ("2N2222", "2N2222 / similar transistor"),
        ("74LS161", "74LS161 Counter"),
    ],
)
def test_keyword_search_exact_identifiers(session, query, expected):
    hits = KeywordSearcher().search(session, query, 5, ANON)
    assert hits, query
    assert hits[0].metadata["component"] == expected
    assert hits[0].ranks["keyword"] == 1


def test_keyword_search_stemming(session):
    hits = KeywordSearcher().search(session, "counters", 10, ANON)
    assert {"74LS161 Counter", "CD4017 Counter", "7493 Binary Counter"} <= {h.metadata["component"] for h in hits}


def test_vector_search_returns_nearest_chunk(container, session):
    from app.db.models import DocumentChunk

    target = session.query(DocumentChunk).filter(DocumentChunk.meta["component"].astext == "Buzzer").one()
    vector = container.embedder.embed_documents([target.content])[0]
    hits = VectorSearcher().search(session, vector, 3, ANON)
    assert hits[0].chunk_id == target.id
    assert hits[0].scores["vector_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_hybrid_combines_both_retrievers(container, session):
    result = container.rag.retriever.retrieve(session, "HC-SR04 ultrasonic distance sensor", ANON)
    top = result.hits[0]
    assert top.metadata["component"] == "HC-SR04 Ultrasonic Sensor"
    assert {"vector", "keyword"} <= set(top.ranks)
    assert result.vector_candidates > 0 and result.keyword_candidates > 0
    assert top.scores["rrf"] > 0


def test_metadata_filters_are_applied_in_sql(container, session):
    rf = RetrievalFilters(categories=["Counter"])
    hits = container.rag.retriever.retrieve(session, "binary counting chip", ANON, rf).hits
    assert hits and all(h.metadata["category"] == "Counter" for h in hits)

    rf = RetrievalFilters(max_price=Decimal(10), currency="EGP")
    hits = VectorSearcher().search(session, container.embedder.embed_query("component"), 50, ANON, rf)
    assert hits and all(h.metadata["price_min"] <= 10 for h in hits)


def test_permissions_enforced_during_retrieval(container):
    private = container.ingestion.ingest(
        "private_notes.md",
        IngestionOptions(
            source_type="text",
            visibility="private",
            owner_user_id="user-1",
            loader_options={"content": "# Lab notes\n\nThe zebratronic flux module needs calibration every week."},
        ),
    )
    org = container.ingestion.ingest(
        "org_notes.md",
        IngestionOptions(
            source_type="text",
            visibility="organization",
            organization_id="org-1",
            loader_options={"content": "# Org notes\n\nThe quokkatron widget is reserved for org-1 labs."},
        ),
    )
    try:
        with container.session_factory() as s:
            searcher = KeywordSearcher()
            assert searcher.search(s, "zebratronic", 5, ANON) == []
            assert searcher.search(s, "zebratronic", 5, Principal(user_id="user-2")) == []
            assert len(searcher.search(s, "zebratronic", 5, Principal(user_id="user-1"))) == 1
            assert searcher.search(s, "quokkatron", 5, Principal(organization_id="org-2")) == []
            assert len(searcher.search(s, "quokkatron", 5, Principal(organization_id="org-1"))) == 1
            assert len(searcher.search(s, "quokkatron", 5, Principal.system())) == 1
    finally:
        container.documents.delete(private.document_id)
        container.documents.delete(org.document_id)
