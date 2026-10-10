"""End-to-end: FastAPI → RAG pipeline → PostgreSQL (hashing embedder, extractive answers)."""

import pytest

from app.rag.types import NO_ANSWER_MESSAGE
from app.retrieval.reranker import Reranker

pytestmark = pytest.mark.integration


def ask(client, question, **extra):
    response = client.post("/api/v1/query", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["embedding_column_dimension"] == 384
    assert body["checks"]["components"] == 32


def test_fact_lookup_with_citation(client):
    body = ask(client, "What is the price of a 74LS85 comparator?")
    assert body["query_type"] == "FACT_LOOKUP"
    assert "25 EGP" in body["answer"]
    assert body["sources"][0]["component"] == "74LS85 / 7485 Comparator"
    assert body["sources"][0]["document"] == "hardware_components.csv"
    assert "Sources:" in body["answer"]


def test_price_filter_uses_sql(client):
    body = ask(client, "Which counter ICs cost less than 25 EGP?")
    assert body["query_type"] == "PRICE_FILTER"
    names = [c["name"] for c in body["structured_results"]]
    assert names == ["74LS161 Counter", "CD4017 Counter"]
    assert [c["partial_match"] for c in body["structured_results"]] == [False, True]
    assert body["retrieval_metadata"].get("final_chunks", 0) == 0  # no embeddings needed


def test_cheapest_display(client):
    body = ask(client, "What is the cheapest 7-segment display?")
    assert body["query_type"] == "PRICE_FILTER"
    assert body["structured_results"][0]["name"] == "Single 7-segment display"


def test_comparison(client):
    body = ask(client, "Which sensor is cheaper, HC-SR04 or IR obstacle sensor?")
    assert body["query_type"] == "COMPARISON"
    assert any(
        "HC-SR04 Ultrasonic Sensor (40 EGP) is cheaper than IR obstacle sensor (45 EGP)" in s
        for s in body["comparison"]
    )


def test_calculation(client):
    body = ask(client, "How much would 2 x NE555, 1 x 400-point breadboard and 5 x LED cost?")
    assert body["query_type"] == "CALCULATION"
    calc = body["calculation"]
    assert calc["total"] == "38–60 EGP"
    assert [i["line_total"] for i in calc["items"]] == ["8–20 EGP", "25 EGP", "5–15 EGP"]
    assert "Estimated total = 38–60 EGP" in body["answer"]


def test_calculation_from_history(client):
    history = [
        {"role": "user", "content": "Compare the DS1307 RTC IC and the HC-SR04."},
        {"role": "assistant", "content": "The DS1307 RTC IC costs 20 EGP and the HC-SR04 costs 40 EGP."},
    ]
    body = ask(client, "How much would these components approximately cost?", history=history)
    assert body["query_type"] == "CALCULATION"
    assert body["calculation"]["total_min"] is not None


def test_source_request_from_history(client):
    history = [{"role": "user", "content": "What is the price of the DS3231?"}]
    body = ask(client, "Show the source of the price you used.", history=history)
    assert body["sources"][0]["component"] == "DS3231 RTC Module"
    assert body["sources"][0]["document_title"] == "Hardware Components Price List"


def test_unknown_component_is_refused(client):
    body = ask(client, "What is the price of an ESP32?")
    assert "could not find ESP32" in body["answer"]
    assert body["sources"] == []
    assert body["retrieval_metadata"]["no_answer"] is True


def test_calculation_without_items_asks_for_them(client):
    body = ask(client, "How much would these components approximately cost?")
    assert body["query_type"] == "CALCULATION"
    assert "Please tell me which components" in body["answer"]


def test_alias_queries(client):
    assert ask(client, "555 timer price")["sources"][0]["component"] == "NE555 Timer IC"
    assert ask(client, "7485")["sources"][0]["component"] == "74LS85 / 7485 Comparator"
    assert ask(client, "HC SR04")["sources"][0]["component"] == "HC-SR04 Ultrasonic Sensor"


def test_search_endpoint(client):
    body = client.post("/api/v1/search", json={"query": "DS3231", "top_k": 3}).json()
    assert body["hits"][0]["metadata"]["component"] == "DS3231 RTC Module"
    assert "rrf" in body["hits"][0]["scores"]


def test_components_endpoints(client):
    body = client.get("/api/v1/components", params={"category": "counter", "max_price": 20}).json()
    assert [c["name"] for c in body["items"]] == ["74LS161 Counter"]
    cid = body["items"][0]["id"]
    one = client.get(f"/api/v1/components/{cid}").json()
    assert one["price_min"] == "18.00" and one["sources"][0]["source_name"] == "hardware_components.csv"
    assert client.get("/api/v1/components/00000000-0000-0000-0000-000000000000").status_code == 404


def test_documents_endpoints_and_delete(client, tmp_path, container):
    docs = client.get("/api/v1/documents").json()
    assert docs["total"] >= 1
    seed_doc = docs["items"][0]
    detail = client.get(f"/api/v1/documents/{seed_doc['id']}", params={"chunk_limit": 2}).json()
    assert detail["chunk_count"] == 32 and len(detail["chunks"]) == 2

    upload = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("notes.md", b"# Notes\n\nThe 7805 needs a heatsink above 1A.", "text/markdown")},
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["document_id"]
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404


def test_ingest_rejects_paths_outside_data_dir(client):
    response = client.post("/api/v1/ingest", json={"source": "/etc/passwd", "source_type": "text"})
    assert response.status_code == 400


class _LowReranker(Reranker):
    name = "low"

    def score(self, query, passages):
        return [0.0] * len(passages)


def test_low_confidence_retrieval_returns_no_answer(container):
    original = container.rag.reranker
    container.rag.reranker = _LowReranker()
    try:
        result = container.rag.answer("Tell me something about lab safety procedures")
        assert result.no_answer and result.answer == NO_ANSWER_MESSAGE
        assert result.sources == []
    finally:
        container.rag.reranker = original


def test_ui_is_served(client):
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200
    assert "<title>Component RAG Console</title>" in page.text
