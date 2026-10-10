import uuid
from decimal import Decimal

import pytest

from app.core.security import Principal
from app.embeddings.provider import HashingEmbeddingProvider
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.keyword_search import build_keyword_query
from app.retrieval.metadata_filter import RetrievalFilters, acl_sql, filters_sql, post_filter
from app.retrieval.reranker import PassThroughReranker, Reranker
from app.retrieval.types import SearchHit


def hit(name: str, **kw) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, name),
        document_id=uuid.uuid5(uuid.NAMESPACE_DNS, "doc"),
        content=name,
        metadata=kw.pop("metadata", {}),
        **kw,
    )


# ------------------------------------------------------------------ RRF
def test_rrf_rewards_agreement():
    a, b, c = hit("a"), hit("b"), hit("c")
    vector = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("c"), hit("a")]
    fused = reciprocal_rank_fusion({"vector": vector, "keyword": keyword}, k=60)
    assert [h.content for h in fused] == ["a", "c", "b"]
    assert fused[0].scores["rrf"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2].scores["rrf"] == pytest.approx(1 / 62)
    # ranks from both retrievers are merged on the fused hit
    assert fused[0].ranks == {"vector": 1, "keyword": 2}
    del a, b, c


def test_rrf_weights_and_top_n():
    vector = [hit("x"), hit("y")]
    keyword = [hit("y"), hit("x")]
    fused = reciprocal_rank_fusion({"vector": vector, "keyword": keyword}, k=60, weights={"keyword": 3.0}, top_n=1)
    assert [h.content for h in fused] == ["y"]


# -------------------------------------------------------------- filters
def test_acl_sql_for_anonymous_user_and_org():
    sql, params = acl_sql(Principal.anonymous())
    assert sql == "(c.visibility = 'public')"
    assert params == {}
    sql, params = acl_sql(Principal(user_id="u1", organization_id="o1"))
    assert ":acl_org" in sql and ":acl_user" in sql
    assert params == {"acl_org": "o1", "acl_user": "u1"}
    assert acl_sql(Principal.system())[0] == "TRUE"


def test_filters_sql():
    f = RetrievalFilters(categories=["Counter"], currency="egp", max_price=Decimal(25), metadata={"supplier": "X"})
    sql, params = filters_sql(f, Principal.anonymous())
    assert "lower(c.metadata->>'category') = ANY(:f_categories)" in sql
    assert params["f_categories"] == ["counter"]
    assert params["f_currency"] == "EGP"
    assert params["f_max_price"] == Decimal(25)
    assert params["f_metadata"] == '{"supplier": "X"}'


def test_post_filter_enforces_acl_filters_and_score_gate():
    public = hit("public", visibility="public", metadata={"category": "Counter", "price_min": 18, "price_max": 18})
    private = hit("private", visibility="private", owner_user_id="someone", metadata={"category": "Counter"})
    weak = hit("weak", visibility="public", metadata={"category": "Counter", "price_min": 1, "price_max": 1})
    lexical = hit("lexical", visibility="public", metadata={"category": "Counter", "price_min": 5, "price_max": 5})
    public.scores["vector_similarity"] = 0.8
    weak.scores["vector_similarity"] = 0.1
    lexical.scores["vector_similarity"] = 0.1
    lexical.ranks["keyword"] = 1
    expensive = hit(
        "expensive", visibility="public", metadata={"category": "Counter", "price_min": 30, "price_max": 30}
    )
    expensive.scores["vector_similarity"] = 0.9

    kept = post_filter(
        [public, private, weak, lexical, expensive],
        RetrievalFilters(max_price=Decimal(25)),
        Principal.anonymous(),
        min_vector_similarity=0.5,
    )
    assert [h.content for h in kept] == ["public", "lexical"]


def test_owner_sees_private_chunk():
    private = hit("private", visibility="private", owner_user_id="u1")
    private.ranks["keyword"] = 1
    assert post_filter([private], None, Principal(user_id="u1"), 0.5) == [private]
    assert post_filter([private], None, Principal(user_id="u2"), 0.5) == []


# -------------------------------------------------------------- keyword
def test_keyword_query_uses_identifier_variants_and_drops_stopwords():
    kq = build_keyword_query("What is the price of the HC SR04?")
    assert "hcsr04" in kq.identifiers.split(" | ")
    assert "price" not in kq.simple and "what" not in kq.simple
    kq2 = build_keyword_query("7485")
    assert "7485" in kq2.identifiers


def test_keyword_query_sanitized():
    kq = build_keyword_query("counter'); DROP TABLE documents; --")
    for term in kq.simple.split(" | "):
        assert term.isalnum()


# ------------------------------------------------------------- reranker
class FakeReranker(Reranker):
    name = "fake"

    def score(self, query, passages):
        return [1.0 if query.lower() in p.lower() else 0.1 for p in passages]


def test_reranker_reorders_and_truncates():
    hits = [hit("unrelated"), hit("NE555 timer"), hit("other")]
    out = FakeReranker().rerank("ne555", hits, top_k=2)
    assert out[0].content == "NE555 timer"
    assert out[0].scores["rerank"] == 1.0
    assert out[0].ranks["rerank"] == 1
    assert len(out) == 2


def test_passthrough_reranker_keeps_order():
    hits = [hit("a"), hit("b"), hit("c")]
    out = PassThroughReranker().rerank("q", hits)
    assert [h.content for h in out] == ["a", "b", "c"]
    assert out[0].scores["rerank"] > out[1].scores["rerank"] > out[2].scores["rerank"]


# ------------------------------------------------------------ embedding
def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbeddingProvider(dimension=64)
    v1, v2 = emb.embed_query("NE555 timer"), emb.embed_query("NE555 timer")
    assert v1 == v2
    assert sum(x * x for x in v1) == pytest.approx(1.0)
    close = sum(a * b for a, b in zip(v1, emb.embed_query("555 timer IC"), strict=True))
    far = sum(a * b for a, b in zip(v1, emb.embed_query("breadboard jumper wires"), strict=True))
    assert close > far
