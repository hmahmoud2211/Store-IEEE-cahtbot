"""Context building, citations, grounding checks, pricing math and extractive answers."""

import uuid
from decimal import Decimal

from app.ingestion.price_parser import parse_price
from app.rag.citations import check_answer, extract_citation_ids, format_sources_footer, select_cited_sources
from app.rag.context_builder import ContextBuilder
from app.rag.llm.extractive import render_extractive_answer
from app.rag.types import ComponentFact, PriceFilterResult, PriceSource, QueryType
from app.retrieval.types import SearchHit
from app.services.pricing_service import PricingService

DOC = PriceSource("doc-1", "Hardware Components Price List", "hardware_components.csv", "csv", None, None)


def fact(name: str, price: str, category: str = "Timer", cid: str | None = None) -> ComponentFact:
    p = parse_price(price)
    return ComponentFact(
        component_id=cid or str(uuid.uuid5(uuid.NAMESPACE_DNS, name)),
        name=name,
        category=category,
        price=p,
        sources=[PriceSource(DOC.document_id, DOC.document_title, DOC.source_name, "csv", None, p)],
    )


NE555 = fact("NE555 Timer IC", "4–10 EGP")
BREADBOARD = fact("400-point breadboard", "25 EGP", "Breadboard")
LED = fact("Standard LED", "1–3 EGP", "Output Device")
DS1307 = fact("DS1307 RTC IC", "20 EGP", "RTC")
DS3231 = fact("DS3231 RTC Module", "135 EGP", "RTC")


# ------------------------------------------------------------- pricing
def test_calculation_matches_spec_example():
    calc = PricingService().estimate(
        [
            (NE555, Decimal(2), "NE555", False, []),
            (BREADBOARD, Decimal(1), "breadboard", True, ["830-point breadboard"]),
            (LED, Decimal(5), "LED", False, []),
        ]
    )
    assert [i.line_total.format() for i in calc.items] == ["8–20 EGP", "25 EGP", "5–15 EGP"]
    assert calc.total.format() == "38–60 EGP"


def test_comparison_statements():
    result = PricingService().compare([DS1307, DS3231])
    text = " ".join(result.statements)
    assert "DS1307 RTC IC (20 EGP) is cheaper than DS3231 RTC Module (135 EGP) by 115 EGP" in text
    assert "same category: RTC" in text


def test_comparison_overlapping_ranges():
    a, b = fact("A", "5–18 EGP", "Display"), fact("B", "15 EGP", "Display")
    assert "overlap" in PricingService().compare([a, b]).statements[0]


# ------------------------------------------------------------- context
def _bundle_with_chunk():
    chunk = SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Timers section: the NE555 can be used as an astable oscillator.",
        metadata={},
        page_number=14,
        document_title="Electronics Catalog",
        source_name="electronics_catalog.pdf",
    )
    dup = SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="NE555 Timer IC. Approximate price 4 to 10 EGP.",
        metadata={"component_id": NE555.component_id},
    )
    return ContextBuilder().build(
        query_type=QueryType.FACT_LOOKUP,
        question="What is the price of the NE555?",
        components=[NE555],
        hits=[dup, chunk],
        not_found=["ESP32"],
    )


def test_context_numbers_sources_and_merges_duplicates():
    bundle = _bundle_with_chunk()
    assert [s.source_id for s in bundle.sources] == [1, 2]  # component chunk merged into structured source
    rendered = bundle.render()
    assert "[SOURCE 1]" in rendered and "[SOURCE 2]" in rendered
    assert "Approximate price: 4–10 EGP" in rendered
    assert "ESP32: no matching record" in rendered
    assert bundle.sources[0].label == "Hardware Components Price List — NE555 Timer IC"
    assert bundle.sources[1].label == "electronics_catalog.pdf — page 14"


def test_citation_validation_and_grounding():
    bundle = _bundle_with_chunk()
    answer = "The NE555 Timer IC costs approximately 4–10 EGP [1][7]. It can be an oscillator [2]."
    cleaned, report = check_answer(answer, bundle)
    assert "[7]" not in cleaned
    assert report.invalid_citations == [7]
    assert report.cited_ids == [1, 2]
    assert report.grounded


def test_grounding_detects_invented_price():
    bundle = _bundle_with_chunk()
    _, report = check_answer("The NE555 costs about 12 EGP [1].", bundle)
    assert not report.grounded
    assert report.ungrounded_numbers == ["12"]


def test_llm_sources_section_is_stripped():
    bundle = _bundle_with_chunk()
    cleaned, _ = check_answer("Price is 4–10 EGP [1].\n\nSources:\n[1] something made up", bundle)
    assert "made up" not in cleaned


def test_sources_footer():
    bundle = _bundle_with_chunk()
    footer = format_sources_footer(select_cited_sources(bundle, [1]))
    assert footer == "Sources:\n[1] Hardware Components Price List — NE555 Timer IC"
    assert extract_citation_ids("see [1, 2] and [SOURCE 3]") == [1, 2, 3]


# ----------------------------------------------------------- extractive
def test_extractive_fact_answer_is_grounded():
    bundle = _bundle_with_chunk()
    text = render_extractive_answer(bundle)
    assert "The NE555 Timer IC costs approximately 4–10 EGP [1]." in text
    assert "could not find ESP32" in text
    _, report = check_answer(text, bundle)
    assert report.grounded


def test_extractive_price_filter_answer():
    counter = fact("74LS161 Counter", "18 EGP", "Counter")
    partial = fact("CD4017 Counter", "15–25 EGP", "Counter")
    partial.partial_match = True
    pf = PriceFilterResult(
        matches=[counter], partial=[partial], constraint="less than 25 EGP", categories=["Counter"], sort=None
    )
    bundle = ContextBuilder().build(
        query_type=QueryType.PRICE_FILTER, question="q", components=[counter, partial], hits=[], price_filter=pf
    )
    text = render_extractive_answer(bundle)
    assert "74LS161 Counter: 18 EGP [1]" in text
    assert "CD4017 Counter: 15–25 EGP [2]" in text
    assert "straddles" in text
