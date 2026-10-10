from app.ingestion.chunking import StructureAwareChunker
from app.ingestion.cleaning import clean_document, clean_text
from app.ingestion.loaders.base import Element, ElementType, LoadedDocument, table_element
from app.ingestion.loaders.text import parse_markdown
from app.ingestion.metadata import MetadataEnricher, chunk_search_keys


def word_count(text: str) -> int:
    return len(text.split())


def _doc(elements, source_type="text"):
    return LoadedDocument(title="Doc", source_type=source_type, source_name="doc.md", elements=elements)


def test_records_become_one_chunk_each_without_overlap():
    elements = [
        Element(ElementType.RECORD, "component: NE555 Timer IC; price: 4–10 EGP", rows=[{}]),
        Element(ElementType.RECORD, "component: 74LS85; price: 25 EGP", rows=[{}]),
    ]
    chunks = StructureAwareChunker(50, 10, word_count).chunk(_doc(elements, "csv"))
    assert [c.content for c in chunks] == [e.text for e in elements]


def test_sections_are_kept_together_and_split_on_headings():
    md = """# Timers

The 555 is a timer.

- astable
- monostable

# Counters

The 4017 is a decade counter."""
    chunks = StructureAwareChunker(200, 20, word_count).chunk(_doc(parse_markdown(md)))
    assert len(chunks) == 2
    assert chunks[0].section == "Timers"
    assert "astable" in chunks[0].content and "555" in chunks[0].content
    assert chunks[1].section == "Counters"
    assert "555" not in chunks[1].content


def test_nested_heading_path():
    md = "# Catalog\n\n## Timers\n\nNE555 details."
    chunks = StructureAwareChunker(200, 20, word_count).chunk(_doc(parse_markdown(md)))
    assert chunks[0].section == "Catalog > Timers"


def test_long_prose_is_split_with_sentence_overlap():
    sentences = [f"Sentence number {i} talks about timers." for i in range(60)]
    elements = [Element(ElementType.PARAGRAPH, " ".join(sentences))]
    chunker = StructureAwareChunker(chunk_size=60, chunk_overlap=12, token_counter=word_count)
    chunks = chunker.chunk(_doc(elements))
    assert len(chunks) > 3
    assert all(word_count(c.content) <= 60 for c in chunks)
    # consecutive chunks share the overlap sentence(s)
    last_sentence = chunks[0].content.split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1].content


def test_large_table_split_by_rows_with_header_repeated():
    rows = [[f"Part{i}", f"{i} EGP"] for i in range(40)]
    element = table_element(["Component", "Price"], rows)
    chunks = StructureAwareChunker(chunk_size=40, chunk_overlap=5, token_counter=word_count).chunk(_doc([element]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content.splitlines()[0].startswith("| Component | Price |")
    all_text = "\n".join(c.content for c in chunks)
    assert all(f"Part{i} " in all_text or f"Part{i} |" in all_text for i in range(40))


def test_cleaning_preserves_technical_values():
    raw = "NE555 Timer   IC​ costs  4–10 EGP;  HC-SR04 runs at 5V"
    cleaned = clean_text(raw)
    assert cleaned == "NE555 Timer IC costs 4–10 EGP; HC-SR04 runs at 5V"


def test_clean_document_drops_boilerplate_and_keeps_tables():
    elements = [
        Element(ElementType.PARAGRAPH, "Accept all cookies"),
        Element(ElementType.PARAGRAPH, "The  74LS85   compares numbers."),
        table_element(["A", "B"], [["1", "2"]]),
    ]
    cleaned = clean_document(_doc(elements, "web"))
    assert [e.type for e in cleaned.elements] == [ElementType.PARAGRAPH, ElementType.TABLE]
    assert cleaned.elements[0].text == "The 74LS85 compares numbers."


def test_metadata_for_component_chunk(seed_records):
    from app.ingestion.chunking import ChunkDraft

    rec = next(r for r in seed_records if r.name == "NE555 Timer IC")
    doc = LoadedDocument(
        title="Hardware Components Price List",
        source_type="csv",
        source_name="hardware_components.csv",
        elements=[],
        metadata={"country": "Egypt"},
    )
    meta = MetadataEnricher("Egypt").build(
        document=doc,
        document_id="123",
        draft=ChunkDraft(content=rec.chunk_text("Egypt"), section="Timer"),
        chunk_index=0,
        component=rec,
        component_id="c-1",
        acl={"visibility": "public", "organization_id": None},
    )
    assert meta["source"] == "hardware_components.csv"
    assert meta["source_type"] == "csv"
    assert meta["component"] == "NE555 Timer IC"
    assert meta["category"] == "Timer"
    assert meta["price_min"] == 4.0 and meta["price_max"] == 10.0
    assert meta["currency"] == "EGP"
    assert meta["country"] == "Egypt"
    assert meta["section"] == "Timer"
    assert meta["visibility"] == "public"
    assert "organization_id" not in meta  # None values are dropped


def test_component_chunk_text(seed_records):
    rec = next(r for r in seed_records if r.name == "NE555 Timer IC")
    text = rec.chunk_text("Egypt")
    assert text.startswith("NE555 Timer IC. Category: Timer.")
    assert "Approximate price in Egypt: 4 to 10 EGP" in text
    assert '"4–10 EGP"' in text  # original preserved


def test_chunk_search_keys():
    keys = chunk_search_keys("The HC-SR04 and 74LS85 are listed.").split()
    assert {"hcsr04", "74ls85", "7485"} <= set(keys)
