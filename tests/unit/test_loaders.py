from pathlib import Path

import pytest

from app.ingestion.loaders.base import ElementType, LoaderError
from app.ingestion.loaders.registry import default_registry
from app.ingestion.loaders.text import TextLoader
from app.ingestion.loaders.web import WebLoader

registry = default_registry()


def test_registry_resolution(tmp_path: Path):
    assert registry.resolve("file.pdf").source_type == "pdf"
    assert registry.resolve("file.docx").source_type == "docx"
    assert registry.resolve("file.csv").source_type == "csv"
    assert registry.resolve("file.xlsx").source_type == "excel"
    assert registry.resolve("notes.md").source_type == "text"
    assert registry.resolve("https://example.com").source_type == "web"
    assert registry.resolve({"url": "sqlite://", "query": "select 1"}).source_type == "database"
    assert registry.resolve({"url": "https://api.example.com"}).source_type == "api"
    with pytest.raises(LoaderError):
        registry.resolve("file.unknown")


def test_csv_loader_rows(seed_records):
    from tests.conftest import SEED_CSV

    doc = registry.get("csv").load(SEED_CSV)
    assert doc.source_name == "hardware_components.csv"
    assert len(doc.elements) == 32
    assert all(e.type is ElementType.RECORD for e in doc.elements)
    assert doc.elements[0].rows[0]["price"] == "4–10 EGP"
    assert len(doc.content_hash()) == 64


def test_text_loader_markdown_structure(tmp_path: Path):
    path = tmp_path / "catalog.md"
    path.write_text(
        "# Catalog\n\nIntro paragraph.\n\n## Timers\n\n- NE555\n- NE556\n\n| Part | Price |\n|---|---|\n| NE555 | 4–10 EGP |\n",
        encoding="utf-8",
    )
    doc = TextLoader().load(path)
    types = [e.type for e in doc.elements]
    assert types == [
        ElementType.HEADING,
        ElementType.PARAGRAPH,
        ElementType.HEADING,
        ElementType.LIST,
        ElementType.TABLE,
    ]
    assert doc.elements[-1].rows == [{"Part": "NE555", "Price": "4–10 EGP"}]
    assert doc.title == "Catalog"


def test_web_loader_strips_boilerplate():
    html = """
    <html><head><title>Supplier X – Timers</title><script>var x=1;</script></head>
    <body>
      <nav>Home | Shop</nav>
      <div class="cookie-banner">Accept cookies</div>
      <main>
        <h1>Timer ICs</h1>
        <p>The NE555 is a popular timer.</p>
        <ul><li>Astable mode</li><li>Monostable mode</li></ul>
        <table><tr><th>Part</th><th>Price</th></tr><tr><td>NE555</td><td>6 EGP</td></tr></table>
      </main>
      <footer>© 2026 Supplier X</footer>
    </body></html>
    """
    doc = WebLoader().parse_html(html, url="https://supplier.example/timers")
    texts = " ".join(e.text for e in doc.elements)
    assert doc.title == "Supplier X – Timers"
    assert "Accept cookies" not in texts and "Home | Shop" not in texts and "2026" not in texts
    assert [e.type for e in doc.elements] == [
        ElementType.HEADING,
        ElementType.PARAGRAPH,
        ElementType.LIST,
        ElementType.TABLE,
    ]
    assert doc.elements[-1].rows == [{"Part": "NE555", "Price": "6 EGP"}]


def test_docx_loader(tmp_path: Path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("RTC Modules", level=1)
    d.add_paragraph("The DS3231 is a precise RTC.")
    d.add_paragraph("Battery backup", style="List Bullet")
    table = d.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Component", "Price"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "DS3231 RTC Module", "135 EGP"
    path = tmp_path / "rtc.docx"
    d.save(path)

    doc = registry.get("docx").load(path)
    types = [e.type for e in doc.elements]
    assert types == [ElementType.HEADING, ElementType.PARAGRAPH, ElementType.LIST, ElementType.TABLE]
    assert doc.elements[-1].rows == [{"Component": "DS3231 RTC Module", "Price": "135 EGP"}]


def test_component_extraction_from_table(tmp_path: Path):
    from app.ingestion.component_extractor import ComponentExtractor

    doc = TextLoader().load(
        tmp_path / "x.md", content="| Item | Unit price |\n|---|---|\n| LM7805 regulator | 12 LE |\n"
    )
    found = ComponentExtractor().extract(doc.elements)
    rec = next(iter(found.values()))[0]
    assert rec.name == "LM7805 regulator"
    assert rec.category.value == "Voltage Regulator"
    assert str(rec.price.min) == "12" and rec.price.currency == "EGP"
