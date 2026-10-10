"""Web page loader: fetches HTML, strips boilerplate, keeps headings/paragraphs/lists/tables."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import httpx

from app.ingestion.loaders.base import (
    BaseDocumentLoader,
    Element,
    ElementType,
    LoadedDocument,
    LoaderError,
    table_element,
)

_REMOVE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe", "svg", "button"]
_BOILERPLATE_ATTR_RE = re.compile(
    r"cookie|consent|banner|advert|\bads?\b|promo|subscribe|newsletter|sidebar|menu|breadcrumb|social|share|popup|modal",
    re.IGNORECASE,
)
_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "pre", "dl"]
_CONTAINER_TAGS = {"table", "ul", "ol", "pre", "dl"}


class WebLoader(BaseDocumentLoader):
    source_type = "web"

    def __init__(self, timeout: float = 20.0, user_agent: str = "HardwareRAGBot/1.0 (+ingestion)") -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        url = str(source)
        html: str | None = options.get("html")
        raw: bytes
        if html is None:
            try:
                response = httpx.get(
                    url, timeout=self.timeout, follow_redirects=True, headers={"User-Agent": self.user_agent}
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise LoaderError(f"Could not fetch {url}: {exc}") from exc
            raw = response.content
            html = response.text
        else:
            raw = html.encode("utf-8")
        return self.parse_html(html, url=url, raw=raw, **options)

    def parse_html(self, html: str, *, url: str, raw: bytes | None = None, **options: Any) -> LoadedDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        page_title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""

        for tag in soup(_REMOVE_TAGS):
            tag.decompose()
        for tag in soup.find_all(True):
            if tag.attrs is None:
                continue
            attrs = " ".join([tag.get("id") or "", *(tag.get("class") or []), tag.get("role") or ""])
            if (
                attrs.strip()
                and _BOILERPLATE_ATTR_RE.search(attrs)
                and tag.name not in ("html", "body", "main", "article")
            ):
                tag.decompose()

        root = soup.find("main") or soup.find("article") or soup.body or soup
        elements: list[Element] = []
        for el in root.find_all(_BLOCK_TAGS):
            # Skip blocks nested in a container we already emit as a whole.
            if any(parent.name in _CONTAINER_TAGS for parent in el.parents if parent is not root):
                continue
            name = el.name
            if name and name.startswith("h") and len(name) == 2:
                text = el.get_text(" ", strip=True)
                if text:
                    elements.append(Element(ElementType.HEADING, text, level=int(name[1])))
            elif name == "p":
                text = el.get_text(" ", strip=True)
                if text:
                    elements.append(Element(ElementType.PARAGRAPH, text))
            elif name in ("ul", "ol"):
                items = [li.get_text(" ", strip=True) for li in el.find_all("li", recursive=False)]
                items = [i for i in items if i]
                if items:
                    elements.append(Element(ElementType.LIST, "\n".join(f"- {i}" for i in items)))
            elif name == "table":
                rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
                rows = [r for r in rows if any(r)]
                if len(rows) >= 2:
                    elements.append(table_element(rows[0], rows[1:]))
            elif name in ("pre", "dl"):
                text = el.get_text("\n", strip=True)
                if text:
                    elements.append(Element(ElementType.PARAGRAPH, text))

        host = urlparse(url).netloc or url
        return LoadedDocument(
            title=options.get("title") or page_title or host,
            source_type=self.source_type,
            source_name=options.get("source_name") or host,
            source_url=url,
            raw_content=raw,
            elements=elements,
            metadata={"source_url": url, "domain": host, "supplier": options.get("supplier")},
        )
