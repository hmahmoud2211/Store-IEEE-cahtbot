"""Load records from a REST/JSON API.

source = {
    "url": "https://supplier.example/api/products",
    "method": "GET",
    "params": {"category": "ics"},
    "headers": {"Authorization": "Bearer ..."},   # never logged
    "records_path": "data.items",                  # dotted path to the list of records
    "title": "Supplier X catalog",
    "supplier": "Supplier X",
}
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from app.ingestion.loaders.base import (
    BaseDocumentLoader,
    Element,
    ElementType,
    LoadedDocument,
    LoaderError,
    records_to_elements,
)


def _dig(data: Any, path: str | None) -> Any:
    if not path:
        return data
    for part in path.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit():
            data = data[int(part)]
        else:
            return None
    return data


def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, list):
            flat[name] = ", ".join(str(v) for v in value)
        else:
            flat[name] = value
    return flat


class APILoader(BaseDocumentLoader):
    source_type = "api"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        spec = source if isinstance(source, dict) else {"url": str(source)}
        url = spec.get("url")
        if not url:
            raise LoaderError("APILoader requires a 'url'")
        try:
            response = httpx.request(
                spec.get("method", "GET"),
                url,
                params=spec.get("params"),
                headers=spec.get("headers"),
                json=spec.get("json"),
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LoaderError(f"API request to {urlparse(url).netloc} failed: {exc.__class__.__name__}") from exc

        data = _dig(payload, spec.get("records_path"))
        if isinstance(data, list) and all(isinstance(r, dict) for r in data):
            elements = records_to_elements([_flatten(r) for r in data])
        elif isinstance(data, dict):
            elements = records_to_elements([_flatten(data)])
        else:
            elements = [Element(ElementType.PARAGRAPH, json.dumps(data, ensure_ascii=False))]

        host = urlparse(url).netloc
        return LoadedDocument(
            title=spec.get("title") or f"API {host}",
            source_type=self.source_type,
            source_name=spec.get("name") or f"{host}{urlparse(url).path}",
            source_url=url,
            raw_content=response.content,
            elements=elements,
            metadata={"source_url": url, "supplier": spec.get("supplier")},
        )
