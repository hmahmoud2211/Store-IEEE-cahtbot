"""Loader registry: resolves a loader by explicit source type, URL or file extension.

New sources are added by registering a `BaseDocumentLoader` subclass; the
ingestion pipeline never imports concrete loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.loaders.api import APILoader
from app.ingestion.loaders.base import BaseDocumentLoader, LoaderError
from app.ingestion.loaders.csv import CSVLoader
from app.ingestion.loaders.database import DatabaseLoader
from app.ingestion.loaders.docx import DOCXLoader
from app.ingestion.loaders.excel import ExcelLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.loaders.text import TextLoader
from app.ingestion.loaders.web import WebLoader


class LoaderRegistry:
    def __init__(self) -> None:
        self._by_type: dict[str, BaseDocumentLoader] = {}
        self._by_ext: dict[str, BaseDocumentLoader] = {}

    def register(self, loader: BaseDocumentLoader, *aliases: str) -> None:
        for name in (loader.source_type, *aliases):
            self._by_type[name.lower()] = loader
        for ext in loader.supported_extensions:
            self._by_ext[ext.lower()] = loader

    @property
    def source_types(self) -> list[str]:
        return sorted(self._by_type)

    def get(self, source_type: str) -> BaseDocumentLoader:
        try:
            return self._by_type[source_type.lower()]
        except KeyError as exc:
            raise LoaderError(f"No loader registered for source type '{source_type}'") from exc

    def resolve(self, source: Any, source_type: str | None = None) -> BaseDocumentLoader:
        if source_type:
            return self.get(source_type)
        if isinstance(source, dict):
            if "query" in source:
                return self.get("database")
            if "url" in source:
                return self.get("api")
        if isinstance(source, str) and source.lower().startswith(("http://", "https://")):
            return self.get("web")
        suffix = Path(str(source)).suffix.lower()
        if suffix in self._by_ext:
            return self._by_ext[suffix]
        raise LoaderError(f"Cannot infer a loader for source '{source}'. Pass source_type explicitly.")


def default_registry() -> LoaderRegistry:
    registry = LoaderRegistry()
    registry.register(CSVLoader())
    registry.register(ExcelLoader(), "xlsx", "xls")
    registry.register(PDFLoader())
    registry.register(DOCXLoader())
    registry.register(WebLoader(), "url", "html")
    registry.register(DatabaseLoader(), "db", "sql")
    registry.register(APILoader(), "rest")
    registry.register(TextLoader(), "txt", "markdown", "md")
    return registry
