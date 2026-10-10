"""Load rows from any SQLAlchemy-supported database as RECORD elements.

source = {
    "url": "postgresql+psycopg://user:pass@host/db",   # never logged
    "query": "SELECT name AS component, price FROM supplier_catalog",
    "title": "Supplier catalog",
    "name": "supplier_catalog",
}
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import create_engine, text

from app.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument, LoaderError, records_to_elements


class DatabaseLoader(BaseDocumentLoader):
    source_type = "database"

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        if not isinstance(source, dict) or "url" not in source or "query" not in source:
            raise LoaderError("DatabaseLoader expects {'url': ..., 'query': ...}")
        query = str(source["query"])
        if not query.lstrip().lower().startswith(("select", "with")):
            raise LoaderError("DatabaseLoader only runs read-only SELECT queries")

        engine = create_engine(source["url"], pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), source.get("params") or {})
                rows = [dict(r._mapping) for r in result]
        except Exception as exc:
            raise LoaderError(f"Database query failed: {exc.__class__.__name__}") from exc
        finally:
            engine.dispose()

        serialized = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
        name = source.get("name") or hashlib.sha1(query.encode()).hexdigest()[:12]
        return LoadedDocument(
            title=source.get("title") or f"Database query {name}",
            source_type=self.source_type,
            source_name=name,
            raw_content=serialized,
            elements=records_to_elements(rows),
            metadata={"row_count": len(rows), "supplier": source.get("supplier")},
        )
