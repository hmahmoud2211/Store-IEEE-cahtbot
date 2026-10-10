#!/usr/bin/env python
"""Generic ingestion CLI for any supported source.

Examples:
    python scripts/ingest.py data/catalogs/electronics_catalog.pdf
    python scripts/ingest.py data/datasheets/ds3231.docx --title "DS3231 datasheet notes"
    python scripts/ingest.py https://supplier.example/timers --type web --supplier "Supplier X"
    python scripts/ingest.py data/prices.xlsx --metadata '{"supplier": "Supplier Y", "date": "2026-10"}'
    python scripts/ingest.py '{"url": "postgresql+psycopg://...", "query": "SELECT name, price FROM items"}' --type database
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.container import AppContainer  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.ingestion.pipeline import IngestionOptions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="File path, URL, or JSON spec for database/api sources")
    parser.add_argument("--type", dest="source_type", help="csv|excel|pdf|docx|web|api|database|text")
    parser.add_argument("--title")
    parser.add_argument("--supplier")
    parser.add_argument("--metadata", default="{}", help="JSON object merged into document metadata")
    parser.add_argument("--visibility", default="public", choices=["public", "organization", "private"])
    parser.add_argument("--organization-id")
    parser.add_argument("--owner-user-id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)

    source: str | dict = args.source
    if args.source.strip().startswith("{"):
        source = json.loads(args.source)
    metadata = json.loads(args.metadata)
    if args.supplier:
        metadata["supplier"] = args.supplier

    container = AppContainer(settings)
    result = container.ingestion.ingest(
        source,
        IngestionOptions(
            source_type=args.source_type,
            title=args.title,
            metadata=metadata,
            force=args.force,
            visibility=args.visibility,
            organization_id=args.organization_id,
            owner_user_id=args.owner_user_id,
        ),
    )
    print(
        json.dumps({k: str(v) if v is not None else None for k, v in result.__dict__.items() if k != "stats"}, indent=2)
    )
    container.close()
    return 0 if result.status in ("completed", "duplicate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
