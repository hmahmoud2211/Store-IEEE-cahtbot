#!/usr/bin/env python
"""Seed the knowledge base with the initial hardware-component price list.

    Raw table → parse component → normalize name → generate aliases → detect category
    → parse price range → store structured component → natural-language chunk
    → embedding → pgvector

Usage:
    python scripts/ingest_initial_data.py [--file data/hardware_components.csv] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.container import AppContainer  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.db.repositories.components import ComponentQuery, ComponentRepository  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.ingestion.pipeline import IngestionOptions  # noqa: E402
from app.ingestion.price_parser import format_amount  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=str(ROOT / "data" / "hardware_components.csv"))
    parser.add_argument("--force", action="store_true", help="Re-ingest even if the file hash is unchanged")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    container = AppContainer(settings)

    print(f"Embedding model: {container.embedder.model_name} ({container.embedder.dimension}-d)")
    result = container.ingestion.ingest(
        args.file,
        IngestionOptions(
            source_type="csv",
            title="Hardware Components Price List",
            metadata={"country": "Egypt", "currency": "EGP", "tags": ["price-list", "seed"], "date": "2026"},
            force=args.force,
        ),
    )
    print(f"\nStatus: {result.status}  document_id={result.document_id}  job_id={result.job_id}")
    if result.message:
        print(f"Message: {result.message}")
    if result.status == "failed":
        return 1
    print(f"Chunks created: {result.chunks_created}   Components upserted: {result.components_upserted}")

    with session_scope(container.session_factory) as session:
        rows, _ = ComponentRepository(session).search(ComponentQuery(limit=500))
        print(f"\n{'Component':38} {'Category':18} {'Price':>14}  Aliases")
        print("-" * 110)
        for c in sorted(rows, key=lambda c: (c.category, c.name)):
            price = (
                f"{format_amount(c.price_min)}–{format_amount(c.price_max)}"
                if c.price_min != c.price_max
                else format_amount(c.price_min)
            )
            print(f"{c.name:38} {c.category:18} {price + ' ' + (c.currency or ''):>14}  {', '.join(c.aliases[:4])}")
    container.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
