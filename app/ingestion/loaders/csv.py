from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument, LoaderError, records_to_elements


class CSVLoader(BaseDocumentLoader):
    """Each CSV row becomes a RECORD element (one logical item per chunk downstream)."""

    source_type = "csv"
    supported_extensions = (".csv", ".tsv")

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        path, raw = self._read_file(source)
        sep = options.get("delimiter") or ("\t" if path.suffix.lower() == ".tsv" else None)
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                dtype=str,
                keep_default_na=False,
                encoding=options.get("encoding", "utf-8-sig"),
            )
        except Exception as exc:  # pandas raises many error types
            raise LoaderError(f"Could not parse CSV {path.name}: {exc}") from exc

        df.columns = [str(c).strip() for c in df.columns]
        elements = records_to_elements(df.to_dict(orient="records"), section=options.get("section"))
        return LoadedDocument(
            title=options.get("title") or Path(path).stem.replace("_", " ").title(),
            source_type=self.source_type,
            source_name=path.name,
            file_path=str(path),
            raw_content=raw,
            elements=elements,
            metadata={"file_name": path.name, "columns": list(df.columns), "row_count": len(df)},
        )
