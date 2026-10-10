from __future__ import annotations

from typing import Any

import pandas as pd

from app.ingestion.loaders.base import (
    BaseDocumentLoader,
    Element,
    ElementType,
    LoadedDocument,
    LoaderError,
    records_to_elements,
)


class ExcelLoader(BaseDocumentLoader):
    """Every sheet becomes a section (heading) followed by one RECORD per row."""

    source_type = "excel"
    supported_extensions = (".xlsx", ".xlsm", ".xls")

    def load(self, source: Any, **options: Any) -> LoadedDocument:
        path, raw = self._read_file(source)
        try:
            sheets = pd.read_excel(path, sheet_name=options.get("sheet_name"), dtype=str, keep_default_na=False)
        except Exception as exc:
            raise LoaderError(f"Could not parse Excel file {path.name}: {exc}") from exc
        if isinstance(sheets, pd.DataFrame):
            sheets = {options.get("sheet_name") or "Sheet1": sheets}

        elements: list[Element] = []
        row_offset = 0
        for sheet_name, df in sheets.items():
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]
            if df.empty:
                continue
            elements.append(Element(type=ElementType.HEADING, text=str(sheet_name), level=1))
            elements.extend(
                records_to_elements(df.to_dict(orient="records"), section=str(sheet_name), start_index=row_offset)
            )
            row_offset += len(df)

        return LoadedDocument(
            title=options.get("title") or path.stem.replace("_", " ").title(),
            source_type=self.source_type,
            source_name=path.name,
            file_path=str(path),
            raw_content=raw,
            elements=elements,
            metadata={"file_name": path.name, "sheets": [str(s) for s in sheets]},
        )
