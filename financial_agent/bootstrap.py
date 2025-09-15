# financial_agent/bootstrap.py
from __future__ import annotations
from typing import cast

from .agent import build_deps  # Keep only necessary imports
from .context import RunDeps
from .tools.ingest import ingest_csv_file

# Pylint struggles with Pydantic's dynamic attributes (config/db fields)
# We explicitly disable no-member here since runtime attributes are valid.
# pylint: disable=no-member


def bootstrap_documents() -> str:
    """Bootstrap the database door CSV, Excel en PDF te verwerken.
    Retourneert een mens-leesbaar verslag.
    """
    # Help static analyzers (pylint/pyright) understand types from pydantic models
    with build_deps() as deps_any:
        deps: RunDeps = cast(RunDeps, deps_any)
        messages: list[str] = []

        # CSVs
        for c in deps.config.documents_dir.glob("*.csv"):
            messages.append(ingest_csv_file(deps, c))

        # Excel
        _ingest_excels(deps, messages)

        # PDFs
        _ingest_pdfs(deps, messages)

        return "\n".join(messages)


def _ingest_excels(deps: RunDeps, messages: list[str]) -> None:
    try:
        from .tools.excel_ingest import process_excel_file

        for pattern in ("*.xlsx", "*.xls"):
            for excel_file in deps.config.documents_dir.glob(pattern):
                try:
                    count = process_excel_file(deps, excel_file)
                    messages.append(
                        f"Ingested {count} transactions from {excel_file.name}"
                    )
                except ValueError:
                    # Known, expected validation/parsing issues
                    messages.append(f"Failed to process {excel_file.name}.")
    except ImportError:
        messages.append("Excel ingestion helper not available at bootstrap time.")


def _ingest_pdfs(deps: RunDeps, messages: list[str]) -> None:
    try:
        from .tools.pdf_ingest import extract_text_from_pdf  # type: ignore
        from .db.sql import INSERT_MEMORY  # type: ignore

        cur = deps.db.conn.cursor()
        pdfs = sorted(deps.config.documents_dir.glob("*.pdf"))[:10]
        count, skipped = 0, 0

        for p in pdfs:
            try:
                text = extract_text_from_pdf(p)
                if not text.strip():
                    skipped += 1
                    continue
                snippet = (text[:2000] + "...") if len(text) > 2000 else text
                params = (
                    "summary",
                    f"PDF {p.name} summary:\n{snippet}",
                    f"pdf,{p.name}",
                )
                cur.execute(INSERT_MEMORY, params)
                count += 1
            except Exception:  # pylint: disable=broad-exception-caught
                skipped += 1
        deps.db.conn.commit()
        pdf_msg = f"Ingested {count} PDF snippets from documents"
        if skipped:
            pdf_msg += f" ({skipped} PDFs skipped due to extraction issues)"
        messages.append(pdf_msg)
    except ImportError:
        messages.append("PDF ingestion helper not available at bootstrap time.")
