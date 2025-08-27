from __future__ import annotations
from pathlib import Path

from .agent import build_agent, build_deps
from .tools.ingest import ingest_csv_file


def bootstrap_documents() -> str:
    with build_deps() as deps:
        # try CSVs first
        documents = deps.config.documents_dir
        csvs = list(documents.glob("*.csv"))
        messages: list[str] = []
        for c in csvs:
            messages.append(ingest_csv_file(deps, c))
        
        # Process Excel files
        try:
            from .tools.excel_ingest import process_excel_file
            excel_files = list(documents.glob("*.xlsx")) + list(documents.glob("*.xls"))
            for excel_file in excel_files:
                try:
                    count = process_excel_file(deps, excel_file)
                    messages.append(f"Ingested {count} transactions from {excel_file.name}")
                except Exception as e:
                    messages.append(f"Failed to process {excel_file.name}: {str(e)[:100]}")
        except ImportError:
            pass  # Excel processing not available

        # PDFs
        # Prefer direct helper if available
        try:
            from .tools.pdf_ingest import extract_text_from_pdf  # type: ignore
            from .db.sql import INSERT_MEMORY  # type: ignore
            cur = deps.db.conn.cursor()
            pdfs = sorted((deps.config.documents_dir).glob("*.pdf"))[:10]
            count = 0
            skipped = 0
            for p in pdfs:
                try:
                    text = extract_text_from_pdf(p)
                    if not text or not text.strip():
                        skipped += 1
                        continue
                    snippet = (text[:2000] + "...") if len(text) > 2000 else text
                    cur.execute(INSERT_MEMORY, ("summary", f"PDF {p.name} summary/snippet:\n{snippet}", f"pdf,{p.name}"))
                    count += 1
                except Exception:
                    skipped += 1
                    continue
            deps.db.conn.commit()
            pdf_msg = f"Ingested {count} PDF snippets from documents"
            if skipped > 0:
                pdf_msg += f" ({skipped} PDFs skipped due to extraction issues)"
            messages.append(pdf_msg)
        except Exception:
            messages.append("PDF ingestion helper not available at bootstrap time.")

        return "\n".join(messages)
