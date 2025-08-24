from __future__ import annotations
from pathlib import Path

from .agent import build_agent, build_deps
from .tools.ingest import ingest_csv_file


def bootstrap_documents() -> str:
    deps = build_deps()
    deps.ensure_ready()

    # try CSVs first
    documents = deps.config.documents_dir
    csvs = list(documents.glob("*.csv"))
    messages: list[str] = []
    for c in csvs:
        messages.append(ingest_csv_file(deps, c))

    # PDFs
    # Prefer direct helper if available
    try:
        from .tools.pdf_ingest import extract_text_from_pdf  # type: ignore
        from .db.sql import INSERT_MEMORY  # type: ignore
        cur = deps.db.conn.cursor()
        pdfs = sorted((deps.config.documents_dir).glob("*.pdf"))[:10]
        count = 0
        for p in pdfs:
            try:
                text = extract_text_from_pdf(p)
                snippet = (text[:2000] + "...") if len(text) > 2000 else text
                cur.execute(INSERT_MEMORY, ("summary", f"PDF {p.name} summary/snippet:\n{snippet}", f"pdf,{p.name}"))
                count += 1
            except Exception:
                continue
        deps.db.conn.commit()
        messages.append(f"Ingested {count} PDF snippets from documents")
    except Exception:
        messages.append("PDF ingestion helper not available at bootstrap time.")

    return "\n".join(messages)
