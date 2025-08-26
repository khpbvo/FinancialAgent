from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader
from agents import RunContextWrapper, function_tool
from typing import Any

from ..context import RunDeps
from ..db.sql import INSERT_MEMORY


@dataclass
class PDFSummary:
    file: str
    num_pages: int
    snippet: str


def extract_text_from_pdf(path: Path, max_chars: int = 4000) -> str:
    reader = PdfReader(str(path))
    texts: list[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
        if sum(len(t) for t in texts) > max_chars:
            break
    return "\n".join(texts)


def pdf_error_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    """Custom error handler for PDF ingestion failures."""
    if "pypdf" in str(error).lower():
        return "Failed to read PDF. The file may be corrupted or password protected."
    elif "not found" in str(error).lower():
        return "No PDF files found in the documents directory."
    else:
        return f"PDF ingestion failed: {str(error)}. Please check the PDF files are valid."

@function_tool(failure_error_function=pdf_error_handler)
def ingest_pdfs(ctx: RunContextWrapper[RunDeps], directory: str | None = None, limit: int = 10) -> str:
    """Ingest PDF files by extracting text snippets and saving as memories.

    Args:
        directory: Optional relative dir under documents/ to scan; defaults to base documents dir
        limit: Max number of PDFs to process
    """
    deps = ctx.context
    base = deps.config.documents_dir if not directory else (deps.config.documents_dir / directory)
    if not base.exists():
        return f"Directory not found: {base}"

    pdfs = sorted([p for p in base.glob("*.pdf")])[:limit]
    if not pdfs:
        return "No PDFs found."

    inserted = 0
    cur = deps.db.conn.cursor()
    for pdf in pdfs:
        try:
            text = extract_text_from_pdf(pdf)
        except Exception as e:
            continue
        snippet = (text[:2000] + "...") if len(text) > 2000 else text
        cur.execute(INSERT_MEMORY, ("summary", f"PDF {pdf.name} summary/snippet:\n{snippet}", f"pdf,{pdf.name}"))
        inserted += 1
    deps.db.conn.commit()
    return f"Ingested {inserted} PDF snippets from {base.name}"
