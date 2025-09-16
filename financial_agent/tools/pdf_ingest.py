from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import re

from PyPDF2 import PdfReader
from agents import RunContextWrapper, function_tool, Agent, ModelSettings, Runner
from pydantic import BaseModel, Field

from ..context import RunDeps
from ..db.sql import INSERT_MEMORY, INSERT_TRANSACTION


@dataclass
class PDFSummary:
    file: str
    num_pages: int
    snippet: str


def extract_text_from_pdf(path: Path, max_chars: int = 4000) -> str:
    """Extract text from PDF with multiple fallback methods."""

    # Method 1: Try PyPDF2 first
    try:
        reader = PdfReader(str(path))
        texts: list[str] = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():  # Only add non-empty text
                    texts.append(page_text)
            except Exception:
                continue
            if sum(len(t) for t in texts) > max_chars:
                break

        text_content = "\n".join(texts).strip()
        if text_content:  # If we got meaningful text, return it
            return text_content

    except Exception:
        pass

    # Method 2: If PyPDF2 fails or returns empty, try pdfplumber for better text extraction
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            texts = []
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        texts.append(page_text)
                except Exception:
                    continue
                if sum(len(t) for t in texts) > max_chars:
                    break
            text_content = "\n".join(texts).strip()
            if text_content:
                return text_content
    except ImportError:
        pass  # pdfplumber not available
    except Exception:
        pass

    # Method 3: As a fallback, provide metadata and structure info
    try:
        reader = PdfReader(str(path))
        metadata_parts = []

        # Add basic file info
        metadata_parts.append(f"PDF File: {path.name}")
        metadata_parts.append(f"Number of pages: {len(reader.pages)}")

        # Try to extract any metadata
        if reader.metadata:
            if reader.metadata.get("/Title"):
                metadata_parts.append(f"Title: {reader.metadata['/Title']}")
            if reader.metadata.get("/Creator"):
                metadata_parts.append(f"Creator: {reader.metadata['/Creator']}")
            if reader.metadata.get("/Subject"):
                metadata_parts.append(f"Subject: {reader.metadata['/Subject']}")

        # Try to get some structural information
        metadata_parts.append(
            "PDF Structure: This appears to be an image-based or protected PDF"
        )
        metadata_parts.append(
            "Note: Text extraction not possible - PDF may contain scanned images"
        )

        # Look for potential patterns in the filename that might indicate content type
        filename_lower = path.name.lower()
        if any(
            term in filename_lower
            for term in ["afschrift", "statement", "bank", "rekening"]
        ):
            metadata_parts.append(
                "Document Type: Likely a bank statement based on filename"
            )

        return "\n".join(metadata_parts)

    except Exception as e:
        return f"PDF processing failed for {path.name}: Unable to extract text or metadata. Error: {str(e)}"


def pdf_error_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    """Custom error handler for PDF ingestion failures."""
    if "pypdf" in str(error).lower():
        return "Failed to read PDF. The file may be corrupted or password protected."
    elif "not found" in str(error).lower():
        return "No PDF files found in the documents directory."
    else:
        return (
            f"PDF ingestion failed: {str(error)}. Please check the PDF files are valid."
        )


def _to_iso_date(d: str) -> str:
    # Accept dd-mm-YYYY or YYYY-mm-dd
    d = d.strip()
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", d)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", d)
    if m:
        return d
    return d


def _parse_amount_eur(s: str) -> Optional[float]:
    # Handle European decimal and optional sign
    s = s.strip()
    m = re.search(r"([+-]?)\s*(\d+[\.,]\d{2})$", s)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    val = m.group(2).replace(".", "").replace(",", ".")
    try:
        return sign * float(val)
    except Exception:
        return None


def _classify_pdf_text(text: str) -> str:
    tl = text.lower()
    if "afschrift creditcard" in tl or (
        "kaartnummer" in tl and "aflossing" in tl and "incasso" in tl
    ):
        return "credit_card_statement"
    if "bij- en afschrijvingen" in tl or "priverekening" in tl:
        return "bank_statement"
    return "other"


class PDFClassification(BaseModel):
    doc_type: str = Field(
        description="One of: credit_card_statement | bank_statement | other"
    )


async def _classify_pdf_llm(deps: RunDeps, text: str) -> str:
    """LLM-only classification using GPT-5 via the Agents SDK with structured outputs."""
    instructions = (
        "You are a precise document classifier."
        " Read the snippet and set doc_type to exactly one of:"
        " credit_card_statement | bank_statement | other."
    )
    agent = Agent[RunDeps](
        name="PDFClassifier",
        instructions=instructions,
        model="gpt-5",
        model_settings=ModelSettings(temperature=0),
        output_type=PDFClassification,
    )
    try:
        res = await Runner.run(
            agent, f"Snippet:\n{text[:7000]}", context=deps, max_turns=1
        )
        if isinstance(res.final_output, PDFClassification):
            return res.final_output.doc_type
        # Fallback to heuristic if structured output missing
        return _classify_pdf_text(text)
    except Exception:
        return _classify_pdf_text(text)


def _parse_credit_card(
    text: str, exclude_repayment: bool = True
) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries: List[Dict[str, Any]] = []

    # Find rows starting with a date and containing a Type and amount
    row_re = re.compile(
        r"^(\d{2}-\d{2}-\d{4})\s+(.+?)\s+(Betaling|Kosten|Geldopname|Incasso)\s+([+-]?\d+[\.,]\d{2})$",
        re.I,
    )
    buffer: List[str] = []

    def flush(buf: List[str]):
        if not buf:
            return
        text = " ".join(buf)
        m = row_re.match(text)
        if not m:
            return
        date, desc, typ, amount_str = m.group(1), m.group(2), m.group(3), m.group(4)
        amount = _parse_amount_eur(amount_str) or 0.0
        if (
            exclude_repayment
            and typ.lower() == "incasso"
            and ("aflossing" in desc.lower() or amount > 0)
        ):
            return
        category = "creditcard"
        if typ.lower() == "kosten":
            category = "fees"
        elif typ.lower() == "geldopname":
            category = "cash_withdrawal"
        entries.append(
            {
                "date": _to_iso_date(date),
                "description": desc,
                "amount": amount,  # Betaling/Kosten/Geldopname are negative in source
                "currency": "EUR",
                "category": category,
            }
        )

    # Build rows; lines may wrap, so accumulate until regex matches
    for line in lines:
        # If line starts with date, consider flushing previous
        if re.match(r"^\d{2}-\d{2}-\d{4}\b", line):
            if buffer:
                flush(buffer)
                buffer = []
        buffer.append(line)
        if row_re.match(" ".join(buffer)):
            flush(buffer)
            buffer = []
    if buffer:
        flush(buffer)

    return entries


def _guess_sign_from_description(desc: str, amount: float) -> float:
    d = desc.lower()
    negative_markers = [
        "bea",
        "betaalpas",
        "apple pay",
        "ideal",
        "incasso",
        "betaling",
        "kosten",
        "geldopname",
    ]
    positive_markers = ["storting", "bijschrijving", "refund", "terugbetaling", "rente"]
    if any(m in d for m in negative_markers):
        return -abs(amount)
    if any(m in d for m in positive_markers):
        return abs(amount)
    # Default to expense
    return -abs(amount)


def _parse_bank_statement(text: str) -> List[Dict[str, Any]]:
    # Very lightweight parser for ABN style text
    entries: List[Dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\d{2}-\d{2}-\d{4})\b(.*)$", line)
        if m:
            date = _to_iso_date(m.group(1))
            desc_parts = [m.group(2).strip()]
            j = i + 1
            amount_val: Optional[float] = None
            # Accumulate until next date or we find a trailing amount
            while j < len(lines) and not re.match(r"^\d{2}-\d{2}-\d{4}\b", lines[j]):
                amt = _parse_amount_eur(lines[j])
                if amt is not None:
                    amount_val = amt
                else:
                    desc_parts.append(lines[j])
                j += 1
            desc = " ".join([p for p in desc_parts if p])
            if amount_val is not None:
                signed = _guess_sign_from_description(desc, amount_val)
                entries.append(
                    {
                        "date": date,
                        "description": desc,
                        "amount": signed,
                        "currency": "EUR",
                        "category": None,
                    }
                )
            i = j
        else:
            i += 1
    return entries


def _insert_transactions(
    deps: RunDeps, source_file: str, rows: List[Dict[str, Any]]
) -> Tuple[int, int]:
    cur = deps.db.conn.cursor()
    inserted = 0
    skipped = 0
    for r in rows:
        date = r.get("date", "")
        desc = r.get("description", "")
        amount = float(r.get("amount", 0.0))
        currency = r.get("currency") or "EUR"
        category = r.get("category")
        # Duplicate guard (global across sources to handle overlapping statements)
        cur.execute(
            """
            SELECT 1 FROM transactions
            WHERE date = ? AND description = ?
              AND ABS(amount - ?) < 1e-9
              AND IFNULL(currency,'') = IFNULL(?, '')
            LIMIT 1
            """,
            (date, desc, amount, currency or ""),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            INSERT_TRANSACTION, (date, desc, amount, currency, category, source_file)
        )
        inserted += 1
    deps.db.conn.commit()
    return inserted, skipped


@function_tool(failure_error_function=pdf_error_handler)
async def ingest_pdfs(
    ctx: RunContextWrapper[RunDeps],
    directory: str | None = None,
    limit: int = 10,
    mode: str = "transactions",
    use_llm_classification: bool = True,
    exclude_credit_repayment: bool = True,
) -> str:
    """Ingest PDFs. Either store summaries (memories) or parse statements into transactions.

    Args:
        directory: Subfolder under documents/ to scan; defaults to base documents dir
        limit: Max PDFs to process
        mode: "transactions" to insert parsed transactions, "memories" to store text snippets
        use_llm_classification: If True, try LLM classification to detect doc type (fallback to heuristics)
        exclude_credit_repayment: When parsing credit card statements, skip monthly repayment lines
    """
    deps = ctx.context
    base = (
        deps.config.documents_dir
        if not directory
        else (deps.config.documents_dir / directory)
    )
    if not base.exists():
        return f"Directory not found: {base}"

    pdfs = sorted([p for p in base.glob("*.pdf")])[:limit]
    if not pdfs:
        return "No PDFs found."

    cur = deps.db.conn.cursor()
    summaries = 0
    tx_inserted = 0
    tx_skipped = 0
    skipped = 0

    for pdf in pdfs:
        try:
            text = extract_text_from_pdf(pdf)
            if not text or not text.strip():
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        if mode == "memories":
            snippet = (text[:2000] + "...") if len(text) > 2000 else text
            cur.execute(
                INSERT_MEMORY,
                (
                    "summary",
                    f"PDF {pdf.name} summary/snippet:\n{snippet}",
                    f"pdf,{pdf.name}",
                ),
            )
            summaries += 1
            continue

        # transactions mode
        # LLM-only classification (fallback to heuristic on error)
        doc_type = await _classify_pdf_llm(deps, text)

        rows: List[Dict[str, Any]] = []
        if doc_type == "credit_card_statement":
            rows = _parse_credit_card(text, exclude_repayment=exclude_credit_repayment)
        elif doc_type == "bank_statement":
            rows = _parse_bank_statement(text)
        else:
            # As a fallback, store summary memory
            snippet = (text[:2000] + "...") if len(text) > 2000 else text
            cur.execute(
                INSERT_MEMORY,
                (
                    "summary",
                    f"PDF {pdf.name} summary/snippet:\n{snippet}",
                    f"pdf,{pdf.name}",
                ),
            )
            summaries += 1
            continue

        ins, skp = _insert_transactions(deps, pdf.name, rows)
        tx_inserted += ins
        tx_skipped += skp

    deps.db.conn.commit()

    if mode == "memories":
        msg = f"Saved {summaries} PDF summaries"
    else:
        msg = f"Inserted {tx_inserted} transactions from PDFs"
        if tx_skipped:
            msg += f" ({tx_skipped} duplicates skipped)"
        if summaries:
            msg += f"; {summaries} non-statement PDFs summarized"
        if skipped:
            msg += f"; {skipped} PDFs skipped (no text)"
    return msg
