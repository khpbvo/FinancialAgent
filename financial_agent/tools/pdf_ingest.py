from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

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
            if reader.metadata.get('/Title'):
                metadata_parts.append(f"Title: {reader.metadata['/Title']}")
            if reader.metadata.get('/Creator'):
                metadata_parts.append(f"Creator: {reader.metadata['/Creator']}")
            if reader.metadata.get('/Subject'):
                metadata_parts.append(f"Subject: {reader.metadata['/Subject']}")
        
        # Try to get some structural information
        metadata_parts.append("PDF Structure: This appears to be an image-based or protected PDF")
        metadata_parts.append("Note: Text extraction not possible - PDF may contain scanned images")
        
        # Look for potential patterns in the filename that might indicate content type
        filename_lower = path.name.lower()
        if any(term in filename_lower for term in ['afschrift', 'statement', 'bank', 'rekening']):
            metadata_parts.append("Document Type: Likely a bank statement based on filename")
        
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
    skipped = 0
    cur = deps.db.conn.cursor()
    for pdf in pdfs:
        try:
            text = extract_text_from_pdf(pdf)
            if not text or not text.strip():
                skipped += 1
                continue
        except Exception as e:
            skipped += 1
            continue
        
        snippet = (text[:2000] + "...") if len(text) > 2000 else text
        cur.execute(INSERT_MEMORY, ("summary", f"PDF {pdf.name} summary/snippet:\n{snippet}", f"pdf,{pdf.name}"))
        inserted += 1
    
    deps.db.conn.commit()
    result_msg = f"Ingested {inserted} PDF snippets from {base.name}"
    if skipped > 0:
        result_msg += f" ({skipped} PDFs skipped due to extraction issues)"
    return result_msg
