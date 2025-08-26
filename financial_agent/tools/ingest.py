from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import csv

from agents import RunContextWrapper, function_tool
from typing import Any

from ..context import RunDeps
from ..db.sql import INSERT_TRANSACTION


@dataclass
class CSVMap:
    date_col: str
    description_col: str
    amount_col: str
    currency_col: str | None = None
    category_col: str | None = None
    date_format: str | None = None  # e.g. "%d-%m-%Y"


def _parse_date(value: str, fmt: str | None) -> str:
    if not value:
        return ""
    if fmt:
        return datetime.strptime(value, fmt).date().isoformat()
    # Try a few common formats
    for f in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, f).date().isoformat()
        except Exception:
            continue
    return value  # fallback raw


def _parse_amount(value: str) -> float:
    """Parse amounts with comma decimal and optional thousands separators."""
    if value is None:
        return 0.0
    s = str(value).strip()
    # Remove thousands dots and convert comma to dot
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def process_csv_file(deps: RunDeps, csv_path: Path) -> int:
    """Programmatic CSV ingestion with heuristics for ING exports.

    Returns number of inserted rows.
    """
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        cur = deps.db.conn.cursor()

        # Detect ING NL export
        is_ing = {"Datum", "Naam / Omschrijving", "Bedrag (EUR)", "Af Bij"}.issubset(set(headers))

        for row in reader:
            if is_ing:
                # ING specifics
                raw_date = row.get("Datum", "")
                date = _parse_date(raw_date, "%Y%m%d")
                name = row.get("Naam / Omschrijving", "").strip()
                notes = row.get("Mededelingen", "").strip()
                desc = f"{name} — {notes}" if notes else name
                amount = _parse_amount(row.get("Bedrag (EUR)", "0"))
                sign = (row.get("Af Bij") or "").strip()
                if sign.lower() == "af":
                    amount = -abs(amount)
                currency = "EUR"
                category = row.get("Mutatiesoort") or None
            else:
                # Fallback to mapping-like defaults
                date = _parse_date(row.get("date", ""), None)
                desc = row.get("description", "")
                amount = _parse_amount(row.get("amount", "0"))
                currency = row.get("currency")
                category = row.get("category")

            cur.execute(INSERT_TRANSACTION, (date, desc, amount, currency, category, str(csv_path.name)))
            inserted += 1
    deps.db.conn.commit()
    return inserted


def ingest_csv_file(deps: RunDeps, csv_path: Path) -> str:
    inserted = process_csv_file(deps, csv_path)
    return f"Ingested {inserted} transactions from {csv_path.name}"


def csv_error_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    """Custom error handler for CSV ingestion failures."""
    if "not found" in str(error).lower():
        return f"Could not find the CSV file. Please check the path and try again."
    elif "permission" in str(error).lower():
        return f"Permission denied accessing the file. Please check file permissions."
    else:
        return f"Failed to ingest CSV: {str(error)}. Please verify the file format is correct."

@function_tool(failure_error_function=csv_error_handler)
def ingest_csv(ctx: RunContextWrapper[RunDeps], path: str) -> str:
    """Ingest a CSV file of transactions into the database.

    Args:
        path: Absolute or relative path to CSV.
    Note: The tool auto-detects ING CSVs; otherwise it expects columns: date, description, amount, [currency], [category]
    """
    deps = ctx.context
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = deps.config.documents_dir / csv_path
    if not csv_path.exists():
        return f"CSV not found: {csv_path}"
    return ingest_csv_file(deps, csv_path)
