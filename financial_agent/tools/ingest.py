from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import csv

from agents import RunContextWrapper, function_tool

from ..context import RunDeps
from ..db.sql import INSERT_TRANSACTION


@dataclass
class CSVMap:
    """Mapping configuration for custom CSV column names.

    Parameters
    ----------
    date_col:
        Name of the column containing the transaction date.
    description_col:
        Column holding the transaction description or counter-party.
    amount_col:
        Column with the numeric amount. Values are parsed using
        :func:`_parse_amount`.
    currency_col:
        Optional column with the currency code.
    category_col:
        Optional column with a category or type description.
    date_format:
        Optional ``datetime.strptime`` format string used to parse ``date_col``.
    sign_col:
        Optional column indicating debit/credit. When provided, ``debit_values``
        and ``credit_values`` control the amount sign.
    debit_values / credit_values:
        Iterables of strings that mark a row as a debit or credit respectively.

    Examples
    --------
    >>> CSVMap(
    ...     date_col="Datum",
    ...     description_col="Omschrijving",
    ...     amount_col="Bedrag",
    ...     sign_col="Af Bij",
    ...     debit_values=["Af"],
    ...     credit_values=["Bij"],
    ...     date_format="%Y%m%d",
    ... )
    """

    date_col: str
    description_col: str
    amount_col: str
    currency_col: str | None = None
    category_col: str | None = None
    date_format: str | None = None  # e.g. "%d-%m-%Y"
    sign_col: str | None = None
    debit_values: Iterable[str] | None = None
    credit_values: Iterable[str] | None = None


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


def process_csv_file(deps: RunDeps, csv_path: Path, csv_map: CSVMap | None = None) -> int:
    """Programmatic CSV ingestion with optional column mapping.

    By supplying a :class:`CSVMap` you can describe the layout of your bank's
    export. The mapping lets you pick column names, specify the date format and
    even indicate which column contains debit/credit information.

    Args:
        deps: Application dependencies
        csv_path: Path to the CSV file
        csv_map: Optional mapping for custom CSV formats. If omitted, ING
            exports are auto-detected and default column names are used.

    Example
    -------
    >>> csv_map = CSVMap(
    ...     date_col="Transaction Date",
    ...     description_col="Details",
    ...     amount_col="Amount",
    ...     sign_col="Type",
    ...     debit_values=["Debit"],
    ...     credit_values=["Credit"],
    ...     date_format="%d/%m/%Y",
    ... )
    >>> process_csv_file(deps, Path("bank.csv"), csv_map)

    Returns:
        Number of inserted rows.
    """
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        cur = deps.db.conn.cursor()

        # Detect ING NL export
        is_ing = (
            csv_map is None
            and {"Datum", "Naam / Omschrijving", "Bedrag (EUR)", "Af Bij"}.issubset(set(headers))
        )

        for row in reader:
            if csv_map:
                date = _parse_date(row.get(csv_map.date_col, ""), csv_map.date_format)
                desc = row.get(csv_map.description_col, "")
                amount = _parse_amount(row.get(csv_map.amount_col, "0"))
                if csv_map.sign_col:
                    sign = (row.get(csv_map.sign_col) or "").lower()
                    debit = {v.lower() for v in csv_map.debit_values or []}
                    credit = {v.lower() for v in csv_map.credit_values or []}
                    if sign in debit:
                        amount = -abs(amount)
                    elif sign in credit:
                        amount = abs(amount)
                currency = row.get(csv_map.currency_col) if csv_map.currency_col else None
                category = row.get(csv_map.category_col) if csv_map.category_col else None
            elif is_ing:
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

            cur.execute(
                INSERT_TRANSACTION,
                (date, desc, amount, currency, category, str(csv_path.name)),
            )
            inserted += 1
    deps.db.conn.commit()
    return inserted


def ingest_csv_file(deps: RunDeps, csv_path: Path, csv_map: CSVMap | None = None) -> str:
    inserted = process_csv_file(deps, csv_path, csv_map)
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
def ingest_csv(
    ctx: RunContextWrapper[RunDeps],
    path: str,
    csv_map: CSVMap | None = None,
) -> str:
    """Ingest a CSV file of transactions into the database.

    Args:
        path: Absolute or relative path to CSV.
        csv_map: Optional mapping configuration for custom bank exports.

    Example:
        >>> csv_map = CSVMap(
        ...     date_col="Datum",
        ...     description_col="Omschrijving",
        ...     amount_col="Bedrag",
        ...     sign_col="Af Bij",
        ...     debit_values=["Af"],
        ...     credit_values=["Bij"],
        ...     date_format="%d-%m-%Y",
        ... )
        >>> ingest_csv(ctx, "mybank.csv", csv_map)

    Note: If no mapping is provided, ING CSVs are auto-detected; otherwise the
    CSV must contain columns: date, description, amount, [currency], [category].
    """
    deps = ctx.context
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = deps.config.documents_dir / csv_path
    if not csv_path.exists():
        return f"CSV not found: {csv_path}"
    return ingest_csv_file(deps, csv_path, csv_map)
