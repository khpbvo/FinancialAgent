from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from agents import RunContextWrapper, function_tool
from typing import Any

from ..context import RunDeps
from ..db.sql import INSERT_TRANSACTION


def _parse_date(value: Any) -> str:
    """Parse dates from various formats including Excel dates."""
    if pd.isna(value) or value is None:
        return ""
    
    # If it's already a datetime object (common in Excel)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    
    # If it's a string, try parsing
    value_str = str(value).strip()
    if not value_str:
        return ""
    
    # Try common date formats
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(value_str, fmt).date().isoformat()
        except:
            continue
    
    # Try pandas date parser as fallback
    try:
        parsed = pd.to_datetime(value_str, dayfirst=True)
        return parsed.date().isoformat()
    except:
        return value_str  # Return as-is if all parsing fails


def _parse_amount(value: Any) -> float:
    """Parse amounts from Excel, handling various formats."""
    if pd.isna(value) or value is None:
        return 0.0
    
    # If it's already numeric
    if isinstance(value, (int, float)):
        return float(value)
    
    # Parse string amounts
    s = str(value).strip()
    # Remove currency symbols and thousands separators
    s = s.replace("€", "").replace("$", "").replace("£", "")
    s = s.replace(",", "").replace(" ", "")
    
    # Handle European format (comma as decimal separator)
    if "." in s and s.count(".") == 1:
        # Might be standard decimal
        pass
    elif "," in s:
        # European format: convert comma to dot
        s = s.replace(".", "").replace(",", ".")
    
    try:
        return float(s)
    except:
        return 0.0


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect column mappings based on common patterns."""
    columns = df.columns.tolist()
    columns_lower = [col.lower() for col in columns]
    
    mapping = {}
    
    # Detect date column
    date_patterns = ["date", "datum", "transaction date", "posting date", "boekdatum", "valutadatum"]
    for pattern in date_patterns:
        for i, col in enumerate(columns_lower):
            if pattern in col:
                mapping["date"] = columns[i]
                break
        if "date" in mapping:
            break
    
    # Detect description column
    desc_patterns = ["description", "omschrijving", "naam", "merchant", "payee", "details", "reference"]
    for pattern in desc_patterns:
        for i, col in enumerate(columns_lower):
            if pattern in col and columns[i] not in mapping.values():
                mapping["description"] = columns[i]
                break
        if "description" in mapping:
            break
    
    # Detect amount column
    amount_patterns = ["amount", "bedrag", "debit", "credit", "value", "sum", "total"]
    for pattern in amount_patterns:
        for i, col in enumerate(columns_lower):
            if pattern in col and columns[i] not in mapping.values():
                mapping["amount"] = columns[i]
                break
        if "amount" in mapping:
            break
    
    # Detect category column
    category_patterns = ["category", "categorie", "type", "mutatiesoort", "classification"]
    for pattern in category_patterns:
        for i, col in enumerate(columns_lower):
            if pattern in col and columns[i] not in mapping.values():
                mapping["category"] = columns[i]
                break
        if "category" in mapping:
            break
    
    # Detect currency column
    currency_patterns = ["currency", "valuta", "curr", "ccy"]
    for pattern in currency_patterns:
        for i, col in enumerate(columns_lower):
            if pattern in col and columns[i] not in mapping.values():
                mapping["currency"] = columns[i]
                break
        if "currency" in mapping:
            break
    
    # Special handling for ING format with separate debit/credit columns
    if "af bij" in columns_lower or ("debit" in columns_lower and "credit" in columns_lower):
        mapping["sign_column"] = None
        for i, col in enumerate(columns_lower):
            if "af bij" in col:
                mapping["sign_column"] = columns[i]
                break
    
    return mapping


def process_excel_file(deps: RunDeps, excel_path: Path, sheet_name: Optional[str] = None) -> int:
    """Process Excel file and insert transactions into database.
    
    Returns number of inserted rows.
    """
    # Read Excel file
    try:
        if sheet_name:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            # Read first sheet by default
            df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")
    
    if df.empty:
        return 0
    
    # Auto-detect column mappings
    mapping = detect_column_mapping(df)
    
    if "date" not in mapping or "amount" not in mapping:
        # Try to use first few columns as fallback
        columns = df.columns.tolist()
        if len(columns) >= 3:
            mapping.setdefault("date", columns[0])
            mapping.setdefault("description", columns[1])
            mapping.setdefault("amount", columns[2])
        else:
            raise ValueError("Unable to detect required columns (date, amount) in Excel file")
    
    # Process rows
    cur = deps.db.conn.cursor()
    inserted = 0
    
    for _, row in df.iterrows():
        # Extract basic fields
        date = _parse_date(row.get(mapping.get("date", ""), ""))
        description = str(row.get(mapping.get("description", ""), "")).strip()
        amount = _parse_amount(row.get(mapping.get("amount", ""), 0))
        
        # Handle sign column (ING format)
        if "sign_column" in mapping and mapping["sign_column"]:
            sign = str(row.get(mapping["sign_column"], "")).strip().lower()
            if sign == "af" or sign == "debit":
                amount = -abs(amount)
        
        # Handle separate debit/credit columns
        elif "debit" in mapping and "credit" in mapping:
            debit = _parse_amount(row.get(mapping["debit"], 0))
            credit = _parse_amount(row.get(mapping["credit"], 0))
            amount = credit - debit  # Credits are positive, debits are negative
        
        # Extract optional fields
        currency = None
        if "currency" in mapping:
            currency = str(row.get(mapping["currency"], "EUR")).strip()
        if not currency:
            currency = "EUR"  # Default to EUR
        
        category = None
        if "category" in mapping:
            category = str(row.get(mapping["category"], "")).strip()
            if category == "nan" or not category:
                category = None
        
        # Build description from multiple fields if available
        if not description and len(df.columns) > 1:
            # Try to combine non-numeric columns for description
            desc_parts = []
            for col in df.columns:
                if col not in [mapping.get("date"), mapping.get("amount"), mapping.get("currency")]:
                    val = str(row.get(col, "")).strip()
                    if val and val != "nan":
                        desc_parts.append(val)
            description = " - ".join(desc_parts[:3])  # Limit to 3 parts
        
        # Skip if no valid date or description
        if not date or not description:
            continue
        
        # Insert transaction
        cur.execute(INSERT_TRANSACTION, (date, description, amount, currency, category, str(excel_path.name)))
        inserted += 1
    
    deps.db.conn.commit()
    return inserted


def excel_error_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    """Custom error handler for Excel ingestion failures."""
    error_str = str(error).lower()
    
    if "not found" in error_str:
        return "Could not find the Excel file. Please check the path and try again."
    elif "permission" in error_str:
        return "Permission denied accessing the file. Please check file permissions."
    elif "column" in error_str:
        return f"Excel format issue: {error}. Please ensure the file contains date and amount columns."
    elif "sheet" in error_str:
        return f"Sheet not found: {error}. Please check the sheet name or leave blank for default."
    else:
        return f"Failed to ingest Excel file: {error}. Please verify the file format is correct."


@function_tool(failure_error_function=excel_error_handler)
def ingest_excel(ctx: RunContextWrapper[RunDeps], path: str, sheet_name: Optional[str] = None) -> str:
    """Ingest an Excel file (xls/xlsx) of transactions into the database.
    
    Args:
        path: Path to the Excel file (absolute or relative to documents folder)
        sheet_name: Optional name of the sheet to read (defaults to first sheet)
    
    Returns:
        Status message with number of transactions ingested
        
    Notes:
        - Auto-detects common column names for date, description, amount, category
        - Supports ING bank format with 'Af Bij' column
        - Handles European number formats (comma as decimal separator)
        - Defaults to EUR currency if not specified
    """
    deps = ctx.context
    excel_path = Path(path)
    
    # Make path absolute if relative
    if not excel_path.is_absolute():
        excel_path = deps.config.documents_dir / excel_path
    
    # Check file exists and has correct extension
    if not excel_path.exists():
        return f"Excel file not found: {excel_path}"
    
    if excel_path.suffix.lower() not in [".xls", ".xlsx"]:
        return f"File is not an Excel file (must be .xls or .xlsx): {excel_path.name}"
    
    try:
        inserted = process_excel_file(deps, excel_path, sheet_name)
        return f"Successfully ingested {inserted} transactions from {excel_path.name}"
    except Exception as e:
        raise ValueError(f"Failed to process Excel file: {e}")


@function_tool
def list_excel_sheets(ctx: RunContextWrapper[RunDeps], path: str) -> str:
    """List all sheet names in an Excel file.
    
    Args:
        path: Path to the Excel file
        
    Returns:
        List of sheet names in the file
    """
    deps = ctx.context
    excel_path = Path(path)
    
    if not excel_path.is_absolute():
        excel_path = deps.config.documents_dir / excel_path
    
    if not excel_path.exists():
        return f"Excel file not found: {excel_path}"
    
    try:
        # Read all sheet names
        excel_file = pd.ExcelFile(excel_path)
        sheets = excel_file.sheet_names
        
        if not sheets:
            return "No sheets found in Excel file"
        
        result = [f"Sheets in {excel_path.name}:"]
        for i, sheet in enumerate(sheets, 1):
            # Try to get row count for each sheet
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet, nrows=0)
                cols = len(df.columns)
                df_full = pd.read_excel(excel_path, sheet_name=sheet)
                rows = len(df_full)
                result.append(f"{i}. '{sheet}' - {rows} rows, {cols} columns")
            except:
                result.append(f"{i}. '{sheet}'")
        
        return "\n".join(result)
    except Exception as e:
        return f"Failed to read Excel file: {e}"