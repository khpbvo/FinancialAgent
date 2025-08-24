from __future__ import annotations

from agents import RunContextWrapper, function_tool

from ..context import RunDeps
from ..db.sql import INSERT_TRANSACTION


@function_tool
def add_transaction(
    ctx: RunContextWrapper[RunDeps],
    date: str,
    description: str,
    amount: float,
    currency: str = "EUR",
    category: str | None = None,
    source_file: str | None = None,
) -> str:
    """Add a single transaction into the database (manual record)."""
    deps = ctx.context
    cur = deps.db.conn.cursor()
    cur.execute(INSERT_TRANSACTION, (date, description, amount, currency, category, source_file or "manual"))
    deps.db.conn.commit()
    return "Transaction added"
