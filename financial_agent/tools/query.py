from __future__ import annotations
from typing import Any

from agents import RunContextWrapper, function_tool

from ..context import RunDeps
from ..db.sql import LIST_RECENT_TRANSACTIONS, SEARCH_TRANSACTIONS


@function_tool
def list_recent_transactions(ctx: RunContextWrapper[RunDeps], limit: int = 20) -> str:
    """List the most recent transactions from memory.

    Args:
        limit: Max number of rows to return.
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    cur.execute(LIST_RECENT_TRANSACTIONS, (limit,))
    rows = cur.fetchall()
    lines = [
        f"{r['date']} | {r['description']} | {r['amount']:.2f} {r['currency']} | {r['category'] or ''} | {r['source_file']}"
        for r in rows
    ]
    return "\n".join(lines) if lines else "No transactions found."


@function_tool
def search_transactions(
    ctx: RunContextWrapper[RunDeps],
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    text: str | None = None,
    limit: int = 100,
) -> str:
    """Search transactions by date range, category and/or fuzzy text match.

    Args:
        start_date: Inclusive lower bound YYYY-MM-DD
        end_date: Inclusive upper bound YYYY-MM-DD
        category: Exact category filter
        text: Substring to match in description (case-sensitive)
        limit: Max rows
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    cur.execute(
        SEARCH_TRANSACTIONS,
        (
            start_date, start_date,
            end_date, end_date,
            category, category,
            text, text,
            limit,
        ),
    )
    rows = cur.fetchall()
    if not rows:
        return "No matching transactions."
    lines = [
        f"{r['date']} | {r['description']} | {r['amount']:.2f} {r['currency']} | {r['category'] or ''} | {r['source_file']}"
        for r in rows
    ]
    return "\n".join(lines)
