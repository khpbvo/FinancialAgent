from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from agents import RunContextWrapper, function_tool

from ..context import RunDeps


def _last_full_month_range() -> tuple[str, str]:
    today = datetime.now()
    first_of_this_month = datetime(today.year, today.month, 1)
    last_of_prev_month = first_of_this_month - timedelta(days=1)
    first_of_prev_month = datetime(last_of_prev_month.year, last_of_prev_month.month, 1)
    return first_of_prev_month.strftime('%Y-%m-%d'), last_of_prev_month.strftime('%Y-%m-%d')


def _is_pos_like(desc: str) -> bool:
    d = desc.lower()
    return any(x in d for x in [
        'betaalautomaat', 'geldmaat', 'albert heijn', ' ah ', 'jumbo', 'coffeeshop', 'restaurant', 'mc donald', 'kfc'
    ])


def _is_variable_bill(desc: str) -> bool:
    dl = desc.lower()
    for k in [
        'vodafone', 'libertel', 'odido', 'kpn', 't-mobile', 'tmobile', 'ziggo',
        'essent', 'energie', 'vandebron', 'nuon', 'eneco', 'waterbedrijf', 'stroom', 'gas',
        'vgz', 'verzekering', 'verzekeraar', 'nn schadeverzekering', 'anwb verzekeren', 'cz', 'fbto',
        'brabantwonen', 'huur', 'hypotheek', 'woning', 'woonverzekering',
        'gemeente', 'belasting', 'heffing', 'waterschap'
    ]:
        if k in dl:
            return True
    return False


@function_tool
def monthly_cost_summary(
    ctx: RunContextWrapper[RunDeps],
    months_back: int = 1,
    last_full_month: bool = True,
    bills_only: bool = False,
    include_breakdown: bool = True
) -> str:
    """Compute monthly costs for the requested period with an optional bills-only filter.

    Args:
        months_back: Number of months lookback (1 = last month). Ignored if last_full_month is True.
        last_full_month: If True, uses the previous calendar month; else uses last 30*months_back days.
        bills_only: Exclude POS-like merchants and fees/cash withdrawals; be lenient on variable bills.
        include_breakdown: Include per-category and top-merchant breakdowns.
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Determine date window
    if last_full_month:
        start_date, end_date = _last_full_month_range()
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=30 * max(months_back, 1))
        start_date, end_date = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

    # Fetch expenses in window (amount < 0)
    cur.execute(
        """SELECT date, description, amount, category
               FROM transactions
               WHERE date >= ? AND date <= ? AND amount < 0
               ORDER BY date ASC""",
        (start_date, end_date)
    )
    rows = cur.fetchall()

    # Apply bills-only filter if requested
    filtered = []
    for r in rows:
        desc = r['description'] or ''
        cat = (r['category'] or '').lower()
        if bills_only:
            # skip fees/cash withdrawals and obvious POS purchases
            if cat in ('fees', 'cash_withdrawal'):
                continue
            if _is_pos_like(desc):
                continue
            # allow variable bills (telecom/energy/insurance/housing/government)
            # everything else passes only if not clearly POS
        filtered.append(r)

    used = filtered if bills_only else rows

    total = sum(abs(r['amount']) for r in used)

    # Build breakdowns
    by_category: Dict[str, float] = {}
    by_merchant: Dict[str, float] = {}
    for r in used:
        cat = r['category'] or 'uncategorized'
        by_category[cat] = by_category.get(cat, 0.0) + abs(r['amount'])
        # Naive merchant: first 30 chars of description
        mname = (r['description'] or '')[:30]
        by_merchant[mname] = by_merchant.get(mname, 0.0) + abs(r['amount'])

    lines: List[str] = []
    title = "Bills-only Monthly Cost" if bills_only else "Monthly Cost"
    lines.append(f"📅 {title}")
    lines.append("=" * 50)
    lines.append(f"Period: {start_date} to {end_date}")
    lines.append(f"Total spent: €{total:.2f}")

    if include_breakdown:
        # Top categories
        lines.append("\n🏷️ By Category (Top 5)")
        for cat, amt in sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = (amt / total * 100) if total > 0 else 0
            lines.append(f"• {cat}: €{amt:.2f} ({pct:.1f}%)")

        # Top merchants (rough)
        lines.append("\n🧾 Top Merchants (Top 5)")
        for m, amt in sorted(by_merchant.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"• {m}: €{amt:.2f}")

    # Hints
    if bills_only:
        lines.append("\n💡 Hint: This excludes fees, cash withdrawals, and POS-like purchases.")
    else:
        lines.append("\n💡 Hint: Use bills-only for subscriptions/utilities/insurance only.")

    return "\n".join(lines)

