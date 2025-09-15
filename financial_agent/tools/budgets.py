from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta
from agents import RunContextWrapper, function_tool
from ..context import RunDeps


@function_tool
def set_budget(
    ctx: RunContextWrapper[RunDeps],
    category: str,
    amount: float,
    period: str = "monthly",
) -> str:
    """Set or update a budget for a specific category.

    Args:
        category: The spending category (e.g., "groceries", "entertainment", "transport")
        amount: The budget amount in EUR
        period: Budget period - "monthly", "weekly", or "yearly" (default: monthly)
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Check if budget exists for this category
    cur.execute("SELECT id FROM budgets WHERE category = ?", (category.lower(),))
    existing = cur.fetchone()

    if existing:
        # Update existing budget
        cur.execute(
            """UPDATE budgets 
               SET amount = ?, period = ?, updated_at = datetime('now')
               WHERE category = ?""",
            (amount, period, category.lower()),
        )
        deps.db.conn.commit()
        return f"✅ Updated {period} budget for '{category}' to €{amount:.2f}"
    else:
        # Create new budget
        cur.execute(
            """INSERT INTO budgets (category, amount, period) 
               VALUES (?, ?, ?)""",
            (category.lower(), amount, period),
        )
        deps.db.conn.commit()
        return f"✅ Created {period} budget for '{category}': €{amount:.2f}"


@function_tool
def check_budget(
    ctx: RunContextWrapper[RunDeps],
    category: Optional[str] = None,
    period_days: int = 30,
) -> str:
    """Check spending against budgets and show remaining amounts.

    Args:
        category: Optional specific category to check (if None, checks all)
        period_days: Number of days to look back for spending (default: 30)
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    # Get budgets
    if category:
        cur.execute("SELECT * FROM budgets WHERE category = ?", (category.lower(),))
    else:
        cur.execute("SELECT * FROM budgets ORDER BY category")

    budgets = cur.fetchall()

    if not budgets:
        return "No budgets set yet. Use set_budget to create one."

    results = []
    results.append(f"📊 Budget Status (last {period_days} days)\n")
    results.append("=" * 50)

    total_budget = 0
    total_spent = 0

    for budget in budgets:
        # Get spending for this category
        cur.execute(
            """SELECT COALESCE(SUM(ABS(amount)), 0) as spent
               FROM transactions 
               WHERE category = ? 
               AND amount < 0
               AND date >= ?
               AND date <= ?""",
            (
                budget["category"],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            ),
        )
        spent = cur.fetchone()["spent"]

        # Adjust budget amount based on period
        if budget["period"] == "weekly":
            adjusted_budget = budget["amount"] * (period_days / 7)
        elif budget["period"] == "yearly":
            adjusted_budget = budget["amount"] * (period_days / 365)
        else:  # monthly
            adjusted_budget = budget["amount"] * (period_days / 30)

        total_budget += adjusted_budget
        total_spent += spent

        remaining = adjusted_budget - spent
        percentage = (spent / adjusted_budget * 100) if adjusted_budget > 0 else 0

        # Create visual progress bar
        bar_length = 20
        filled = int(bar_length * percentage / 100)
        bar = "█" * min(filled, bar_length) + "░" * max(0, bar_length - filled)

        # Determine status emoji
        if percentage > 100:
            status = "🔴 OVER"
        elif percentage > 90:
            status = "🟡 WARNING"
        else:
            status = "🟢 OK"

        results.append(f"\n📁 {budget['category'].upper()}")
        results.append(f"   Budget: €{adjusted_budget:.2f} ({budget['period']})")
        results.append(f"   Spent:  €{spent:.2f}")
        results.append(f"   Left:   €{remaining:.2f}")
        results.append(f"   [{bar}] {percentage:.1f}% {status}")

    # Overall summary
    results.append("\n" + "=" * 50)
    results.append("📈 TOTAL SUMMARY")
    total_remaining = total_budget - total_spent
    total_percentage = (total_spent / total_budget * 100) if total_budget > 0 else 0
    results.append(f"   Total Budget: €{total_budget:.2f}")
    results.append(f"   Total Spent:  €{total_spent:.2f}")
    results.append(
        f"   Total Left:   €{total_remaining:.2f} ({100-total_percentage:.1f}% remaining)"
    )

    return "\n".join(results)


@function_tool
def list_budgets(ctx: RunContextWrapper[RunDeps]) -> str:
    """List all current budgets."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute("SELECT * FROM budgets ORDER BY category")
    budgets = cur.fetchall()

    if not budgets:
        return "No budgets set yet. Use set_budget to create budgets for your spending categories."

    results = ["📊 Current Budgets\n" + "=" * 40]

    for budget in budgets:
        results.append(
            f"• {budget['category']}: €{budget['amount']:.2f} ({budget['period']})"
        )

    return "\n".join(results)


@function_tool
def suggest_budgets(ctx: RunContextWrapper[RunDeps], months_back: int = 3) -> str:
    """Analyze spending patterns and suggest budgets for each category.

    Args:
        months_back: Number of months to analyze for patterns (default: 3)
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)

    # Get spending by category
    cur.execute(
        """SELECT category, 
                  COUNT(*) as transaction_count,
                  AVG(ABS(amount)) as avg_amount,
                  SUM(ABS(amount)) as total_spent,
                  MIN(ABS(amount)) as min_spent,
                  MAX(ABS(amount)) as max_spent
           FROM transactions 
           WHERE amount < 0 
           AND date >= ?
           AND category IS NOT NULL
           GROUP BY category
           ORDER BY total_spent DESC""",
        (start_date.strftime("%Y-%m-%d"),),
    )

    spending_data = cur.fetchall()

    if not spending_data:
        return "Not enough transaction data to suggest budgets. Import more transactions first."

    results = [
        f"💡 Budget Suggestions (based on {months_back} months of data)\n" + "=" * 50
    ]

    for data in spending_data:
        monthly_avg = data["total_spent"] / months_back

        # Suggest budget with 10% buffer
        suggested_budget = monthly_avg * 1.1

        # Check if budget already exists
        cur.execute(
            "SELECT amount FROM budgets WHERE category = ?", (data["category"],)
        )
        existing = cur.fetchone()

        results.append(f"\n📁 {data['category'].upper()}")
        results.append(f"   Average monthly: €{monthly_avg:.2f}")
        results.append(f"   Range: €{data['min_spent']:.2f} - €{data['max_spent']:.2f}")
        results.append(f"   Transactions: {data['transaction_count']}")
        results.append(f"   💡 Suggested budget: €{suggested_budget:.2f}/month")

        if existing:
            diff = suggested_budget - existing["amount"]
            if diff > 0:
                results.append(
                    f"   ⚠️ Current budget (€{existing['amount']:.2f}) may be too low"
                )
            else:
                results.append(
                    f"   ✅ Current budget (€{existing['amount']:.2f}) looks good"
                )

    return "\n".join(results)


@function_tool
def delete_budget(ctx: RunContextWrapper[RunDeps], category: str) -> str:
    """Delete a budget for a specific category.

    Args:
        category: The category to remove the budget for
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute("DELETE FROM budgets WHERE category = ?", (category.lower(),))
    affected = cur.rowcount
    deps.db.conn.commit()

    if affected > 0:
        return f"✅ Deleted budget for '{category}'"
    else:
        return f"No budget found for '{category}'"
