from __future__ import annotations
from typing import Optional
from datetime import datetime
from agents import RunContextWrapper, function_tool
from ..context import RunDeps


@function_tool
def create_goal(
    ctx: RunContextWrapper[RunDeps],
    name: str,
    target_amount: float,
    target_date: Optional[str] = None,
    category: str = "savings",
    initial_amount: float = 0,
) -> str:
    """Create a new financial goal.

    Args:
        name: Name of the goal (e.g., "Emergency Fund", "New Car", "Vacation")
        target_amount: Target amount to save in EUR
        target_date: Optional target completion date (YYYY-MM-DD format)
        category: Goal category - "savings", "debt_reduction", or "investment"
        initial_amount: Starting amount already saved towards this goal
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Check if goal with same name exists
    cur.execute("SELECT id FROM goals WHERE name = ? AND status = 'active'", (name,))
    if cur.fetchone():
        return f"⚠️ An active goal with name '{name}' already exists"

    cur.execute(
        """INSERT INTO goals (name, target_amount, current_amount, target_date, category, status)
           VALUES (?, ?, ?, ?, ?, 'active')""",
        (name, target_amount, initial_amount, target_date, category.lower()),
    )
    deps.db.conn.commit()

    if target_date:
        days_left = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.now()).days
        return f"🎯 Created goal '{name}': €{target_amount:.2f} by {target_date} ({days_left} days)"
    else:
        return f"🎯 Created goal '{name}': €{target_amount:.2f}"


@function_tool
def update_goal_progress(
    ctx: RunContextWrapper[RunDeps], name: str, amount: float, action: str = "add"
) -> str:
    """Update progress towards a financial goal.

    Args:
        name: Name of the goal to update
        amount: Amount to add or set in EUR
        action: "add" to add to current amount, "set" to set new amount
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get current goal
    cur.execute("SELECT * FROM goals WHERE name = ? AND status = 'active'", (name,))
    goal = cur.fetchone()

    if not goal:
        return f"No active goal found with name '{name}'"

    if action == "add":
        new_amount = goal["current_amount"] + amount
    else:  # set
        new_amount = amount

    # Check if goal is completed
    status = "completed" if new_amount >= goal["target_amount"] else "active"

    cur.execute(
        """UPDATE goals 
           SET current_amount = ?, status = ?, updated_at = datetime('now')
           WHERE id = ?""",
        (new_amount, status, goal["id"]),
    )
    deps.db.conn.commit()

    progress = (new_amount / goal["target_amount"]) * 100
    remaining = goal["target_amount"] - new_amount

    if status == "completed":
        return f"🎉 GOAL COMPLETED! '{name}' has reached €{goal['target_amount']:.2f}!"
    else:
        return f"✅ Updated '{name}': €{new_amount:.2f} / €{goal['target_amount']:.2f} ({progress:.1f}%)\n   Remaining: €{remaining:.2f}"


@function_tool
def check_goals(
    ctx: RunContextWrapper[RunDeps], include_completed: bool = False
) -> str:
    """Check progress on all financial goals.

    Args:
        include_completed: Whether to include completed goals (default: False)
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    if include_completed:
        cur.execute("SELECT * FROM goals ORDER BY status, created_at DESC")
    else:
        cur.execute(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY created_at DESC"
        )

    goals = cur.fetchall()

    if not goals:
        return "No active goals. Use create_goal to set financial targets!"

    results = ["🎯 Financial Goals Progress\n" + "=" * 50]

    active_goals = []
    completed_goals = []

    for goal in goals:
        if goal["status"] == "active":
            active_goals.append(goal)
        else:
            completed_goals.append(goal)

    # Active goals
    if active_goals:
        results.append("\n📊 ACTIVE GOALS")
        for goal in active_goals:
            progress = (goal["current_amount"] / goal["target_amount"]) * 100
            remaining = goal["target_amount"] - goal["current_amount"]

            # Create visual progress bar
            bar_length = 20
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)

            results.append(f"\n🎯 {goal['name']} ({goal['category']})")
            results.append(
                f"   Progress: €{goal['current_amount']:.2f} / €{goal['target_amount']:.2f}"
            )
            results.append(f"   [{bar}] {progress:.1f}%")
            results.append(f"   Remaining: €{remaining:.2f}")

            if goal["target_date"]:
                days_left = (
                    datetime.strptime(goal["target_date"], "%Y-%m-%d") - datetime.now()
                ).days
                if days_left > 0:
                    daily_needed = remaining / days_left
                    results.append(
                        f"   📅 {days_left} days left (€{daily_needed:.2f}/day needed)"
                    )
                else:
                    results.append("   ⚠️ Target date has passed!")

    # Completed goals
    if completed_goals and include_completed:
        results.append("\n✅ COMPLETED GOALS")
        for goal in completed_goals:
            results.append(f"• {goal['name']}: €{goal['target_amount']:.2f} ✓")

    # Summary statistics
    total_target = sum(g["target_amount"] for g in active_goals)
    total_saved = sum(g["current_amount"] for g in active_goals)

    if active_goals:
        results.append("\n" + "=" * 50)
        results.append("📈 OVERALL PROGRESS")
        overall_progress = (total_saved / total_target * 100) if total_target > 0 else 0
        results.append(
            f"   Total saved: €{total_saved:.2f} / €{total_target:.2f} ({overall_progress:.1f}%)"
        )
        results.append(f"   Still needed: €{total_target - total_saved:.2f}")

    return "\n".join(results)


@function_tool
def suggest_savings_plan(
    ctx: RunContextWrapper[RunDeps],
    goal_name: str,
    monthly_income: Optional[float] = None,
) -> str:
    """Generate a savings plan to reach a financial goal.

    Args:
        goal_name: Name of the goal to create a plan for
        monthly_income: Optional monthly income for percentage calculations
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get the goal
    cur.execute(
        "SELECT * FROM goals WHERE name = ? AND status = 'active'", (goal_name,)
    )
    goal = cur.fetchone()

    if not goal:
        return f"No active goal found with name '{goal_name}'"

    remaining = goal["target_amount"] - goal["current_amount"]

    results = [f"💰 Savings Plan for '{goal_name}'\n" + "=" * 50]
    results.append(f"Target: €{goal['target_amount']:.2f}")
    results.append(f"Current: €{goal['current_amount']:.2f}")
    results.append(f"Needed: €{remaining:.2f}\n")

    # Calculate different scenarios
    scenarios = [
        (3, "3 months"),
        (6, "6 months"),
        (12, "1 year"),
        (24, "2 years"),
        (36, "3 years"),
    ]

    results.append("📅 SAVINGS SCENARIOS")

    for months, label in scenarios:
        monthly_amount = remaining / months
        weekly_amount = monthly_amount / 4.33  # avg weeks per month

        scenario = f"\n• Save in {label}:"
        scenario += f"\n  Monthly: €{monthly_amount:.2f}"
        scenario += f"\n  Weekly: €{weekly_amount:.2f}"
        scenario += f"\n  Daily: €{monthly_amount/30:.2f}"

        if monthly_income:
            percentage = (monthly_amount / monthly_income) * 100
            scenario += f"\n  % of income: {percentage:.1f}%"
            if percentage > 20:
                scenario += " ⚠️ (high)"
            elif percentage < 10:
                scenario += " ✅ (comfortable)"

        results.append(scenario)

    # Check if target date exists
    if goal["target_date"]:
        days_left = (
            datetime.strptime(goal["target_date"], "%Y-%m-%d") - datetime.now()
        ).days
        if days_left > 0:
            required_daily = remaining / days_left
            required_monthly = required_daily * 30

            results.append(f"\n📍 TO MEET TARGET DATE ({goal['target_date']}):")
            results.append(f"   Monthly: €{required_monthly:.2f}")
            results.append(f"   Daily: €{required_daily:.2f}")

            if monthly_income:
                percentage = (required_monthly / monthly_income) * 100
                results.append(f"   % of income: {percentage:.1f}%")

    # Get recent spending to suggest where to cut
    cur.execute(
        """SELECT category, SUM(ABS(amount)) as total
           FROM transactions 
           WHERE amount < 0 
           AND date >= date('now', '-30 days')
           AND category IS NOT NULL
           GROUP BY category
           ORDER BY total DESC
           LIMIT 3"""
    )
    top_spending = cur.fetchall()

    if top_spending:
        results.append("\n💡 TOP SPENDING CATEGORIES (last 30 days):")
        for spend in top_spending:
            results.append(f"   • {spend['category']}: €{spend['total']:.2f}")
        results.append("   Consider reducing these to increase savings!")

    return "\n".join(results)


@function_tool
def complete_goal(ctx: RunContextWrapper[RunDeps], name: str) -> str:
    """Mark a goal as completed.

    Args:
        name: Name of the goal to complete
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute(
        """UPDATE goals 
           SET status = 'completed', updated_at = datetime('now')
           WHERE name = ? AND status = 'active'""",
        (name,),
    )

    if cur.rowcount > 0:
        deps.db.conn.commit()
        return f"🎉 Congratulations! Goal '{name}' marked as completed!"
    else:
        return f"No active goal found with name '{name}'"


@function_tool
def pause_goal(ctx: RunContextWrapper[RunDeps], name: str) -> str:
    """Pause a financial goal.

    Args:
        name: Name of the goal to pause
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute(
        """UPDATE goals 
           SET status = 'paused', updated_at = datetime('now')
           WHERE name = ? AND status = 'active'""",
        (name,),
    )

    if cur.rowcount > 0:
        deps.db.conn.commit()
        return f"⏸️ Goal '{name}' has been paused"
    else:
        return f"No active goal found with name '{name}'"
