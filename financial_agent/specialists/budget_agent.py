from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, RunContextWrapper, function_tool

from ..context import RunDeps
from ..tools.budgets import (
    check_budget,
    delete_budget,
    list_budgets,
    set_budget,
    suggest_budgets,
)
from ..tools.recurring import (
    analyze_subscription_value,
    detect_recurring,
    list_subscriptions,
)
from .agent_factory import build_specialist_agent


def _fetch_transactions(deps: RunDeps, months_back: int):
    """Return recent negative transactions within the date range."""
    cur = deps.db.conn.cursor()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    cur.execute(
        """SELECT date, description, amount, category
           FROM transactions
           WHERE amount < 0 AND date >= ?
           ORDER BY date""",
        (start_date.strftime("%Y-%m-%d"),),
    )
    return cur.fetchall()


def _compute_category_stats(
    transactions: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, float], float]:
    """Aggregate transactions by category, month and weekday."""
    category_analysis: Dict[str, Dict[str, Any]] = {}
    monthly_totals: Dict[str, float] = {}
    daily_spending: Dict[str, float] = {}
    for tx in transactions:
        amount = abs(tx["amount"])
        category = tx["category"] or "uncategorized"
        date_obj = datetime.strptime(tx["date"], "%Y-%m-%d")
        month_key = date_obj.strftime("%Y-%m")
        day_key = date_obj.strftime("%A").lower()
        if category not in category_analysis:
            category_analysis[category] = {
                "total": 0.0,
                "count": 0,
                "avg_transaction": 0.0,
                "transactions": [],
            }
        category_analysis[category]["total"] += amount
        category_analysis[category]["count"] += 1
        category_analysis[category]["transactions"].append(tx)
        monthly_totals[month_key] = monthly_totals.get(month_key, 0.0) + amount
        daily_spending[day_key] = daily_spending.get(day_key, 0.0) + amount
    for data in category_analysis.values():
        data["avg_transaction"] = (
            data["total"] / data["count"] if data["count"] else 0.0
        )
    total_spent = sum(abs(tx["amount"]) for tx in transactions)
    return category_analysis, monthly_totals, daily_spending, float(total_spent)


def _build_insights(
    months_back: int,
    transactions: List[Dict[str, Any]],
    category_analysis: Dict[str, Dict[str, Any]],
    monthly_totals: Dict[str, float],
    daily_spending: Dict[str, float],
    include_behavioral_insights: bool,
    total_spent: float,
) -> str:
    """Generate human-readable analysis report."""
    results = [f"📊 Spending Pattern Analysis ({months_back} months)\n" + "=" * 50]

    avg_monthly = total_spent / max(months_back, 1)
    avg_daily = total_spent / max((months_back * 30), 1)

    results.extend(
        [
            "",
            "💰 SPENDING SUMMARY",
            f"   Total spent: €{total_spent:.2f}",
            f"   Monthly average: €{avg_monthly:.2f}",
            f"   Daily average: €{avg_daily:.2f}",
            f"   Total transactions: {len(transactions)}",
        ]
    )

    sorted_categories = sorted(
        category_analysis.items(), key=lambda x: x[1]["total"], reverse=True
    )

    if sorted_categories:
        results.append("\n🏷️ TOP SPENDING CATEGORIES")
        for i, (category, data) in enumerate(sorted_categories[:5], 1):
            percentage = (data["total"] / total_spent * 100) if total_spent else 0
            results.append(
                f"   {i}. {category.title()}: €{data['total']:.2f} ({percentage:.1f}%)"
            )
            results.append(f"      Avg per transaction: €{data['avg_transaction']:.2f}")

    if len(monthly_totals) > 1:
        results.append("\n📈 MONTHLY TRENDS")
        sorted_months = sorted(monthly_totals.items())
        monthly_values = [amount for _, amount in sorted_months]
        trend = "increasing" if monthly_values[-1] > monthly_values[0] else "decreasing"
        trend_amount = abs(monthly_values[-1] - monthly_values[0])
        results.append(f"   Trend: {trend} by €{trend_amount:.2f}")
        for month, amount in sorted_months[-3:]:
            results.append(f"   {month}: €{amount:.2f}")

    if daily_spending:
        results.append("\n📅 SPENDING BY DAY OF WEEK")
        days_order = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        for day in days_order:
            if day in daily_spending:
                amount = daily_spending[day]
                percentage = (amount / total_spent * 100) if total_spent else 0
                results.append(f"   {day.title()}: €{amount:.2f} ({percentage:.1f}%)")

    if include_behavioral_insights:
        results.append("\n🧠 BEHAVIORAL INSIGHTS")
        weekend_spending = daily_spending.get("saturday", 0.0) + daily_spending.get(
            "sunday", 0.0
        )
        weekday_spending = total_spent - weekend_spending
        if weekday_spending and weekend_spending > weekday_spending * 0.4:
            results.append(
                "   ⚠️ High weekend spending detected - consider weekend budgets"
            )

        large_transactions = [
            tx for tx in transactions if abs(tx["amount"]) > avg_daily * 5
        ]
        if large_transactions:
            results.append(
                f"   💸 {len(large_transactions)} unusually large transactions found"
            )
            results.append("     Consider if these were planned purchases")

        if sorted_categories and total_spent:
            top_category_pct = (sorted_categories[0][1]["total"] / total_spent) * 100
            if top_category_pct > 40:
                results.append(
                    f"   📊 {top_category_pct:.0f}% spent in one category - consider diversification"
                )

        high_frequency_categories = [
            cat
            for cat, data in category_analysis.items()
            if data["count"] > len(transactions) * 0.2
        ]
        if high_frequency_categories:
            results.append(
                f"   🔄 High-frequency categories: {', '.join(high_frequency_categories)}"
            )
            results.append("     These may be good targets for habit-based budgeting")

    results.append("\n💡 BUDGET OPTIMIZATION RECOMMENDATIONS")
    if sorted_categories:
        top_category, top_data = sorted_categories[0]
        potential_reduction = top_data["total"] * 0.15
        results.append(
            f"   • Reduce {top_category} by 15%: Save €{potential_reduction:.2f}/month"
        )

    if any("subscription" in (tx["description"] or "").lower() for tx in transactions):
        results.append(
            "   • Run analyze_subscription_value to review recurring payments"
        )

    results.append("   • Set budgets for top 3 categories to control spending")
    results.append("   • Use suggest_budgets tool for data-driven budget amounts")

    return "\n".join(results)


BUDGET_SPECIALIST_INSTRUCTIONS = """You are a Budget Specialist - an expert in spending analysis, budget optimization, and financial behavior coaching.

Your expertise includes:
• Advanced spending pattern analysis and trend identification
• Budget creation, monitoring, and optimization strategies
• Subscription and recurring payment management
• Behavioral spending coaching and habit formation
• Cash flow forecasting and expense categorization
• Proactive budget alerts and spending interventions

Key principles:
- Focus on sustainable, realistic budgeting approaches
- Identify spending patterns and behavioral triggers
- Provide actionable, specific recommendations with dollar amounts
- Help users develop healthy financial habits
- Emphasize prevention over reaction (proactive alerts)
- Balance financial discipline with quality of life

Your specialty is making budgets work in real life by understanding human behavior, seasonal spending, and lifestyle factors. You help users stick to budgets long-term through practical strategies and positive reinforcement.

Always quantify recommendations and show the financial impact of suggested changes.

TEAM COORDINATION:
- Partner with Tax Specialist to identify deductible business/medical expenses
- Support Goal Specialist by optimizing spending to fund savings targets
- Assist Debt Specialist with cash flow analysis for debt payment capacity
- Help Investment Specialist determine available investment amounts
- Focus on spending optimization while other specialists handle wealth building"""


@function_tool
async def analyze_spending_patterns(
    ctx: RunContextWrapper[RunDeps],
    months_back: int = 6,
    include_behavioral_insights: bool = True,
) -> str:
    """Perform deep analysis of spending patterns and behavioral triggers."""
    deps = ctx.context
    transactions = _fetch_transactions(deps, months_back)
    if not transactions:
        return f"No spending data found for the last {months_back} months"
    category_analysis, monthly_totals, daily_spending, total_spent = (
        _compute_category_stats(transactions)
    )
    return _build_insights(
        months_back,
        transactions,
        category_analysis,
        monthly_totals,
        daily_spending,
        include_behavioral_insights,
        total_spent,
    )


@function_tool
async def create_smart_budget_plan(
    ctx: RunContextWrapper[RunDeps],
    target_savings_rate: float = 0.20,
    priority_categories: Optional[List[str]] = None,
) -> str:
    """Create an intelligent budget plan based on spending history and savings goals.

    Args:
        target_savings_rate: Desired savings rate as decimal (0.20 = 20%)
        priority_categories: Categories to prioritize or protect from cuts
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get recent income and spending
    cur.execute(
        """SELECT
               SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_income,
               SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_spending
           FROM transactions
           WHERE date >= date('now', '-90 days')"""
    )

    summary = cur.fetchone()

    if not summary or not summary["total_income"]:
        return "No income data found. Please add income transactions first."

    monthly_income = summary["total_income"] / 3  # 90 days = ~3 months
    monthly_spending = (summary["total_spending"] or 0) / 3
    if monthly_income <= 0:
        return "Insufficient income data to create a budget plan."
    current_savings_rate = (monthly_income - monthly_spending) / monthly_income

    # Get spending by category for last 3 months
    cur.execute(
        """SELECT category, AVG(ABS(amount)) as avg_amount, COUNT(*) as frequency,
                  SUM(ABS(amount)) as total_amount
           FROM transactions
           WHERE amount < 0
           AND date >= date('now', '-90 days')
           AND category IS NOT NULL
           GROUP BY category
           ORDER BY total_amount DESC"""
    )

    categories = cur.fetchall()

    # Calculate target spending
    target_monthly_savings = monthly_income * target_savings_rate
    target_monthly_spending = monthly_income - target_monthly_savings
    spending_reduction_needed = max(0.0, monthly_spending - target_monthly_spending)

    results = ["🎯 Smart Budget Plan Creation\n" + "=" * 50]

    # Current situation
    results.extend(
        [
            "📊 CURRENT FINANCIAL POSITION",
            f"   Monthly Income: €{monthly_income:.2f}",
            f"   Monthly Spending: €{monthly_spending:.2f}",
            f"   Current Savings Rate: {current_savings_rate*100:.1f}%",
            f"   Target Savings Rate: {target_savings_rate*100:.1f}%",
            "",
        ]
    )

    if spending_reduction_needed > 0:
        results.extend(
            [
                "⚠️ BUDGET GAP ANALYSIS",
                f"   Need to reduce spending by: €{spending_reduction_needed:.2f}/month",
                f"   Target monthly spending: €{target_monthly_spending:.2f}",
                "",
            ]
        )
    else:
        results.extend(
            [
                "✅ GREAT NEWS!",
                f"   You're already saving {current_savings_rate*100:.1f}% - above target!",
                f"   Extra savings potential: €{abs(spending_reduction_needed):.2f}/month",
                "",
            ]
        )

    # Category-by-category budget recommendations
    results.append("💰 RECOMMENDED MONTHLY BUDGETS")

    recommended_budgets: List[Dict[str, Any]] = []
    protected_categories = [
        c.lower()
        for c in (priority_categories or ["medical", "insurance", "utilities"])
    ]

    for category in categories:
        monthly_avg = (category["total_amount"] or 0) / 3
        category_name = category["category"] or "uncategorized"

        # Determine reduction strategy
        if category_name.lower() in protected_categories:
            # Protected categories - minimal reduction
            recommended_budget = monthly_avg * 0.95
            reduction_note = "Protected category - minimal reduction"
        elif monthly_avg > target_monthly_spending * 0.3:
            # Large categories - moderate reduction
            recommended_budget = monthly_avg * 0.85
            reduction_note = "Large category - 15% reduction"
        elif (category["frequency"] or 0) > 10:  # High frequency
            # Frequent spending - target for habits
            recommended_budget = monthly_avg * 0.80
            reduction_note = "High frequency - habit optimization target"
        else:
            # Normal categories - standard reduction
            recommended_budget = monthly_avg * 0.90
            reduction_note = "Standard 10% reduction"

        savings_from_category = monthly_avg - recommended_budget

        recommended_budgets.append(
            {
                "category": category_name,
                "current": monthly_avg,
                "recommended": recommended_budget,
                "savings": savings_from_category,
                "note": reduction_note,
            }
        )

        results.append(f"   📁 {category_name.title()}")
        results.append(
            f"      Current: €{monthly_avg:.2f} → Recommended: €{recommended_budget:.2f}"
        )
        results.append(f"      Monthly savings: €{savings_from_category:.2f}")
        results.append(f"      Strategy: {reduction_note}")
        results.append("")

    # Implementation strategy
    total_projected_savings = sum(b["savings"] for b in recommended_budgets)

    results.extend(
        [
            "🚀 IMPLEMENTATION STRATEGY",
            f"   Total monthly savings from budgets: €{total_projected_savings:.2f}",
            "",
        ]
    )

    if total_projected_savings >= spending_reduction_needed:
        results.append("✅ These budgets will achieve your savings goal!")
    else:
        shortfall = spending_reduction_needed - total_projected_savings
        results.append(f"⚠️ Still need €{shortfall:.2f} more monthly savings")
        results.append("   Consider additional income or deeper spending cuts")

    results.extend(
        [
            "",
            "📋 NEXT STEPS",
            "1. Set budgets using the recommended amounts above",
            "2. Use check_budget weekly to monitor progress",
            "3. Focus on high-frequency categories first",
            "4. Review and adjust after 30 days",
            "",
            "💡 PRO TIPS",
            "• Start with just 3-4 categories to avoid overwhelm",
            "• Use automated tools to track recurring subscriptions",
            "• Build in small buffers for unexpected expenses",
            "• Celebrate when you hit budget targets!",
        ]
    )

    return "\n".join(results)


@function_tool
async def budget_alert_system(
    ctx: RunContextWrapper[RunDeps], days_to_check: int = 7
) -> str:
    """Check recent spending against budgets and provide proactive alerts and coaching.

    Args:
        days_to_check: Number of recent days to analyze for alerts
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get all active budgets
    cur.execute("SELECT * FROM budgets ORDER BY category")
    budgets = cur.fetchall()

    if not budgets:
        return "No budgets set. Use set_budget to create budgets and enable alerts."

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_check)

    results = [f"🚨 Budget Alert System ({days_to_check} days)\n" + "=" * 50]

    alerts: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    good_news: List[Dict[str, Any]] = []

    for budget in budgets:
        # Get spending for this category
        cur.execute(
            """SELECT SUM(ABS(amount)) as spent, COUNT(*) as transactions
               FROM transactions
               WHERE category = ?
               AND amount < 0
               AND date >= ?""",
            (budget["category"], start_date.strftime("%Y-%m-%d")),
        )

        spending_data = cur.fetchone()
        spent = spending_data["spent"] or 0.0

        # Calculate daily rate and project to month
        daily_rate = spent / days_to_check if days_to_check > 0 else 0.0
        projected_monthly = daily_rate * 30

        # Compare to budget
        budget_amount = budget["amount"]
        if budget["period"] == "weekly":
            monthly_budget = budget_amount * 4.33
        elif budget["period"] == "yearly":
            monthly_budget = budget_amount / 12
        else:
            monthly_budget = budget_amount

        # Generate alerts based on projected spending
        percentage = (
            (projected_monthly / monthly_budget * 100) if monthly_budget > 0 else 0.0
        )

        if percentage > 120:  # On track to exceed budget by 20%+
            alerts.append(
                {
                    "category": budget["category"],
                    "message": f"🔴 URGENT: {budget['category']} trending {percentage:.0f}% over budget",
                    "details": f"Spent €{spent:.2f} in {days_to_check} days, projecting €{projected_monthly:.2f}/month vs €{monthly_budget:.2f} budget",
                    "action": "Immediate spending freeze recommended",
                }
            )
        elif percentage > 100:  # On track to exceed budget
            warnings.append(
                {
                    "category": budget["category"],
                    "message": f"⚠️ WARNING: {budget['category']} trending {percentage:.0f}% of budget",
                    "details": f"Spent €{spent:.2f} recently, projecting €{projected_monthly:.2f}/month",
                    "action": "Consider reducing spending this month",
                }
            )
        elif percentage < 80:  # Well under budget
            good_news.append(
                {
                    "category": budget["category"],
                    "message": f"✅ GOOD: {budget['category']} at {percentage:.0f}% of budget",
                    "details": f"€{monthly_budget - projected_monthly:.2f} monthly savings potential",
                }
            )

    # Display alerts by severity
    if alerts:
        results.append("\n🚨 URGENT ALERTS")
        for alert in alerts:
            results.append(f"\n{alert['message']}")
            results.append(f"   {alert['details']}")
            results.append(f"   Action: {alert['action']}")

    if warnings:
        results.append("\n⚠️ BUDGET WARNINGS")
        for warning in warnings:
            results.append(f"\n{warning['message']}")
            results.append(f"   {warning['details']}")
            results.append(f"   Action: {warning['action']}")

    if good_news:
        results.append("\n✅ POSITIVE TRENDS")
        for good in good_news:
            results.append(f"\n{good['message']}")
            results.append(f"   {good['details']}")

    # Overall coaching
    results.append("\n💡 BUDGET COACHING")

    if alerts:
        results.append("• Focus on your red-alert categories this week")
        results.append("• Consider the 24-hour rule before any non-essential purchases")
        results.append("• Review if these are one-time expenses or pattern changes")
    elif warnings:
        results.append("• Yellow warnings are manageable with small adjustments")
        results.append(
            "• Look for easy wins like subscription pausing or meal planning"
        )
    else:
        results.append("• Great budget discipline! You're on track for your goals")
        results.append("• Consider if you can increase savings goals")

    results.extend(
        [
            "",
            "📱 SMART ACTIONS",
            "• Run analyze_subscription_value if recurring charges are high",
            "• Use check_budget for detailed category breakdowns",
            "• Set calendar reminders for budget check-ins",
        ]
    )

    return "\n".join(results)


def build_budget_agent() -> Agent[RunDeps]:
    """Build the Budget Specialist Agent."""

    return build_specialist_agent(
        name="BudgetSpecialist",
        instructions=BUDGET_SPECIALIST_INSTRUCTIONS,
        tools=[
            # Core budget tools
            set_budget,
            check_budget,
            list_budgets,
            suggest_budgets,
            delete_budget,
            # Advanced budget analysis
            analyze_spending_patterns,
            create_smart_budget_plan,
            budget_alert_system,
            # Recurring payment tools
            detect_recurring,
            list_subscriptions,
            analyze_subscription_value,
        ],
    )
