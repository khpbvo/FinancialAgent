from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from agents import Agent, RunContextWrapper, function_tool

from ..context import RunDeps
from ..tools.goals import (
    create_goal,
    update_goal_progress,
    check_goals,
    suggest_savings_plan,
    complete_goal,
    pause_goal,
)
from .agent_factory import build_specialist_agent


GOAL_SPECIALIST_INSTRUCTIONS = """You are a Goal Achievement Specialist - an expert in financial goal setting, progress tracking, and motivation coaching for long-term financial success.

Key principles:
- Make goals SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- Break large goals into milestones and celebrate progress
- Favor sustainable progress over perfection
- Provide concrete amounts and dates for every recommendation

Coordinate with other specialists when needed (Investment, Tax, Budget, Debt)."""


@function_tool
async def create_comprehensive_financial_plan(
    ctx: RunContextWrapper[RunDeps],
    monthly_income: float,
    current_savings: float = 0,
    time_horizon_years: int = 10,
) -> str:
    """Create a comprehensive plan dividing savings across emergency, short-term, and long-term goals."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Estimate monthly spending from last 90 days
    cur.execute(
        """SELECT AVG(ABS(amount)) AS avg_daily_spending
           FROM transactions
           WHERE amount < 0 AND date >= date('now', '-90 days')"""
    )
    row = cur.fetchone()
    avg_monthly_spending = (row["avg_daily_spending"] or 0.0) * 30

    # Available savings capacity
    potential_monthly_savings = (
        monthly_income - avg_monthly_spending
        if avg_monthly_spending > 0
        else monthly_income * 0.2
    )

    results = ["🎯 Comprehensive Financial Plan\n" + "=" * 50]
    results.extend(
        [
            "📊 FINANCIAL BASELINE",
            f"   Monthly Income: €{monthly_income:.2f}",
            f"   Est. Monthly Spending: €{avg_monthly_spending:.2f}",
            f"   Available for Goals: €{potential_monthly_savings:.2f}",
            f"   Current Savings: €{current_savings:.2f}",
            f"   Planning Horizon: {time_horizon_years} years",
            "",
        ]
    )

    # Recommended allocations
    alloc_emergency = potential_monthly_savings * 0.5
    alloc_short_term = potential_monthly_savings * 0.3
    alloc_long_term = potential_monthly_savings * 0.2

    emergency_target = avg_monthly_spending * 6
    emergency_needed = max(0.0, emergency_target - current_savings)
    months_to_emergency = emergency_needed / alloc_emergency if alloc_emergency else 0

    results.append("💰 RECOMMENDED ALLOCATIONS")
    results.append(
        f"   Emergency Fund: €{alloc_emergency:.2f}/mo (target €{emergency_target:.2f})"
    )
    results.append(f"   Short-term Fund: €{alloc_short_term:.2f}/mo (2-year horizon)")
    results.append(
        f"   Long-term Wealth: €{alloc_long_term:.2f}/mo ({time_horizon_years} years)"
    )

    if emergency_needed > 0 and alloc_emergency > 0:
        eta = datetime.now() + timedelta(days=int(months_to_emergency * 30))
        results.append(
            f"   Emergency fund fully funded in ~{months_to_emergency:.0f} months (by {eta.strftime('%b %Y')})"
        )

    results.extend(
        [
            "",
            "📋 NEXT STEPS",
            "1. Create individual goals (emergency, short-term, long-term)",
            "2. Automate monthly transfers on payday",
            "3. Review progress monthly and rebalance allocations as needed",
        ]
    )

    return "\n".join(results)


@function_tool
async def goal_motivation_coach(
    ctx: RunContextWrapper[RunDeps],
    goal_name: Optional[str] = None,
    include_progress_celebration: bool = True,
) -> str:
    """Provide motivational coaching and accountability for financial goal achievement."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    if goal_name:
        cur.execute(
            "SELECT * FROM goals WHERE name = ? AND status = 'active'",
            (goal_name,),
        )
        goals = cur.fetchall()
        if not goals:
            return f"No active goal found with name '{goal_name}'"
    else:
        cur.execute("SELECT * FROM goals WHERE status = 'active' ORDER BY created_at")
        goals = cur.fetchall()

    if not goals:
        return "No active goals found. Create a goal to get started!"

    results = ["🎯 Goal Achievement Coaching Session\n" + "=" * 50]

    total_goals = len(goals)
    goals_on_track = 0
    goals_behind = 0
    total_progress = 0.0

    for goal in goals:
        progress = (goal["current_amount"] / max(goal["target_amount"], 1)) * 100
        total_progress += progress

        created_date = datetime.strptime(goal["created_at"], "%Y-%m-%d %H:%M:%S")
        days_since_creation = (datetime.now() - created_date).days

        if goal["target_date"]:
            target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
            total_days = (target_date - created_date).days
            time_progress = (
                (days_since_creation / total_days) * 100 if total_days > 0 else 0
            )
            if progress >= time_progress - 10:
                goals_on_track += 1
            else:
                goals_behind += 1
        else:
            goals_on_track += 1 if progress > 0 else 0
            goals_behind += 0 if progress > 0 else 1

    avg_progress = total_progress / max(total_goals, 1)
    results.append("🌟 MOTIVATION & PROGRESS CHECK")
    if avg_progress > 75:
        results.append("   🚀 AMAZING PROGRESS! Keep the momentum going!")
    elif avg_progress > 50:
        results.append("   💪 SOLID PROGRESS! You're more than halfway there!")
    elif avg_progress > 25:
        results.append("   🌱 GREAT START! Consistency will compound your results!")
    else:
        results.append("   🎯 TIME TO ACCELERATE! Small daily steps add up!")

    results.append("\n📊 INDIVIDUAL GOAL COACHING")
    for goal in goals:
        progress = (goal["current_amount"] / max(goal["target_amount"], 1)) * 100
        remaining = goal["target_amount"] - goal["current_amount"]
        results.append(f"\n🎯 {goal['name']}")
        results.append(
            f"   Progress: {progress:.1f}% (€{goal['current_amount']:.2f}/€{goal['target_amount']:.2f})"
        )
        if goal["target_date"]:
            target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
            days_remaining = (target_date - datetime.now()).days
            if days_remaining > 0:
                daily_needed = remaining / days_remaining
                results.append(
                    f"   📅 {days_remaining} days left → save €{daily_needed:.2f}/day"
                )
            else:
                results.append("   ⚠️ Target date passed — consider adjusting timeline")

    results.append("\n💡 MOTIVATION BOOSTERS")
    if include_progress_celebration and any(g["current_amount"] > 0 for g in goals):
        total_saved = sum(g["current_amount"] for g in goals)
        results.append(f"   🎉 You've saved €{total_saved:.2f} across all goals!")

    if goals_on_track > goals_behind:
        results.append("   🏆 You're winning at goal achievement — keep it up!")
    elif goals_behind > 0:
        results.extend(
            [
                "   🔧 ADJUSTMENT STRATEGIES:",
                "   • Break big goals into weekly targets",
                "   • Automate transfers on payday",
                "   • Cut one small recurring expense",
            ]
        )

    results.append("\n📋 THIS WEEK'S ACTION ITEMS")
    for i, goal in enumerate(goals[:3], 1):
        progress = (goal["current_amount"] / max(goal["target_amount"], 1)) * 100
        if progress < 10:
            action = f"Make your first €50 contribution to {goal['name']}"
        elif progress < 50:
            weekly_target = (goal["target_amount"] - goal["current_amount"]) / 52
            action = f"Save €{weekly_target:.0f} toward {goal['name']}"
        else:
            action = f"Celebrate progress and add €{goal['target_amount']*0.01:.0f} to {goal['name']}"
        results.append(f"   {i}. {action}")

    results.extend(
        [
            "",
            "🌟 REMEMBER WHY YOU STARTED",
            "Freedom, security, and opportunity come from consistent action.",
        ]
    )

    return "\n".join(results)


@function_tool
async def optimize_goal_strategy(
    ctx: RunContextWrapper[RunDeps], life_change_event: Optional[str] = None
) -> str:
    """Analyze and optimize goal strategy based on current situation and life changes."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute("SELECT * FROM goals WHERE status = 'active' ORDER BY created_at")
    goals = cur.fetchall()
    if not goals:
        return "No active goals to optimize. Create goals first using create_goal."

    cur.execute(
        """SELECT 
               AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS avg_income,
               AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS avg_expenses
           FROM transactions 
           WHERE date >= date('now', '-90 days')"""
    )
    financial_data = cur.fetchone()
    monthly_income = (financial_data["avg_income"] or 0) * 30
    monthly_expenses = (financial_data["avg_expenses"] or 0) * 30
    available_for_goals = monthly_income - monthly_expenses

    results = ["⚡ Goal Strategy Optimization\n" + "=" * 50]
    if life_change_event:
        results.append(
            f"🔄 Adapting to life change: {life_change_event.replace('_', ' ').title()}"
        )
        results.append("")

    # Performance analysis
    goal_perf = []
    for goal in goals:
        created_date = datetime.strptime(goal["created_at"], "%Y-%m-%d %H:%M:%S")
        days_active = max((datetime.now() - created_date).days, 1)
        progress_rate = goal["current_amount"] / days_active
        if goal["target_date"]:
            target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
            total_days = max((target_date - created_date).days, 1)
            expected = (days_active / total_days) * goal["target_amount"]
            perf_ratio = goal["current_amount"] / max(expected, 1)
        else:
            perf_ratio = 1.0
        goal_perf.append({"goal": goal, "perf": perf_ratio, "rate": progress_rate})

    goal_perf.sort(key=lambda x: x["perf"], reverse=True)
    best = goal_perf[0]
    worst = goal_perf[-1]

    results.append("📊 PERFORMANCE SNAPSHOT")
    results.append(
        f"   Best: {best['goal']['name']} ({best['perf']*100:.0f}% of target pace)"
    )
    results.append(
        f"   Needs Attention: {worst['goal']['name']} ({worst['perf']*100:.0f}% of target pace)"
    )

    # Allocation analysis
    total_monthly_needed = 0.0
    for perf in goal_perf:
        g = perf["goal"]
        if g["target_date"]:
            target_date = datetime.strptime(g["target_date"], "%Y-%m-%d")
            months_left = max((target_date - datetime.now()).days / 30, 1)
            remaining = g["target_amount"] - g["current_amount"]
            total_monthly_needed += remaining / months_left

    results.append("\n💰 RESOURCE ALLOCATION")
    results.append(f"   Available monthly: €{available_for_goals:.2f}")
    results.append(f"   Goals need monthly: €{total_monthly_needed:.2f}")

    if total_monthly_needed > available_for_goals:
        shortfall = total_monthly_needed - available_for_goals
        results.extend(
            [
                f"   ⚠️ Shortfall: €{shortfall:.2f}/mo",
                "   Extend timelines for lower-priority goals",
                "   Focus on top 2-3 goals and automate transfers",
            ]
        )
    else:
        surplus = available_for_goals - total_monthly_needed
        results.extend(
            [
                f"   ✅ Surplus: €{surplus:.2f}/mo",
                "   Accelerate highest-priority goal or build buffer",
            ]
        )

    return "\n".join(results)


def build_goal_agent() -> Agent[RunDeps]:
    """Build the Goal Achievement Specialist Agent."""

    return build_specialist_agent(
        name="GoalSpecialist",
        instructions=GOAL_SPECIALIST_INSTRUCTIONS,
        tools=[
            create_goal,
            update_goal_progress,
            check_goals,
            suggest_savings_plan,
            complete_goal,
            pause_goal,
            create_comprehensive_financial_plan,
            goal_motivation_coach,
            optimize_goal_strategy,
        ],
    )
