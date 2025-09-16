from __future__ import annotations

from datetime import datetime
from math import log, pow
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agents import Agent, RunContextWrapper, function_tool

from ..context import RunDeps
from .agent_factory import build_specialist_agent


class DebtInfo(BaseModel):
    """Information about a single debt."""

    name: str
    balance: float
    interest_rate: float
    minimum_payment: Optional[float] = None


DEBT_SPECIALIST_INSTRUCTIONS = """You are a Debt Management Specialist focused on debt elimination strategies, loan optimization, and cash-flow planning.

Principles:
- Prioritize high-interest debt elimination
- Balance psychological wins (snowball) with mathematical optimization (avalanche)
- Provide clear timelines and celebrate milestones
- Be supportive and non-judgmental
"""


@function_tool
async def analyze_debt_situation(
    ctx: RunContextWrapper[RunDeps],
    debts: List[DebtInfo],
    monthly_payment_capacity: Optional[float] = None,
) -> str:
    """Comprehensive debt analysis with payoff strategies and recommendations."""
    if not debts:
        return "No debts provided. Please include name, balance, interest_rate, and minimum_payment."

    deps = ctx.context

    if monthly_payment_capacity is None:
        # Estimate capacity from recent transactions (very rough)
        cur = deps.db.conn.cursor()
        cur.execute(
            """SELECT
                   AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS avg_income,
                   AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS avg_expenses
               FROM transactions
               WHERE date >= date('now', '-90 days')"""
        )
        data = cur.fetchone()
        monthly_income = (data["avg_income"] or 0) * 30
        monthly_expenses = (data["avg_expenses"] or 0) * 30
        monthly_payment_capacity = max((monthly_income - monthly_expenses) * 0.2, 100.0)

    # monthly_payment_capacity is guaranteed to be set by the logic above
    assert monthly_payment_capacity is not None
    payment_capacity = max(float(monthly_payment_capacity), 50.0)

    total_debt = sum(d.balance for d in debts)
    weighted_avg_rate = (
        sum(d.balance * d.interest_rate for d in debts) / total_debt
        if total_debt
        else 0
    )
    total_minimum = sum(d.minimum_payment or (d.balance * 0.02) for d in debts)

    results = ["💳 Comprehensive Debt Analysis\n" + "=" * 50]
    results.extend(
        [
            "📊 DEBT OVERVIEW",
            f"   Total Debt: €{total_debt:,.2f}",
            f"   Number of Debts: {len(debts)}",
            f"   Weighted Avg Interest: {weighted_avg_rate:.2f}%",
            f"   Total Minimum Payments: €{total_minimum:.2f}/month",
            f"   Available for Payments: €{payment_capacity:.2f}/month",
            "",
        ]
    )

    # Individual debt details
    results.append("📋 DEBT DETAILS")
    for d in sorted(debts, key=lambda x: x.interest_rate, reverse=True):
        min_pay = d.minimum_payment or (d.balance * 0.02)
        monthly_interest = d.balance * (d.interest_rate / 100 / 12)
        results.append(
            f"\n   💳 {d.name}\n      Balance: €{d.balance:,.2f}\n      Rate: {d.interest_rate:.2f}%\n      Min Payment: €{min_pay:.2f}\n      Monthly Interest: €{monthly_interest:.2f}"
        )

    # Strategy comparisons
    def calculate_payoff_timeline(
        ordered_debts: List[DebtInfo], monthly_payment: float, strategy: str
    ) -> Dict[str, Any]:
        total = sum(d.balance for d in ordered_debts)
        avg_rate = (
            sum(d.balance * d.interest_rate for d in ordered_debts) / total
            if total
            else 0
        )
        r = avg_rate / 100 / 12
        if r > 0 and monthly_payment > total * r:
            months = int(-log(1 - (total * r) / monthly_payment) / log(1 + r)) + 1
            total_interest = monthly_payment * months - total
        elif monthly_payment > 0:
            months = int(total / monthly_payment) + 1
            total_interest = 0.0
        else:
            months = 999
            total_interest = total * 2

        if strategy == "snowball" and ordered_debts:
            smallest = min(ordered_debts, key=lambda x: x.balance)
            first_payoff = max(1, int(smallest.balance / max(monthly_payment * 0.3, 1)))
        else:
            first_payoff = max(1, months // max(len(ordered_debts), 1))
        return {
            "months": months,
            "total_interest": total_interest,
            "first_payoff_months": first_payoff,
        }

    avalanche_order = sorted(debts, key=lambda x: x.interest_rate, reverse=True)
    snowball_order = sorted(debts, key=lambda x: x.balance)
    avalanche = calculate_payoff_timeline(
        avalanche_order, payment_capacity, "avalanche"
    )
    snowball = calculate_payoff_timeline(snowball_order, payment_capacity, "snowball")

    results.append("\n❄️ AVALANCHE METHOD (Mathematically Optimal)")
    for i, d in enumerate(avalanche_order, 1):
        results.append(f"   {i}. {d.name} ({d.interest_rate:.2f}%)")
    results.extend(
        [
            f"   Total Interest: €{avalanche['total_interest']:,.2f}",
            f"   Debt-free in: {avalanche['months']} months",
        ]
    )

    results.append("\n⛄ SNOWBALL METHOD (Psychological Momentum)")
    for i, d in enumerate(snowball_order, 1):
        results.append(f"   {i}. {d.name} (€{d.balance:,.2f})")
    results.extend(
        [
            f"   Total Interest: €{snowball['total_interest']:,.2f}",
            f"   Debt-free in: {snowball['months']} months",
            f"   First Payoff in: {snowball['first_payoff_months']} months",
        ]
    )

    interest_savings = snowball["total_interest"] - avalanche["total_interest"]
    results.append("\n📊 STRATEGY COMPARISON")
    if interest_savings > 0:
        results.append(f"   Avalanche saves €{interest_savings:,.2f} in interest")
        results.append(
            "   Use Avalanche if motivated by numbers; Snowball for quick wins"
        )
    else:
        results.append("   Both methods similar — choose based on motivation style")

    # Acceleration example
    extra = 100.0
    accel = calculate_payoff_timeline(
        avalanche_order, payment_capacity + extra, "avalanche"
    )
    months_saved = avalanche["months"] - accel["months"]
    interest_saved = avalanche["total_interest"] - accel["total_interest"]
    results.extend(
        [
            "\n🚀 ACCELERATION OPPORTUNITY",
            f"   +€{extra:.0f}/mo → saves {months_saved} months and €{interest_saved:,.2f} interest",
        ]
    )

    return "\n".join(results)


@function_tool
async def create_debt_freedom_plan(
    ctx: RunContextWrapper[RunDeps],
    target_date: Optional[str] = None,
    aggressive_mode: bool = False,
) -> str:
    """Create a personalized debt freedom plan with action steps."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute(
        """SELECT
               AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS avg_income,
               AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS avg_expenses
           FROM transactions
           WHERE date >= date('now', '-90 days')"""
    )
    data = cur.fetchone()
    if not data or data["avg_income"] is None:
        return "Insufficient transaction data. Please add at least 3 months of income and expenses."

    monthly_income = (data["avg_income"] or 0) * 30
    monthly_expenses = (data["avg_expenses"] or 0) * 30
    current_surplus = monthly_income - monthly_expenses

    results = ["🎯 Debt Freedom Action Plan\n" + "=" * 50]
    results.extend(
        [
            "📊 CURRENT SITUATION",
            f"   Monthly Income: €{monthly_income:.2f}",
            f"   Monthly Expenses: €{monthly_expenses:.2f}",
            f"   Current Surplus: €{current_surplus:.2f}",
            f"   Mode: {'AGGRESSIVE 🔥' if aggressive_mode else 'BALANCED ⚖️'}",
            "",
        ]
    )

    if target_date:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        months_to_target = max(1, int((target_dt - datetime.now()).days / 30))
        results.append(
            f"🎯 Target: Debt-free by {target_date} (~{months_to_target} months)"
        )
        results.append("")

    # Expense cuts and income ideas
    cut_pct = 0.30 if aggressive_mode else 0.15
    reduction_target = monthly_expenses * cut_pct
    income_target = monthly_income * (0.20 if aggressive_mode else 0.10)

    new_surplus = current_surplus + reduction_target + income_target
    results.append("✂️ EXPENSE REDUCTION PLAN")
    results.append(f"   Target: Reduce expenses by €{reduction_target:.2f}/mo")
    results.append("   • Review subscriptions and bills; negotiate rates")
    results.append("   • Reduce dining/entertainment temporarily")
    results.append("   • Implement 24-hour purchase rule")

    results.append("\n💰 INCOME INCREASE PLAN")
    results.append(f"   Target: Increase income by €{income_target:.2f}/mo")
    results.append("   • Extra shifts/freelance work; sell unused items")

    results.append("\n💳 DEBT PAYMENT CAPACITY")
    results.append(f"   TOTAL AVAILABLE FOR DEBT: €{new_surplus:.2f}/month")

    results.append("\n📅 WEEKLY ACTION PLAN")
    if aggressive_mode:
        results.append(
            "   MON: Expense audit; TUE: Apply gigs; WED: List items; FRI: Pay debt"
        )
    else:
        results.append("   MON: Track spending; WED: Review bills; FRI: Pay debt")

    results.append("\n📊 SUCCESS METRICS")
    results.append(f"   30 Days: €{new_surplus:.2f} paid toward debt")
    results.append(f"   90 Days: €{new_surplus*3:.2f} debt reduction")
    results.append("   Remember: Progress > Perfection. Every euro counts! 💪")

    return "\n".join(results)


@function_tool
async def debt_consolidation_analyzer(
    ctx: RunContextWrapper[RunDeps],
    current_debts_total: float,
    current_avg_interest: float,
    consolidation_loan_rate: float,
    consolidation_loan_term_months: int,
) -> str:
    """Analyze whether debt consolidation makes financial sense."""
    if current_debts_total <= 0:
        return "Debt amount must be positive"
    if consolidation_loan_term_months <= 0:
        return "Loan term must be positive"

    results = ["🔄 Debt Consolidation Analysis\n" + "=" * 50]

    # Current situation (assume 48 months remaining for comparison)
    current_term = 48
    r_cur = current_avg_interest / 100 / 12
    if r_cur > 0:
        cur_payment = (
            current_debts_total
            * (r_cur * pow(1 + r_cur, current_term))
            / (pow(1 + r_cur, current_term) - 1)
        )
        cur_total_paid = cur_payment * current_term
        cur_interest = cur_total_paid - current_debts_total
    else:
        cur_payment = current_debts_total / current_term
        cur_total_paid = current_debts_total
        cur_interest = 0.0

    r_con = consolidation_loan_rate / 100 / 12
    if r_con > 0:
        con_payment = (
            current_debts_total
            * (r_con * pow(1 + r_con, consolidation_loan_term_months))
            / (pow(1 + r_con, consolidation_loan_term_months) - 1)
        )
        con_total_paid = con_payment * consolidation_loan_term_months
        con_interest = con_total_paid - current_debts_total
    else:
        con_payment = current_debts_total / consolidation_loan_term_months
        con_total_paid = current_debts_total
        con_interest = 0.0

    monthly_diff = con_payment - cur_payment
    interest_diff = con_interest - cur_interest

    results.extend(
        [
            "💳 CURRENT DEBT SITUATION",
            f"   Total Debt: €{current_debts_total:,.2f}",
            f"   Avg Interest: {current_avg_interest:.2f}%",
            f"   Est. Monthly Payment: €{cur_payment:.2f}",
            f"   Total Interest (48 mo): €{cur_interest:,.2f}",
            "",
            "🏦 CONSOLIDATION OFFER",
            f"   Rate: {consolidation_loan_rate:.2f}% for {consolidation_loan_term_months} months",
            f"   Monthly Payment: €{con_payment:.2f}",
            f"   Total Interest: €{con_interest:,.2f}",
            "",
            "📊 COMPARISON",
            f"   Monthly Difference: €{monthly_diff:.2f}",
            f"   Total Interest Difference: €{interest_diff:,.2f}",
        ]
    )

    if consolidation_loan_rate < current_avg_interest - 2:
        results.append("\n✅ RECOMMENDATION: CONSOLIDATE (meaningfully lower rate)")
        if monthly_diff < 0:
            results.append(f"   Monthly savings: €{abs(monthly_diff):.2f}")
        if interest_diff < 0:
            results.append(f"   Interest savings: €{abs(interest_diff):,.2f}")
    elif consolidation_loan_rate > current_avg_interest:
        results.append("\n❌ RECOMMENDATION: DO NOT CONSOLIDATE (higher rate)")
    else:
        results.append(
            "\n⚖️ RECOMMENDATION: NEUTRAL (similar rate; consider cash flow needs)"
        )

    results.append(
        "\n⚠️ Consider fees, credit impact, and discipline to avoid new debt."
    )

    return "\n".join(results)


def build_debt_agent() -> Agent[RunDeps]:
    """Build the Debt Management Specialist Agent."""

    return build_specialist_agent(
        name="DebtSpecialist",
        instructions=DEBT_SPECIALIST_INSTRUCTIONS,
        tools=[
            analyze_debt_situation,
            create_debt_freedom_plan,
            debt_consolidation_analyzer,
        ],
    )
