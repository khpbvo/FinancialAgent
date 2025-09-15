        """SELECT 
           FROM transactions 
           FROM transactions 
           WHERE amount > 0 
from __future__ import annotations
           WHERE date >= date('now', '-90 days')"""
from __future__ import annotations
from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from openai.types.shared import Reasoning
from ..context import RunDeps
from .agent_factory import build_specialist_agent


INVESTMENT_SPECIALIST_INSTRUCTIONS = """You are an Investment Portfolio Specialist - an expert in investment strategy, portfolio analysis, and wealth building through smart asset allocation.

Your expertise includes:
• Portfolio diversification and risk assessment
• Asset allocation strategies based on age and goals
• Investment opportunity evaluation and timing
• Cost-benefit analysis of investment vehicles
• Investment growth calculations and compound interest projections
• Tax-efficient investment account placement (defer to Tax Specialist for tax law details)
• Market trend analysis and rebalancing strategies

Key principles:
- Focus on long-term wealth building over short-term gains
- Emphasize low-cost, diversified index fund strategies
- Consider tax implications of investment decisions
- Balance risk tolerance with return expectations
- Provide education about investment fundamentals
- Stress the importance of time in the market vs timing the market

Your specialty is making investing accessible and understandable while helping users build wealth through proven, evidence-based strategies tailored to their risk tolerance and timeline.

Always provide specific calculations, risk assessments, and actionable recommendations with clear reasoning.

TEAM COORDINATION:
- Work closely with Tax Specialist on tax-efficient investment placement and timing
- Support Goal Specialist by aligning investment strategies with specific financial targets
- Consider Budget Specialist insights on available investment amounts and cash flow
- Collaborate with Debt Specialist on debt payoff vs investment prioritization decisions
- Handle portfolio mechanics while Goal Specialist provides motivation and milestone tracking"""


@function_tool
async def analyze_investment_readiness(
    ctx: RunContextWrapper[RunDeps],
    emergency_fund_months: int = 3,
    high_interest_debt_threshold: float = 7.0,
) -> str:
    """Analyze user's financial readiness for investing based on prerequisites.

    Args:
        emergency_fund_months: Required months of expenses in emergency fund
        high_interest_debt_threshold: Interest rate above which debt should be paid first
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get recent income and expenses
    cur.execute(
        """SELECT
               AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) as avg_income,
               AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as avg_expenses,
               COUNT(*) as transaction_count
           FROM transactions
           WHERE date >= date('now', '-90 days')"""
    )

    financial_data = cur.fetchone()

    if not financial_data or financial_data["transaction_count"] == 0:
        return "Not enough transaction data. Please add at least 3 months of transactions first."

    monthly_income = (financial_data["avg_income"] or 0) * 30
    monthly_expenses = (financial_data["avg_expenses"] or 0) * 30
    monthly_surplus = monthly_income - monthly_expenses

    # Check for savings/emergency fund
    cur.execute(
        "SELECT * FROM goals WHERE (name LIKE '%emergency%' OR name LIKE '%savings%') AND status = 'active'"
    )
    emergency_goals = cur.fetchall()

    current_emergency_savings = (
        sum(g["current_amount"] for g in emergency_goals) if emergency_goals else 0
    )
    target_emergency_fund = monthly_expenses * emergency_fund_months

    results = ["💼 Investment Readiness Assessment\n" + "=" * 50]

    # Financial baseline
    results.extend(
        [
            "📊 FINANCIAL BASELINE",
            f"   Monthly Income: €{monthly_income:.2f}",
            f"   Monthly Expenses: €{monthly_expenses:.2f}",
            f"   Monthly Surplus: €{monthly_surplus:.2f}",
            (
                f"   Savings Rate: {(monthly_surplus/monthly_income*100):.1f}%"
                if monthly_income > 0
                else "   Savings Rate: N/A"
            ),
            "",
        ]
    )

    # Investment readiness checklist
    results.append("✅ INVESTMENT READINESS CHECKLIST")

    readiness_score = 0
    max_score = 5

    # 1. Emergency fund check
    emergency_fund_complete = current_emergency_savings >= target_emergency_fund
    if emergency_fund_complete:
        results.append(
            f"   ✅ Emergency Fund: €{current_emergency_savings:.2f} ({emergency_fund_months} months)"
        )
        readiness_score += 1
    else:
        shortfall = target_emergency_fund - current_emergency_savings
        results.append(f"   ❌ Emergency Fund: Need €{shortfall:.2f} more")
        results.append(
            f"      Current: €{current_emergency_savings:.2f}, Target: €{target_emergency_fund:.2f}"
        )

    # 2. Positive cash flow
    has_positive_cashflow = monthly_surplus > 0
    if has_positive_cashflow:
        results.append(
            f"   ✅ Positive Cash Flow: €{monthly_surplus:.2f}/month available"
        )
        readiness_score += 1
    else:
        results.append(
            f"   ❌ Negative Cash Flow: €{abs(monthly_surplus):.2f}/month deficit"
        )

    # 3. High-interest debt check (simplified - would need debt tracking in real implementation)
    results.append(
        f"   ⚠️ High-Interest Debt: Check manually (pay off debt >{high_interest_debt_threshold}% first)"
    )

    # 4. Income stability check
    cur.execute(
        """SELECT COUNT(DISTINCT strftime('%Y-%m', date)) as income_months
           FROM transactions
           WHERE amount > 0
           AND date >= date('now', '-180 days')"""
    )
    income_stability = cur.fetchone()
    stable_income = (income_stability["income_months"] or 0) >= 5

    if stable_income:
        results.append(
            f"   ✅ Income Stability: {income_stability['income_months']} months of income"
        )
        readiness_score += 1
    else:
        results.append(
            f"   ❌ Income Stability: Only {income_stability['income_months']} months recorded"
        )

    # 5. Investment knowledge (assumed based on query)
    results.append("   📚 Investment Knowledge: Self-assess your understanding")

    # Overall readiness assessment
    results.append(f"\n🎯 READINESS SCORE: {readiness_score}/{max_score-1}")

    if readiness_score >= 3:
        results.extend(
            [
                "✅ You're READY to start investing!",
                "",
                "💡 RECOMMENDED NEXT STEPS:",
                "1. Start with low-cost index funds (S&P 500, Total Market)",
                "2. Consider tax-advantaged accounts first",
                "3. Begin with 10-20% of monthly surplus",
                "4. Dollar-cost average to reduce timing risk",
            ]
        )

        # Investment amount recommendation
        conservative_investment = monthly_surplus * 0.3
        moderate_investment = monthly_surplus * 0.5
        aggressive_investment = monthly_surplus * 0.7

        results.extend(
            [
                "",
                "💰 INVESTMENT AMOUNT SUGGESTIONS:",
                f"   Conservative (30%): €{conservative_investment:.2f}/month",
                f"   Moderate (50%): €{moderate_investment:.2f}/month",
                f"   Aggressive (70%): €{aggressive_investment:.2f}/month",
            ]
        )

    elif readiness_score == 2:
        results.extend(
            [
                "⚠️ ALMOST READY - Address remaining items first",
                "",
                "Priority Actions:",
                "1. Complete emergency fund before investing",
                "2. Ensure consistent positive cash flow",
                "3. Pay off high-interest debt (credit cards)",
                "4. Then start with small amounts (€50-100/month)",
            ]
        )
    else:
        results.extend(
            [
                "❌ NOT YET READY - Build foundation first",
                "",
                "📋 Foundation Building Steps:",
                "1. Focus on emergency fund (top priority)",
                "2. Eliminate high-interest debt",
                "3. Create consistent budget with surplus",
                "4. Learn investment basics while preparing",
                "",
                "Remember: A strong foundation prevents future problems!",
            ]
        )

    # Risk tolerance assessment prompt
    results.extend(
        [
            "",
            "🎲 RISK TOLERANCE ASSESSMENT",
            "Consider your comfort level with:",
            "• Market volatility (can you handle -20% drops?)",
            "• Investment timeline (5+ years minimum recommended)",
            "• Sleep test (will market drops keep you awake?)",
            "",
            "📊 TYPICAL ALLOCATIONS BY AGE:",
            "• 20s-30s: 80-90% stocks, 10-20% bonds",
            "• 40s: 70-80% stocks, 20-30% bonds",
            "• 50s: 60-70% stocks, 30-40% bonds",
            "• 60s+: 40-60% stocks, 40-60% bonds",
        ]
    )

    return "\n".join(results)


@function_tool
async def calculate_investment_projections(
    ctx: RunContextWrapper[RunDeps],
    monthly_investment: float,
    years: int = 10,
    annual_return: float = 7.0,
    inflation_rate: float = 2.5,
) -> str:
    """Calculate investment growth projections with compound interest.

    Args:
        monthly_investment: Amount to invest each month
        years: Investment timeline in years
        annual_return: Expected annual return percentage
        inflation_rate: Annual inflation rate for real return calculation
    """
    from math import pow

    results = ["📈 Investment Growth Projections\n" + "=" * 50]

    # Input validation
    if monthly_investment <= 0:
        return "Monthly investment must be positive"

    if years <= 0 or years > 50:
        return "Investment timeline must be between 1 and 50 years"

    # Display assumptions
    results.extend(
        [
            "💰 INVESTMENT PARAMETERS",
            f"   Monthly Investment: €{monthly_investment:.2f}",
            f"   Timeline: {years} years",
            f"   Expected Annual Return: {annual_return}%",
            f"   Inflation Rate: {inflation_rate}%",
            f"   Real Return: {annual_return - inflation_rate}%",
            "",
        ]
    )

    # Calculate projections
    monthly_rate = annual_return / 100 / 12
    months = years * 12
    total_invested = monthly_investment * months

    # Future value of monthly investments (compound interest formula)
    if monthly_rate > 0:
        future_value = monthly_investment * (
            (pow(1 + monthly_rate, months) - 1) / monthly_rate
        )
    else:
        future_value = total_invested

    investment_gains = future_value - total_invested

    # Inflation-adjusted value
    real_rate = (annual_return - inflation_rate) / 100 / 12
    if real_rate > 0:
        real_future_value = monthly_investment * (
            (pow(1 + real_rate, months) - 1) / real_rate
        )
    else:
        real_future_value = total_invested / pow(1 + inflation_rate / 100, years)

    results.extend(
        [
            "📊 PROJECTION RESULTS",
            f"   Total Invested: €{total_invested:,.2f}",
            f"   Future Value: €{future_value:,.2f}",
            f"   Investment Gains: €{investment_gains:,.2f}",
            f"   Return on Investment: {(investment_gains/total_invested*100):.1f}%",
            "",
            "💵 INFLATION-ADJUSTED VALUE",
            f"   Real Value (today's euros): €{real_future_value:,.2f}",
            f"   Purchasing Power Lost: €{future_value - real_future_value:,.2f}",
            "",
        ]
    )

    # Milestone projections
    results.append("🎯 MILESTONE PROJECTIONS")

    milestones = [1, 5, 10, 15, 20, 25, 30]
    for milestone_year in milestones:
        if milestone_year <= years:
            milestone_months = milestone_year * 12
            milestone_invested = monthly_investment * milestone_months

            if monthly_rate > 0:
                milestone_value = monthly_investment * (
                    (pow(1 + monthly_rate, milestone_months) - 1) / monthly_rate
                )
            else:
                milestone_value = milestone_invested

            results.append(
                f"   Year {milestone_year}: €{milestone_value:,.0f} (invested: €{milestone_invested:,.0f})"
            )

    # Scenario analysis
    results.append("\n📊 SCENARIO ANALYSIS")

    # Conservative scenario
    conservative_rate = (annual_return - 2) / 100 / 12
    if conservative_rate > 0:
        conservative_value = monthly_investment * (
            (pow(1 + conservative_rate, months) - 1) / conservative_rate
        )
    else:
        conservative_value = total_invested

    # Optimistic scenario
    optimistic_rate = (annual_return + 2) / 100 / 12
    optimistic_value = monthly_investment * (
        (pow(1 + optimistic_rate, months) - 1) / optimistic_rate
    )

    results.extend(
        [
            f"   Conservative ({annual_return-2}%): €{conservative_value:,.2f}",
            f"   Expected ({annual_return}%): €{future_value:,.2f}",
            f"   Optimistic ({annual_return+2}%): €{optimistic_value:,.2f}",
            "",
        ]
    )

    # Power of starting early
    if years >= 10:
        # Calculate what happens if you delay by 5 years
        delayed_years = years - 5
        delayed_months = delayed_years * 12

        # To reach same goal, need higher monthly investment
        if monthly_rate > 0:
            delayed_future_value = monthly_investment * (
                (pow(1 + monthly_rate, delayed_months) - 1) / monthly_rate
            )
            required_monthly = (
                future_value
                * monthly_rate
                / ((pow(1 + monthly_rate, delayed_months) - 1))
            )
        else:
            delayed_future_value = monthly_investment * delayed_months
            required_monthly = future_value / delayed_months

        cost_of_waiting = future_value - delayed_future_value

        results.extend(
            [
                "⏰ COST OF WAITING 5 YEARS",
                f"   Lost Growth: €{cost_of_waiting:,.2f}",
                f"   Required Monthly (to catch up): €{required_monthly:.2f}",
                f"   Extra Monthly Needed: €{required_monthly - monthly_investment:.2f}",
                "",
            ]
        )

    # Retirement income potential
    safe_withdrawal_rate = 0.04  # 4% rule
    annual_income = future_value * safe_withdrawal_rate
    monthly_income = annual_income / 12

    results.extend(
        [
            "🏖️ RETIREMENT INCOME POTENTIAL",
            f"   Portfolio Value: €{future_value:,.2f}",
            f"   Safe Annual Withdrawal (4%): €{annual_income:,.2f}",
            f"   Monthly Income: €{monthly_income:.2f}",
            "",
            "💡 KEY INSIGHTS",
            f"• Every €{monthly_investment} invested grows to €{future_value/total_invested*monthly_investment:.2f}",
            f"• Compound interest adds €{investment_gains:,.2f} to your wealth",
            (
                f"• Starting now vs 5 years later: €{cost_of_waiting:,.2f} difference"
                if years >= 10
                else "• Time in market beats timing the market"
            ),
            "• Consistency is more important than amount",
        ]
    )

    return "\n".join(results)


@function_tool
async def portfolio_rebalancing_advisor(
    ctx: RunContextWrapper[RunDeps],
    target_stock_allocation: float = 70,
    target_bond_allocation: float = 30,
    rebalancing_threshold: float = 5.0,
) -> str:
    """Analyze portfolio allocation and provide rebalancing recommendations.

    Args:
        target_stock_allocation: Target percentage for stocks
        target_bond_allocation: Target percentage for bonds
        rebalancing_threshold: Trigger rebalancing if off by this percentage
    """

    # Note: In a real implementation, we'd track actual portfolio holdings
    # For now, we'll provide educational content about rebalancing

    results = ["⚖️ Portfolio Rebalancing Guide\n" + "=" * 50]

    # Validate allocations
    if abs((target_stock_allocation + target_bond_allocation) - 100) > 0.01:
        return "Target allocations must sum to 100%"

    results.extend(
        [
            "🎯 TARGET ALLOCATION",
            f"   Stocks: {target_stock_allocation}%",
            f"   Bonds: {target_bond_allocation}%",
            f"   Rebalance Threshold: ±{rebalancing_threshold}%",
            "",
        ]
    )

    # Rebalancing strategies
    results.extend(
        [
            "📊 REBALANCING STRATEGIES",
            "",
            "1️⃣ CALENDAR REBALANCING",
            "   • Quarterly: Good for active investors",
            "   • Semi-Annual: Balance between cost and drift",
            "   • Annual: Minimal costs, may allow more drift",
            "",
            "2️⃣ THRESHOLD REBALANCING",
            f"   • Rebalance when allocation drifts >{rebalancing_threshold}%",
            "   • More responsive to market movements",
            "   • Can capture reversion to mean",
            "",
            "3️⃣ HYBRID APPROACH",
            "   • Check quarterly, rebalance if threshold exceeded",
            "   • Annual rebalancing regardless of drift",
            "   • Best of both strategies",
        ]
    )

    # Example calculation
    example_portfolio = 100000
    example_stocks = example_portfolio * (target_stock_allocation / 100)
    example_bonds = example_portfolio * (target_bond_allocation / 100)

    # Simulate market movement
    stock_growth = 1.15  # 15% gain
    bond_growth = 1.02  # 2% gain

    new_stocks = example_stocks * stock_growth
    new_bonds = example_bonds * bond_growth
    new_total = new_stocks + new_bonds

    new_stock_pct = (new_stocks / new_total) * 100
    new_bond_pct = (new_bonds / new_total) * 100

    stock_drift = new_stock_pct - target_stock_allocation
    bond_drift = new_bond_pct - target_bond_allocation

    results.extend(
        [
            "",
            "📈 REBALANCING EXAMPLE",
            f"   Starting Portfolio: €{example_portfolio:,.0f}",
            f"   Initial: {target_stock_allocation}% stocks, {target_bond_allocation}% bonds",
            "",
            "   After Market Movement:",
            f"   • Stocks: +15% → €{new_stocks:,.0f} ({new_stock_pct:.1f}%)",
            f"   • Bonds: +2% → €{new_bonds:,.0f} ({new_bond_pct:.1f}%)",
            f"   • Total: €{new_total:,.0f}",
            "",
            f"   Drift: Stocks {stock_drift:+.1f}%, Bonds {bond_drift:+.1f}%",
        ]
    )

    if abs(stock_drift) > rebalancing_threshold:
        # Calculate rebalancing trades
        target_stocks_value = new_total * (target_stock_allocation / 100)
        target_bonds_value = new_total * (target_bond_allocation / 100)

        stocks_to_sell = new_stocks - target_stocks_value
        bonds_to_buy = target_bonds_value - new_bonds

        results.extend(
            [
                "",
                f"⚠️ REBALANCING NEEDED (drift >{rebalancing_threshold}%)",
                "   Actions:",
                f"   • Sell €{stocks_to_sell:,.0f} of stocks",
                f"   • Buy €{bonds_to_buy:,.0f} of bonds",
                f"   Result: Back to {target_stock_allocation}/{target_bond_allocation} allocation",
            ]
        )
    else:
        results.append(f"\n✅ No rebalancing needed (drift <{rebalancing_threshold}%)")

    # Tax-efficient rebalancing tips
    results.extend(
        [
            "",
            "💡 TAX-EFFICIENT REBALANCING TIPS",
            "• Use new contributions to rebalance (no taxable sales)",
            "• Rebalance in tax-deferred accounts first",
            "• Consider tax-loss harvesting opportunities",
            "• Avoid short-term capital gains (<1 year)",
            "• Use dividend reinvestment strategically",
            "",
            "📊 REBALANCING BENEFITS",
            "• Forces 'buy low, sell high' discipline",
            "• Maintains risk level over time",
            "• Captures mean reversion profits",
            "• Reduces emotional decision-making",
            "",
            "⚠️ COMMON MISTAKES TO AVOID",
            "• Over-rebalancing (too frequent = high costs)",
            "• Ignoring tax consequences",
            "• Rebalancing during high volatility",
            "• Not considering transaction costs",
            "• Abandoning strategy during downturns",
        ]
    )

    return "\n".join(results)


def build_investment_agent() -> Agent[RunDeps]:
    """Build the Investment Portfolio Specialist Agent."""

    return build_specialist_agent(
        name="InvestmentSpecialist",
        instructions=INVESTMENT_SPECIALIST_INSTRUCTIONS,
        tools=[
            analyze_investment_readiness,
            calculate_investment_projections,
            portfolio_rebalancing_advisor,
        ],
    )