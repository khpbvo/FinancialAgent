from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from openai.types.shared import Reasoning
from ..context import RunDeps


class DebtInfo(BaseModel):
    """Information about a single debt."""
    name: str
    balance: float
    interest_rate: float
    minimum_payment: Optional[float] = None


DEBT_SPECIALIST_INSTRUCTIONS = """You are a Debt Management Specialist - an expert in debt elimination strategies, loan optimization, and credit score improvement.

Your expertise includes:
• Debt payoff strategies (avalanche, snowball, hybrid methods)
• Interest rate optimization and refinancing analysis
• Credit score improvement tactics and timeline planning
• Debt consolidation evaluation and recommendations
• Student loan forgiveness and repayment programs
• Mortgage optimization and early payoff calculations
• Emergency debt management during financial crisis

Key principles:
- Prioritize high-interest debt elimination
- Balance psychological wins with mathematical optimization
- Consider both short-term relief and long-term wealth building (coordinate with Investment Specialist on post-debt plans)
- Educate about the true cost of debt and compound interest
- Provide hope and clear paths out of debt situations
- Never judge, always support and encourage

Your specialty is creating personalized debt freedom plans that balance mathematical efficiency with human psychology, helping users become debt-free faster while maintaining quality of life.

Always provide specific calculations, timelines, and celebrate progress milestones.

TEAM COORDINATION:
- Work with Tax Specialist on tax implications of debt consolidation and payoff strategies
- Partner with Budget Specialist to identify cash flow for aggressive debt payments
- Collaborate with Goal Specialist on balancing debt payoff vs emergency fund building
- Advise Investment Specialist on debt payoff vs investment prioritization decisions
- Focus on debt elimination while Goal Specialist maintains motivation for long-term plans"""


@function_tool
async def analyze_debt_situation(
    ctx: RunContextWrapper[RunDeps],
    debts: List[DebtInfo],
    monthly_payment_capacity: Optional[float] = None
) -> str:
    """Comprehensive debt analysis with payoff strategies and recommendations.
    
    Args:
        debts: List of DebtInfo objects with name, balance, interest_rate, minimum_payment
        monthly_payment_capacity: Total amount available for debt payments monthly
    """
    if not debts:
        return "No debts provided. Please add debt information with balance, interest rate, and minimum payment."
    
    deps = ctx.context
    
    # If payment capacity not provided, try to estimate from transactions
    if not monthly_payment_capacity:
        cur = deps.db.conn.cursor()
        cur.execute(
            """SELECT 
                   AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) as avg_income,
                   AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as avg_expenses
               FROM transactions 
               WHERE date >= date('now', '-90 days')"""
        )
        financial_data = cur.fetchone()
        
        if financial_data and financial_data['avg_income']:
            monthly_income = financial_data['avg_income'] * 30
            monthly_expenses = financial_data['avg_expenses'] * 30
            # Rough estimate: 20% of surplus for debt payments
            monthly_payment_capacity = max((monthly_income - monthly_expenses) * 0.2, 100)
        else:
            # Default conservative estimate
            monthly_payment_capacity = 500
    
    results = [f"💳 Comprehensive Debt Analysis\n" + "=" * 50]
    
    # Calculate debt metrics
    total_debt = sum(d.balance for d in debts)
    total_minimum = sum(d.minimum_payment or (d.balance * 0.02) for d in debts)
    weighted_avg_rate = sum(d.balance * d.interest_rate for d in debts) / total_debt if total_debt > 0 else 0
    
    results.extend([
        f"📊 DEBT OVERVIEW",
        f"   Total Debt: €{total_debt:,.2f}",
        f"   Number of Debts: {len(debts)}",
        f"   Weighted Avg Interest: {weighted_avg_rate:.2f}%",
        f"   Total Minimum Payments: €{total_minimum:.2f}/month",
        f"   Available for Payments: €{monthly_payment_capacity:.2f}/month",
        ""
    ])
    
    # Individual debt analysis
    results.append("📋 DEBT DETAILS")
    
    sorted_by_rate = sorted(debts, key=lambda x: x.interest_rate, reverse=True)
    sorted_by_balance = sorted(debts, key=lambda x: x.balance)
    
    for debt in sorted_by_rate:
        min_payment = debt.minimum_payment or (debt.balance * 0.02)
        monthly_interest = debt.balance * (debt.interest_rate / 100 / 12)
        
        results.append(f"\n   💳 {debt.name}")
        results.append(f"      Balance: €{debt.balance:,.2f}")
        results.append(f"      Interest Rate: {debt.interest_rate:.2f}%")
        results.append(f"      Min Payment: €{min_payment:.2f}")
        results.append(f"      Monthly Interest Cost: €{monthly_interest:.2f}")
    
    # Strategy 1: Avalanche Method (Highest Interest First)
    results.append(f"\n❄️ AVALANCHE METHOD (Mathematically Optimal)")
    
    avalanche_order = sorted_by_rate.copy()
    avalanche_timeline = calculate_payoff_timeline(
        avalanche_order, 
        monthly_payment_capacity,
        "avalanche"
    )
    
    results.append(f"   Pay debts in this order:")
    for i, debt in enumerate(avalanche_order, 1):
        results.append(f"   {i}. {debt.name} ({debt.interest_rate:.2f}%)")
    
    results.extend([
        f"   Total Interest Paid: €{avalanche_timeline['total_interest']:,.2f}",
        f"   Time to Debt Freedom: {avalanche_timeline['months']} months",
        f"   Monthly Payment: €{monthly_payment_capacity:.2f}"
    ])
    
    # Strategy 2: Snowball Method (Smallest Balance First)
    results.append(f"\n⛄ SNOWBALL METHOD (Psychological Wins)")
    
    snowball_order = sorted_by_balance.copy()
    snowball_timeline = calculate_payoff_timeline(
        snowball_order,
        monthly_payment_capacity,
        "snowball"
    )
    
    results.append(f"   Pay debts in this order:")
    for i, debt in enumerate(snowball_order, 1):
        results.append(f"   {i}. {debt.name} (€{debt.balance:,.2f})")
    
    results.extend([
        f"   Total Interest Paid: €{snowball_timeline['total_interest']:,.2f}",
        f"   Time to Debt Freedom: {snowball_timeline['months']} months",
        f"   First Debt Eliminated: {snowball_timeline['first_payoff_months']} months"
    ])
    
    # Strategy comparison
    interest_savings = snowball_timeline['total_interest'] - avalanche_timeline['total_interest']
    
    results.append(f"\n📊 STRATEGY COMPARISON")
    if interest_savings > 0:
        results.extend([
            f"   Avalanche saves: €{interest_savings:,.2f} in interest",
            f"   Recommendation: Use AVALANCHE if motivated by numbers",
            f"   Alternative: Use SNOWBALL if you need quick wins"
        ])
    else:
        results.append("   Both methods are similar - choose based on preference")
    
    # Acceleration opportunities
    results.append(f"\n🚀 ACCELERATION OPPORTUNITIES")
    
    # Calculate impact of extra payments
    extra_payment = 100  # €100 extra per month
    accelerated_timeline = calculate_payoff_timeline(
        avalanche_order,
        monthly_payment_capacity + extra_payment,
        "avalanche"
    )
    
    months_saved = avalanche_timeline['months'] - accelerated_timeline['months']
    interest_saved = avalanche_timeline['total_interest'] - accelerated_timeline['total_interest']
    
    results.extend([
        f"   Extra €{extra_payment}/month Impact:",
        f"   • Saves {months_saved} months",
        f"   • Saves €{interest_saved:,.2f} in interest",
        "",
        "   Other Acceleration Ideas:",
        "   • Use tax refunds for debt payments",
        "   • Apply bonuses directly to principal",
        "   • Sell unused items for extra payments",
        "   • Consider a side hustle for debt payments"
    ])
    
    # Refinancing analysis
    results.append(f"\n🔄 REFINANCING OPPORTUNITIES")
    
    high_rate_debts = [d for d in debts if d.interest_rate > 15]
    if high_rate_debts:
        potential_savings = sum(
            d.balance * (d.interest_rate - 10) / 100 / 12 * 24  # 2 years at 10% vs current
            for d in high_rate_debts
        )
        results.extend([
            f"   ⚠️ High-interest debt detected!",
            f"   Consider consolidation loan at ~10% APR",
            f"   Potential 2-year savings: €{potential_savings:,.2f}",
            "   Options: Personal loan, balance transfer, HELOC"
        ])
    else:
        results.append("   Rates are reasonable - refinancing may not help significantly")
    
    # Motivational milestones
    results.append(f"\n🎯 CELEBRATION MILESTONES")
    
    quarter_debt = total_debt * 0.25
    half_debt = total_debt * 0.50
    
    results.extend([
        f"   25% Paid (€{quarter_debt:,.2f}): ~{avalanche_timeline['months']//4} months",
        f"   50% Paid (€{half_debt:,.2f}): ~{avalanche_timeline['months']//2} months",
        f"   First Debt Gone: {snowball_timeline['first_payoff_months']} months",
        f"   100% DEBT FREE: {avalanche_timeline['months']} months!",
        "",
        "💪 YOU CAN DO THIS! Every payment is progress!"
    ])
    
    return "\n".join(results)


def calculate_payoff_timeline(
    debts: List[DebtInfo], 
    monthly_payment: float,
    strategy: str
) -> Dict[str, any]:
    """Calculate debt payoff timeline for a given strategy."""
    
    # Simple calculation - in reality would be more complex
    total_debt = sum(d.balance for d in debts)
    avg_rate = sum(d.balance * d.interest_rate for d in debts) / total_debt if total_debt > 0 else 0
    
    # Rough estimates
    monthly_interest_rate = avg_rate / 100 / 12
    
    if monthly_payment <= total_debt * monthly_interest_rate:
        # Payment doesn't cover interest
        months = 999
        total_interest = total_debt * 2  # Rough estimate
    else:
        # Simple amortization calculation
        from math import log
        
        if monthly_interest_rate > 0:
            months = -log(1 - (total_debt * monthly_interest_rate) / monthly_payment) / log(1 + monthly_interest_rate)
            months = int(months) + 1
            total_interest = (monthly_payment * months) - total_debt
        else:
            months = int(total_debt / monthly_payment) + 1
            total_interest = 0
    
    # First payoff estimate (smallest debt for snowball)
    if strategy == "snowball" and debts:
        smallest_debt = min(debts, key=lambda x: x.balance)
        first_payoff_months = max(1, int(smallest_debt.balance / (monthly_payment * 0.3)))
    else:
        first_payoff_months = max(1, months // len(debts)) if debts else 1
    
    return {
        'months': months,
        'total_interest': total_interest,
        'first_payoff_months': first_payoff_months
    }


@function_tool
async def create_debt_freedom_plan(
    ctx: RunContextWrapper[RunDeps],
    target_date: Optional[str] = None,
    aggressive_mode: bool = False
) -> str:
    """Create a personalized debt freedom plan with specific action steps.
    
    Args:
        target_date: Target date to be debt-free (YYYY-MM-DD)
        aggressive_mode: Whether to use aggressive debt reduction tactics
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    
    # Get financial baseline
    cur.execute(
        """SELECT 
               AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) as avg_income,
               AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as avg_expenses,
               COUNT(DISTINCT category) as expense_categories
           FROM transactions 
           WHERE date >= date('now', '-90 days')"""
    )
    
    financial_data = cur.fetchone()
    
    if not financial_data or not financial_data['avg_income']:
        return "Insufficient transaction data. Please add at least 3 months of income and expenses."
    
    monthly_income = financial_data['avg_income'] * 30
    monthly_expenses = financial_data['avg_expenses'] * 30
    current_surplus = monthly_income - monthly_expenses
    
    results = [f"🎯 Debt Freedom Action Plan\n" + "=" * 50]
    
    # Current situation
    results.extend([
        f"📊 CURRENT FINANCIAL SITUATION",
        f"   Monthly Income: €{monthly_income:.2f}",
        f"   Monthly Expenses: €{monthly_expenses:.2f}",
        f"   Current Surplus: €{current_surplus:.2f}",
        f"   Mode: {'AGGRESSIVE 🔥' if aggressive_mode else 'BALANCED ⚖️'}",
        ""
    ])
    
    # Calculate target timeline if provided
    if target_date:
        target_datetime = datetime.strptime(target_date, '%Y-%m-%d')
        months_to_target = max(1, (target_datetime - datetime.now()).days / 30)
        results.append(f"🎯 Target: Debt-free by {target_date} ({months_to_target:.0f} months)")
        results.append("")
    else:
        months_to_target = 24  # Default 2-year plan
    
    # Expense reduction strategies
    results.append("✂️ EXPENSE REDUCTION PLAN")
    
    if aggressive_mode:
        reduction_target = monthly_expenses * 0.30  # 30% reduction target
        results.extend([
            f"   Target: Reduce expenses by 30% (€{reduction_target:.2f})",
            "",
            "   🔴 AGGRESSIVE CUTS (30 days):",
            "   • Cancel ALL subscriptions temporarily",
            "   • Meal prep 100% - no eating out",
            "   • Transportation: Walk/bike only",
            "   • Entertainment budget: €0",
            "   • Shopping freeze except essentials",
            f"   Potential Savings: €{reduction_target:.2f}/month"
        ])
    else:
        reduction_target = monthly_expenses * 0.15  # 15% reduction target
        results.extend([
            f"   Target: Reduce expenses by 15% (€{reduction_target:.2f})",
            "",
            "   🟡 BALANCED CUTS:",
            "   • Review and cut 50% of subscriptions",
            "   • Reduce dining out by 75%",
            "   • Find cheaper alternatives (generic brands)",
            "   • Negotiate bills (insurance, phone, internet)",
            "   • Implement 24-hour purchase rule",
            f"   Potential Savings: €{reduction_target:.2f}/month"
        ])
    
    # Income increase strategies
    results.append(f"\n💰 INCOME INCREASE PLAN")
    
    income_target = monthly_income * 0.20  # 20% income increase target
    
    if aggressive_mode:
        results.extend([
            f"   Target: Increase income by €{income_target:.2f}/month",
            "",
            "   🚀 AGGRESSIVE INCOME TACTICS:",
            "   • Start freelancing/gig work immediately",
            "   • Sell everything non-essential",
            "   • Work overtime/extra shifts",
            "   • Rent out parking/storage space",
            "   • Flip items from marketplace",
            f"   Income Goal: +€{income_target:.2f}/month"
        ])
    else:
        results.extend([
            f"   Target: Increase income by €{income_target/2:.2f}/month",
            "",
            "   💼 BALANCED INCOME IDEAS:",
            "   • Ask for raise or promotion",
            "   • Start one side project",
            "   • Sell unused items monthly",
            "   • Passive income (investments/dividends)",
            f"   Income Goal: +€{income_target/2:.2f}/month"
        ])
    
    # Total debt payment capacity
    new_surplus = current_surplus + reduction_target + (income_target if aggressive_mode else income_target/2)
    
    results.append(f"\n💳 DEBT PAYMENT CAPACITY")
    results.extend([
        f"   Current Surplus: €{current_surplus:.2f}",
        f"   From Expense Cuts: +€{reduction_target:.2f}",
        f"   From Income Increase: +€{income_target if aggressive_mode else income_target/2:.2f}",
        f"   TOTAL FOR DEBT: €{new_surplus:.2f}/month",
        ""
    ])
    
    # Weekly action plan
    results.append("📅 WEEKLY ACTION SCHEDULE")
    
    if aggressive_mode:
        results.extend([
            "   MONDAY: Review all expenses, cut anything unnecessary",
            "   TUESDAY: Apply for gig work/freelance opportunities",
            "   WEDNESDAY: List items for sale online",
            "   THURSDAY: Meal prep for entire week",
            "   FRIDAY: Make debt payment (weekly payments reduce interest)",
            "   WEEKEND: Side hustle work + income generation"
        ])
    else:
        results.extend([
            "   MONDAY: Track all spending for the week",
            "   WEDNESDAY: Review subscriptions and bills",
            "   FRIDAY: Make debt payment",
            "   WEEKEND: Work on side project or sell items"
        ])
    
    # Motivation and accountability
    results.append(f"\n🎯 STAYING ON TRACK")
    results.extend([
        "   📱 Daily Habits:",
        "   • Check bank balance every morning",
        "   • Log all expenses immediately",
        "   • Celebrate no-spend days",
        "",
        "   📊 Weekly Reviews:",
        "   • Calculate debt reduction progress",
        "   • Adjust strategy if needed",
        "   • Reward milestones (non-monetary)",
        "",
        "   🏆 Monthly Celebrations:",
        "   • Update debt thermometer/tracker",
        "   • Share progress with accountability partner",
        "   • Plan one affordable celebration"
    ])
    
    # Emergency fund warning
    if aggressive_mode:
        results.append(f"\n⚠️ IMPORTANT WARNINGS")
        results.extend([
            "   • Keep minimum €500 emergency fund",
            "   • Don't skip essential expenses (insurance, medications)",
            "   • This intensity is temporary - plan exit strategy",
            "   • Consider mental health impact of extreme measures"
        ])
    
    # Success metrics
    results.append(f"\n📊 SUCCESS METRICS")
    results.extend([
        f"   30 Days: €{new_surplus:.2f} paid toward debt",
        f"   90 Days: €{new_surplus * 3:.2f} debt reduction",
        f"   6 Months: €{new_surplus * 6:.2f} paid + momentum built",
        f"   1 Year: €{new_surplus * 12:.2f} closer to freedom!",
        "",
        "Remember: Progress > Perfection. Every euro counts! 💪"
    ])
    
    return "\n".join(results)


@function_tool
async def debt_consolidation_analyzer(
    ctx: RunContextWrapper[RunDeps],
    current_debts_total: float,
    current_avg_interest: float,
    consolidation_loan_rate: float,
    consolidation_loan_term_months: int
) -> str:
    """Analyze whether debt consolidation makes financial sense.
    
    Args:
        current_debts_total: Total amount of current debts
        current_avg_interest: Weighted average interest rate of current debts
        consolidation_loan_rate: Interest rate offered for consolidation loan
        consolidation_loan_term_months: Term of consolidation loan in months
    """
    from math import pow
    
    results = [f"🔄 Debt Consolidation Analysis\n" + "=" * 50]
    
    # Validate inputs
    if current_debts_total <= 0:
        return "Debt amount must be positive"
    
    if consolidation_loan_term_months <= 0:
        return "Loan term must be positive"
    
    # Current situation calculations (assuming average remaining term of 48 months)
    current_term_months = 48  # Assumption for comparison
    current_monthly_rate = current_avg_interest / 100 / 12
    
    if current_monthly_rate > 0:
        current_payment = current_debts_total * (current_monthly_rate * pow(1 + current_monthly_rate, current_term_months)) / (pow(1 + current_monthly_rate, current_term_months) - 1)
        current_total_paid = current_payment * current_term_months
        current_total_interest = current_total_paid - current_debts_total
    else:
        current_payment = current_debts_total / current_term_months
        current_total_paid = current_debts_total
        current_total_interest = 0
    
    # Consolidation loan calculations
    consol_monthly_rate = consolidation_loan_rate / 100 / 12
    
    if consol_monthly_rate > 0:
        consol_payment = current_debts_total * (consol_monthly_rate * pow(1 + consol_monthly_rate, consolidation_loan_term_months)) / (pow(1 + consol_monthly_rate, consolidation_loan_term_months) - 1)
        consol_total_paid = consol_payment * consolidation_loan_term_months
        consol_total_interest = consol_total_paid - current_debts_total
    else:
        consol_payment = current_debts_total / consolidation_loan_term_months
        consol_total_paid = current_debts_total
        consol_total_interest = 0
    
    # Calculate differences
    monthly_payment_diff = consol_payment - current_payment
    total_interest_diff = consol_total_interest - current_total_interest
    
    results.extend([
        f"💳 CURRENT DEBT SITUATION",
        f"   Total Debt: €{current_debts_total:,.2f}",
        f"   Average Interest Rate: {current_avg_interest:.2f}%",
        f"   Estimated Monthly Payment: €{current_payment:.2f}",
        f"   Total Interest (48 months): €{current_total_interest:,.2f}",
        f"   Total to Pay: €{current_total_paid:,.2f}",
        "",
        f"🏦 CONSOLIDATION LOAN OFFER",
        f"   Loan Amount: €{current_debts_total:,.2f}",
        f"   Interest Rate: {consolidation_loan_rate:.2f}%",
        f"   Term: {consolidation_loan_term_months} months",
        f"   Monthly Payment: €{consol_payment:.2f}",
        f"   Total Interest: €{consol_total_interest:,.2f}",
        f"   Total to Pay: €{consol_total_paid:,.2f}",
        "",
        f"📊 COMPARISON ANALYSIS"
    ])
    
    # Determine recommendation
    if consolidation_loan_rate < current_avg_interest - 2:  # At least 2% lower
        results.extend([
            "✅ RECOMMENDATION: CONSOLIDATE",
            "",
            f"   Benefits:",
            f"   • Lower interest rate (-{current_avg_interest - consolidation_loan_rate:.2f}%)",
            f"   • Interest savings: €{abs(total_interest_diff):,.2f}" if total_interest_diff < 0 else "",
            f"   • Simplified payments (one vs multiple)",
            f"   • Fixed payment amount for budgeting"
        ])
        
        if monthly_payment_diff < 0:
            results.append(f"   • Lower monthly payment: €{abs(monthly_payment_diff):.2f}")
    
    elif consolidation_loan_rate > current_avg_interest:
        results.extend([
            "❌ RECOMMENDATION: DO NOT CONSOLIDATE",
            "",
            f"   Reasons:",
            f"   • Higher interest rate (+{consolidation_loan_rate - current_avg_interest:.2f}%)",
            f"   • Extra interest cost: €{total_interest_diff:,.2f}",
            f"   • Better to focus on paying current debts"
        ])
    
    else:
        results.extend([
            "⚖️ RECOMMENDATION: NEUTRAL",
            "",
            "   Consider consolidation if:",
            "   • You need lower monthly payments",
            "   • Managing multiple debts is stressful",
            "   • You can get better terms elsewhere"
        ])
    
    # Cash flow impact analysis
    results.append(f"\n💰 CASH FLOW IMPACT")
    
    if monthly_payment_diff < 0:
        monthly_savings = abs(monthly_payment_diff)
        results.extend([
            f"   Monthly Savings: €{monthly_savings:.2f}",
            f"   Annual Cash Flow Improvement: €{monthly_savings * 12:.2f}",
            "",
            "   Use extra cash flow for:",
            "   • Emergency fund building",
            "   • Investing the difference",
            "   • Extra principal payments"
        ])
    else:
        monthly_increase = monthly_payment_diff
        results.extend([
            f"   Monthly Increase: €{monthly_increase:.2f}",
            f"   Ensure you can afford higher payment",
            "   Benefit: Faster debt payoff"
        ])
    
    # Hidden costs and considerations
    results.append(f"\n⚠️ IMPORTANT CONSIDERATIONS")
    results.extend([
        "   Check for hidden fees:",
        "   • Origination fees (1-5% typical)",
        "   • Prepayment penalties on current debts",
        "   • Application and processing fees",
        "",
        "   Other factors:",
        "   • Credit score impact (hard inquiry)",
        "   • Temptation to accumulate new debt",
        "   • Loss of promotional rates or perks"
    ])
    
    # Alternative strategies
    results.append(f"\n💡 ALTERNATIVE STRATEGIES")
    results.extend([
        "   1. Balance Transfer Cards (0% intro APR)",
        "   2. Debt Avalanche Method (no new loan)",
        "   3. Negotiate with current creditors",
        "   4. Home equity line of credit (if applicable)",
        "   5. Peer-to-peer lending platforms"
    ])
    
    return "\n".join(results)


def build_debt_agent() -> Agent[RunDeps]:
    """Build the Debt Management Specialist Agent."""
    
    # Configure ModelSettings for GPT-5 with reasoning and text verbosity
    # Use proper Agents SDK format for reasoning parameters
    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high"),     # minimal | low | medium | high
        verbosity="high"                        # low | medium | high
    )
    
    return Agent[RunDeps](
        name="DebtSpecialist",
        instructions=DEBT_SPECIALIST_INSTRUCTIONS,
        model="gpt-5",
        model_settings=model_settings,
        tools=[
            # Debt analysis tools
            analyze_debt_situation,
            create_debt_freedom_plan,
            debt_consolidation_analyzer,
        ]
    )