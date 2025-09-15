TAX_SPECIALIST_INSTRUCTIONS = """You are a Tax Specialist - an expert in personal and business tax optimization, deductions, and compliance.
Your expertise includes:
• Identifying tax-deductible expenses and categorizing transactions
• Generating comprehensive tax reports with proper categorization
• Advising on tax optimization strategies and timing
• Explaining tax implications of financial decisions
• Preparing documentation for tax filing and audits
Key principles:
- Always prioritize tax law compliance and accuracy
- Suggest legitimate deduction opportunities the user might miss
- Explain tax implications in simple, actionable terms
- Recommend timing strategies for income and expenses
- Help organize financial records for tax preparation
When analyzing transactions, automatically categorize them for tax purposes:
- Business expenses (office supplies, equipment, travel)
- Medical expenses and healthcare costs
- Charitable donations and contributions
- Educational expenses and training
- Home office and utility deductions
- Investment-related fees and expenses
Always provide specific, quantified advice with dollar amounts and percentages when possible.
TEAM COORDINATION:
- Work with Investment Specialist on tax-efficient investment strategies
- Support Goal Specialist with tax-optimized savings timing
- Assist Budget Specialist with deductible expense categorization
- Collaborate with Debt Specialist on tax implications of debt strategies
- Always consider tax consequences when other specialists make recommendations"""
@function_tool
async def analyze_tax_deductions(
    ctx: RunContextWrapper[RunDeps],
    year: int,
    include_estimates: bool = True,
) -> str:
    """Analyze transactions to identify potential tax deductions and optimization opportunities.

    Args:
        year: Tax year to analyze
        include_estimates: Whether to include estimated tax savings
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get transactions for the tax year

    if not transactions:
        return f"No expenses found for tax year {year}"

    # Categorize potential deductions
    deduction_categories = {
        "business_expenses": {
            "keywords": ["office", "business", "equipment", "software", "supplies"],
            "total": 0,
            "transactions": [],
            "description": "Business Equipment & Supplies",
        },
        "medical_expenses": {
            "keywords": ["medical", "health", "dental", "pharmacy", "doctor"],
            "total": 0,
            "transactions": [],
            "description": "Medical & Healthcare",
        },
        "charitable": {
            "keywords": ["donation", "charity", "nonprofit", "church", "foundation"],
            "total": 0,
            "transactions": [],
            "description": "Charitable Contributions",
        },
        "education": {
            "keywords": ["education", "training", "course", "tuition", "books"],
            "total": 0,
            "transactions": [],
            "description": "Education & Training",
        },
        "professional": {
            "keywords": [
                "professional",
                "license",
                "membership",
                "conference",
                "networking",
            ],
            "total": 0,
            "transactions": [],
            "description": "Professional Development",
        },
        "home_office": {
            "keywords": ["internet", "phone", "utilities"],
            "total": 0,
            "transactions": [],
            "description": "Home Office Expenses",
        },
    }

    # Categorize transactions
    total_potential_deductions = 0

    for tx in transactions:
        description = tx["description"].lower()
        category = (tx["category"] or "").lower()

        for _, data in deduction_categories.items():
            if any(
                keyword in description or keyword in category
                for keyword in data["keywords"]
            ):
                amount = abs(tx["amount"])
                data["total"] += amount
                data["transactions"].append(tx)
                total_potential_deductions += amount
                break

    # Generate analysis report
    results = [f"🏛️ Tax Deduction Analysis for {year}\n" + "=" * 50]

    # Summary
    estimated_savings = (
        total_potential_deductions * 0.24
    )  # Rough 24% tax bracket estimate
    results.append(f"💰 Total Potential Deductions: €{total_potential_deductions:.2f}")

    if include_estimates:
        results.append(
            f"💡 Estimated Tax Savings: €{estimated_savings:.2f} (24% bracket)"
        )

    results.append("\n📊 DEDUCTION CATEGORIES")

    # Details by category
    for _, data in deduction_categories.items():
        if data["total"] > 0:
            results.append(f"\n📁 {data['description']}")
            results.append(f"   Total: €{data['total']:.2f}")
            results.append(f"   Transactions: {len(data['transactions'])}")

            # Show top 3 transactions in this category
            top_transactions = sorted(
                data["transactions"], key=lambda x: abs(x["amount"]), reverse=True
            )[:3]
            for tx in top_transactions:
                results.append(
                    f"   • {tx['date']}: {tx['description'][:40]} - €{abs(tx['amount']):.2f}"
                )

    # Tax optimization suggestions
    results.append("\n💡 TAX OPTIMIZATION OPPORTUNITIES")

    if deduction_categories["business_expenses"]["total"] > 500:
        results.append(
            "• Consider tracking business use percentage for mixed-use items"
        )
        results.append("• Keep receipts for all business equipment purchases")

    if deduction_categories["home_office"]["total"] > 0:
        results.append("• Calculate home office deduction based on square footage")
        results.append("• Consider dedicated business phone line for better deduction")

    if deduction_categories["medical_expenses"]["total"] > 1000:
        results.append("• Medical expenses over 7.5% of AGI may be deductible")
        results.append("• Consider HSA contributions for future tax benefits")

    results.append("\n📋 NEXT STEPS")
    results.append("• Generate detailed tax report using generate_tax_report")
    results.append("• Organize receipts and documentation for each category")
    results.append("• Consult tax professional for complex deductions")

    return "\n".join(results)


@function_tool
async def suggest_tax_timing(
    ctx: RunContextWrapper[RunDeps], current_month: Optional[int] = None
) -> str:
    """Suggest tax-optimized timing for income and expenses based on current financial position.

    Args:
        current_month: Current month (1-12), defaults to current month
    """
    from datetime import datetime

    if current_month is None:
        current_month = datetime.now().month

    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get current year spending patterns
    current_year = datetime.now().year


    suggestions = [f"📅 Tax Timing Strategy (Month {current_month})\n" + "=" * 50]

    # Year-end strategies (October-December)
    if current_month >= 10:
        suggestions.extend(
            [
                "\n🎯 YEAR-END TAX STRATEGIES",
                "",
                "🔴 URGENT - Do Before Dec 31:",
                "• Maximize business equipment purchases (Section 179 deduction)",
                "• Make charitable contributions (get receipt before Dec 31)",
                "• Pay outstanding business expenses and professional fees",
                "• Consider tax-loss harvesting for investments",
                "",
                "💡 Income Deferral:",
                "• Delay invoicing clients until January if beneficial",
                "• Defer bonus payments to next year if in high bracket",
                "• Consider retirement account contributions",
            ]
        )

        # Check for large deductible categories
        cur.execute(
            """SELECT category, SUM(ABS(amount)) as total
               FROM transactions
               WHERE date >= ? AND amount < 0 AND category IN ('business', 'medical', 'charity')
               GROUP BY category
               ORDER BY total DESC""",
            (f"{current_year}-01-01",),
        )

        categories = cur.fetchall()
        if categories:
            suggestions.append("\n📊 Your Current Deduction Categories:")
            for cat in categories:
                suggestions.append(
                    f"   • {cat['category'].title()}: €{cat['total']:.2f}"
                )

    # Mid-year planning (June-September)
    elif 6 <= current_month <= 9:
        suggestions.extend(
            [
                "\n🎯 MID-YEAR TAX PLANNING",
                "",
                "📊 Review & Adjust:",
                "• Check if estimated quarterly taxes are on track",
                "• Review year-to-date deductions vs. last year",
                "• Plan major purchases for optimal tax timing",
                "",
                "💰 Strategic Moves:",
                "• Consider bunching charitable contributions",
                "• Plan equipment purchases for business",
                "• Review retirement contribution limits",
            ]
        )

    # Early year (January-May)
    else:
        suggestions.extend(
            [
                "\n🎯 EARLY YEAR TAX PLANNING",
                "",
                "📋 Current Year Setup:",
                "• Set up better expense tracking systems",
                "• Open tax-advantaged accounts (HSA, retirement)",
                "• Plan major deductible expenses throughout the year",
                "",
                "📊 Last Year Review:",
                "• Gather all tax documents and receipts",
                "• Review last year's return for missed deductions",
                "• Plan to avoid last year's tax surprises",
            ]
        )

    # Add current year summary if available
    if year_summary and year_summary["transactions"] > 0:
        income = year_summary["income"] or 0
        expenses = year_summary["expenses"] or 0

        suggestions.extend(
            [
                f"\n📈 {current_year} YEAR-TO-DATE SUMMARY",
                f"   Income: €{income:.2f}",
                f"   Expenses: €{expenses:.2f}",
                f"   Transactions: {year_summary['transactions']}",
            ]
        )

        # Provide specific advice based on numbers
        if expenses > income * 0.15:  # High expense ratio
            suggestions.append(
                "   💡 High expense ratio - ensure proper categorization for deductions"
            )

        if income > 50000:  # Higher income bracket
            suggestions.append("   🎯 Consider tax-deferral strategies for high income")

    suggestions.extend(
        [
            "",
            "⚠️ IMPORTANT REMINDER:",
            "Tax laws change frequently. Consult a tax professional for personalized advice.",
            "This analysis is for educational purposes and not professional tax advice.",
        ]







from __future__ import annotations

    )
    return "\n".join(suggestions)

from __future__ import annotations
from typing import Optional

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from openai.types.shared import Reasoning

from ..context import RunDeps
from ..tools.export import generate_tax_report, export_transactions
from .agent_factory import build_specialist_agent
    cur.execute(
        """SELECT * FROM transactions
           WHERE date >= ? AND date <= ?
           AND amount < 0
           ORDER BY category, ABS(amount) DESC""",
        (f"{year}-01-01", f"{year}-12-31"),
    )
    transactions = cur.fetchall()
    cur.execute(
        """SELECT
               SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as income,
               SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as expenses,
               COUNT(*) as transactions
           FROM transactions
           WHERE date >= ? AND date <= ?""",
        (f"{current_year}-01-01", f"{current_year}-12-31"),
    )
    year_summary = cur.fetchone()
def build_tax_agent() -> Agent[RunDeps]:
    """Build the Tax Specialist Agent."""

    return build_specialist_agent(
        name="TaxSpecialist",
        instructions=TAX_SPECIALIST_INSTRUCTIONS,
        tools=[
            generate_tax_report,
            analyze_tax_deductions,
            suggest_tax_timing,
            export_transactions,
        ],
    )