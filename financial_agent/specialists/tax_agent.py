from __future__ import annotations

from typing import Optional

from agents import Agent, RunContextWrapper, function_tool

from ..context import RunDeps
from ..tools.export import generate_tax_report, export_transactions
from .agent_factory import build_specialist_agent


TAX_SPECIALIST_INSTRUCTIONS = """You are a Tax Specialist focused on legitimate deduction discovery, tax timing, and compliance.

Principles:
- Prioritize compliance and accuracy
- Identify missed deductions and document requirements
- Provide clear, quantified guidance in simple terms
"""


@function_tool
async def analyze_tax_deductions(
    ctx: RunContextWrapper[RunDeps],
    year: int,
    include_estimates: bool = True,
) -> str:
    """Analyze transactions to identify potential tax deductions and opportunities."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute(
        """SELECT * FROM transactions
           WHERE date >= ? AND date <= ? AND amount < 0
           ORDER BY category, ABS(amount) DESC""",
        (f"{year}-01-01", f"{year}-12-31"),
    )
    transactions = cur.fetchall()
    if not transactions:
        return f"No expenses found for tax year {year}"

    deduction_categories = {
        "business_expenses": {
            "keywords": ["office", "business", "equipment", "software", "supplies"],
            "total": 0.0,
            "transactions": [],
            "description": "Business Equipment & Supplies",
        },
        "medical_expenses": {
            "keywords": ["medical", "health", "dental", "pharmacy", "doctor"],
            "total": 0.0,
            "transactions": [],
            "description": "Medical & Healthcare",
        },
        "charitable": {
            "keywords": ["donation", "charity", "nonprofit", "church", "foundation"],
            "total": 0.0,
            "transactions": [],
            "description": "Charitable Contributions",
        },
        "education": {
            "keywords": ["education", "training", "course", "tuition", "books"],
            "total": 0.0,
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
            "total": 0.0,
            "transactions": [],
            "description": "Professional Development",
        },
        "home_office": {
            "keywords": ["internet", "phone", "utilities"],
            "total": 0.0,
            "transactions": [],
            "description": "Home Office Expenses",
        },
    }

    total_potential_deductions = 0.0
    for tx in transactions:
        description = (tx["description"] or "").lower()
        category = (tx["category"] or "").lower()
        for data in deduction_categories.values():
            if any(k in description or k in category for k in data["keywords"]):
                amount = abs(tx["amount"])
                data["total"] += amount
                data["transactions"].append(tx)
                total_potential_deductions += amount
                break

    results = [f"🏛️ Tax Deduction Analysis for {year}\n" + "=" * 50]
    results.append(f"💰 Total Potential Deductions: €{total_potential_deductions:.2f}")
    if include_estimates:
        estimated_savings = total_potential_deductions * 0.24  # Rough 24% bracket
        results.append(f"💡 Estimated Tax Savings: €{estimated_savings:.2f}")

    results.append("\n📊 DEDUCTION CATEGORIES")
    for data in deduction_categories.values():
        if data["total"] > 0:
            results.append(f"\n📁 {data['description']}")
            results.append(f"   Total: €{data['total']:.2f}")
            results.append(f"   Transactions: {len(data['transactions'])}")
            top_tx = sorted(
                data["transactions"], key=lambda x: abs(x["amount"]), reverse=True
            )[:3]
            for tx in top_tx:
                results.append(
                    f"   • {tx['date']}: {tx['description'][:40]} - €{abs(tx['amount']):.2f}"
                )

    results.append("\n💡 TAX OPTIMIZATION OPPORTUNITIES")
    if deduction_categories["business_expenses"]["total"] > 500:
        results.append("• Track business-use % for mixed-use items; keep receipts")
    if deduction_categories["home_office"]["total"] > 0:
        results.append("• Calculate home office deduction by square footage")
    if deduction_categories["medical_expenses"]["total"] > 1000:
        results.append("• Expenses over 7.5% of AGI may be deductible; consider HSA")

    results.append("\n📋 NEXT STEPS")
    results.append("• Generate detailed tax report using generate_tax_report")
    results.append("• Organize receipts and documentation for each category")
    results.append("• Consult a tax professional for complex situations")

    return "\n".join(results)


@function_tool
async def suggest_tax_timing(
    ctx: RunContextWrapper[RunDeps], current_month: Optional[int] = None
) -> str:
    """Suggest tax-optimized timing for income and expenses based on current financial position."""
    from datetime import datetime as _dt

    if current_month is None:
        current_month = _dt.now().month

    deps = ctx.context
    cur = deps.db.conn.cursor()
    current_year = _dt.now().year

    suggestions = [f"📅 Tax Timing Strategy (Month {current_month})\n" + "=" * 50]

    if current_month >= 10:
        suggestions.extend(
            [
                "\n🎯 YEAR-END TAX STRATEGIES",
                "🔴 Do before Dec 31:",
                "• Consider equipment purchases (Section 179; verify eligibility)",
                "• Make charitable contributions (get receipts)",
                "• Prepay deductible expenses where appropriate",
                "• Consider tax-loss harvesting (investments)",
            ]
        )

        cur.execute(
            """SELECT category, SUM(ABS(amount)) AS total
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
    elif 6 <= current_month <= 9:
        suggestions.extend(
            [
                "\n🎯 MID-YEAR TAX PLANNING",
                "• Review estimated quarterly taxes",
                "• Compare YTD deductions vs last year",
                "• Plan purchases for optimal timing",
            ]
        )
    else:
        suggestions.extend(
            [
                "\n🎯 EARLY YEAR TAX PLANNING",
                "• Improve expense tracking systems",
                "• Open/Review tax-advantaged accounts (HSA, retirement)",
                "• Plan deductible expenses across the year",
            ]
        )

    cur.execute(
        """SELECT
               SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS income,
               SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS expenses,
               COUNT(*) AS transactions
           FROM transactions
           WHERE date >= ? AND date <= ?""",
        (f"{current_year}-01-01", f"{current_year}-12-31"),
    )
    year_summary = cur.fetchone()
    if year_summary and year_summary["transactions"] > 0:
        income = year_summary["income"] or 0.0
        expenses = year_summary["expenses"] or 0.0
        suggestions.extend(
            [
                f"\n📈 {current_year} YEAR-TO-DATE SUMMARY",
                f"   Income: €{income:.2f}",
                f"   Expenses: €{expenses:.2f}",
                f"   Transactions: {year_summary['transactions']}",
            ]
        )
        if expenses > income * 0.15:
            suggestions.append(
                "   💡 High expense ratio — ensure proper categorization for deductions"
            )
        if income > 50000:
            suggestions.append("   🎯 Consider deferral strategies for higher income")

    suggestions.extend(
        [
            "",
            "⚠️ IMPORTANT: This is educational guidance, not tax advice. Consult a professional.",
        ]
    )

    return "\n".join(suggestions)


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
