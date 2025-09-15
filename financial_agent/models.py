"""Structured data models for financial agent outputs."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


class Transaction(BaseModel):
    """Structured transaction data."""

    date: Optional[str] = Field(description="Transaction date in ISO format")
    description: str = Field(description="Transaction description")
    amount: float = Field(description="Transaction amount")
    currency: str = Field(default="EUR", description="Currency code")
    category: Optional[str] = Field(description="Transaction category")
    source_file: Optional[str] = Field(description="Source file name")


class TransactionList(BaseModel):
    """List of transactions with metadata."""

    transactions: List[Transaction]
    count: int = Field(description="Total number of transactions")
    total_amount: float = Field(description="Sum of all transaction amounts")
    date_range: Optional[str] = Field(description="Date range of transactions")


class BudgetAdvice(BaseModel):
    """Structured budget advice output."""

    summary: str = Field(description="Executive summary of financial situation")
    insights: List[str] = Field(description="Key financial insights")
    recommendations: List[str] = Field(description="Actionable recommendations")
    warnings: Optional[List[str]] = Field(description="Financial warnings if any")
    savings_potential: Optional[float] = Field(
        description="Estimated monthly savings potential"
    )


class SpendingAnalysis(BaseModel):
    """Detailed spending analysis output."""

    total_spent: float = Field(description="Total amount spent")
    average_daily: float = Field(description="Average daily spending")
    top_categories: List[CategorySpending] = Field(
        description="Top spending categories"
    )
    unusual_transactions: Optional[List[Transaction]] = Field(
        description="Unusual or large transactions"
    )
    trend: str = Field(description="Spending trend: increasing, decreasing, or stable")


class CategorySpending(BaseModel):
    """Spending by category."""

    category: str = Field(description="Category name")
    amount: float = Field(description="Total amount spent")
    percentage: float = Field(description="Percentage of total spending")
    transaction_count: int = Field(description="Number of transactions")


class FinancialSummary(BaseModel):
    """Comprehensive financial summary."""

    period: str = Field(description="Time period covered")
    total_income: float = Field(description="Total income")
    total_expenses: float = Field(description="Total expenses")
    net_savings: float = Field(description="Net savings or deficit")
    expense_breakdown: List[CategorySpending] = Field(
        description="Breakdown by category"
    )
    key_insights: List[str] = Field(description="Important observations")
    next_steps: List[str] = Field(description="Recommended actions")
