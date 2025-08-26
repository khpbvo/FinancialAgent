"""Specialized analysis agent with structured outputs."""
from __future__ import annotations
from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from typing import Any

from .context import RunDeps
from .models import BudgetAdvice, SpendingAnalysis, FinancialSummary
from .tools.query import list_recent_transactions, search_transactions


def build_analysis_agent() -> Agent[RunDeps]:
    """Create an analysis agent with structured output."""
    
    instructions = """You are a financial analysis expert. Analyze the user's financial data and provide structured insights.
    Always be specific with numbers and percentages. Focus on actionable recommendations."""
    
    return Agent[RunDeps](
        name="FinancialAnalysisAgent",
        instructions=instructions,
        model="gpt-4o",  # Use a stable model for structured outputs
        model_settings=ModelSettings(),
        output_type=BudgetAdvice,  # Structured output
        tools=[list_recent_transactions, search_transactions],
    )


def build_spending_analyzer() -> Agent[RunDeps]:
    """Create a spending analysis agent with structured output."""
    
    instructions = """You are a spending analysis specialist. Analyze transaction patterns and provide detailed spending insights.
    Calculate exact amounts and percentages. Identify unusual patterns and trends."""
    
    return Agent[RunDeps](
        name="SpendingAnalyzer",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        output_type=SpendingAnalysis,  # Structured output
        tools=[list_recent_transactions, search_transactions],
    )


def build_summary_agent() -> Agent[RunDeps]:
    """Create a financial summary agent with structured output."""
    
    instructions = """You are a financial summary expert. Create comprehensive summaries of financial situations.
    Include income, expenses, savings, and provide actionable next steps."""
    
    return Agent[RunDeps](
        name="SummaryAgent",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        output_type=FinancialSummary,  # Structured output
        tools=[list_recent_transactions, search_transactions],
    )