"""Orchestrator agent with handoffs to specialized agents."""
from __future__ import annotations
from agents import Agent, ModelSettings, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters

from .context import RunDeps
from .analysis_agent import build_analysis_agent, build_spending_analyzer, build_summary_agent
from .agent import build_agent


def build_orchestrator_agent() -> Agent[RunDeps]:
    """Build the main orchestrator agent with handoffs to specialized agents."""
    
    # Create specialized agents
    main_agent = build_agent()
    analysis_agent = build_analysis_agent()
    spending_agent = build_spending_analyzer()
    summary_agent = build_summary_agent()
    
    instructions = f"""{RECOMMENDED_PROMPT_PREFIX}
    
    You are the Financial Orchestrator. You help users with their financial queries by delegating to specialized agents:
    
    1. For general queries, transaction ingestion, and basic searches - handle yourself
    2. For budget advice and recommendations - handoff to the Analysis Agent
    3. For detailed spending analysis - handoff to the Spending Analyzer
    4. For comprehensive financial summaries - handoff to the Summary Agent
    
    Always choose the most appropriate agent for the task. If unsure, handle it yourself first.
    """
    
    return Agent[RunDeps](
        name="FinancialOrchestrator",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        handoffs=[
            handoff(
                agent=main_agent,
                tool_name_override="general_assistant",
                tool_description_override="Handle general queries, file ingestion, and basic transaction searches",
                input_filter=handoff_filters.remove_all_tools,  # Clean history for general agent
            ),
            handoff(
                agent=analysis_agent,
                tool_name_override="budget_advisor",
                tool_description_override="Provide structured budget advice and financial recommendations",
                on_handoff=log_handoff("Budget Analysis"),
            ),
            handoff(
                agent=spending_agent,
                tool_name_override="spending_analyzer",
                tool_description_override="Analyze spending patterns and identify trends",
                on_handoff=log_handoff("Spending Analysis"),
            ),
            handoff(
                agent=summary_agent,
                tool_name_override="financial_summarizer",
                tool_description_override="Create comprehensive financial summaries with income, expenses, and insights",
                on_handoff=log_handoff("Financial Summary"),
            ),
        ],
    )


def log_handoff(agent_type: str):
    """Create a logging function for handoffs."""
    def _log(ctx: RunContextWrapper[RunDeps]):
        print(f"\n🔄 Handing off to {agent_type} agent...")
    return _log


def build_budget_expert() -> Agent[RunDeps]:
    """Build a specialized budget expert agent."""
    
    instructions = """You are a Budget Expert. Focus on:
    1. Analyzing income vs expenses
    2. Identifying savings opportunities
    3. Creating realistic budget plans
    4. Providing specific percentage-based recommendations
    
    Always be specific with numbers and provide actionable steps."""
    
    return Agent[RunDeps](
        name="BudgetExpert",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        tools=[],  # Will inherit tools from parent context
    )


def build_investment_advisor() -> Agent[RunDeps]:
    """Build a specialized investment advisor agent."""
    
    instructions = """You are an Investment Advisor. Focus on:
    1. Analyzing surplus funds available for investment
    2. Suggesting appropriate investment strategies based on spending patterns
    3. Identifying regular savings potential
    4. Risk assessment based on financial stability
    
    Provide conservative, moderate, and aggressive options."""
    
    return Agent[RunDeps](
        name="InvestmentAdvisor",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        tools=[],  # Will inherit tools from parent context
    )