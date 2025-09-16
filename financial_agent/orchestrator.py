from __future__ import annotations
from typing import Any, Dict, List, Optional, cast
from agents import Agent, ModelSettings, function_tool, RunContextWrapper, Runner
from openai.types.shared import Reasoning
from .context import RunDeps
from .logging_utils import get_logger
from .specialists.tax_agent import build_tax_agent
from .specialists.budget_agent import build_budget_agent
from .specialists.goal_agent import build_goal_agent
from .specialists.investment_agent import build_investment_agent
from .specialists.debt_agent import build_debt_agent
from .logged_tools import (
    logged_ingest_csv as ingest_csv,
    logged_ingest_excel as ingest_excel,
    logged_list_excel_sheets as list_excel_sheets,
    logged_list_recent_transactions as list_recent_transactions,
    logged_search_transactions as search_transactions,
    logged_ingest_pdfs as ingest_pdfs,
    logged_list_memories as list_memories,
    logged_summarize_file as summarize_file,
    logged_summarize_overview as summarize_overview,
    logged_add_transaction as add_transaction,
    logged_export_transactions as export_transactions,
    logged_export_recurring_payments as export_recurring_payments,
    logged_export_clean_monthly_recurring as export_clean_monthly_recurring,
    logged_monthly_cost_summary as monthly_cost_summary,
)

# from .debug_analyzer import create_debug_tool  # TODO: Fix type issues


async def _logged_handoff(
    specialist_name: str,
    specialist_agent: Agent[RunDeps],
    user_query: str,
    additional_context: Optional[str],
    deps: RunDeps,
    emoji: str,
    routing_reason: str,
) -> str:
    """Helper function to execute specialist handoffs with comprehensive logging."""
    logger = get_logger()

    # Log routing decision
    logger.log_orchestrator_route(user_query, specialist_name, routing_reason)

    # Prepare query with context
    full_query = user_query
    if additional_context:
        full_query += f"\n\nAdditional context: {additional_context}"

    # Log handoff details
    logger.log_handoff(
        "Orchestrator",
        specialist_name,
        {
            "query": user_query,
            "has_additional_context": bool(additional_context),
            "full_query_length": len(full_query),
            "routing_reason": routing_reason,
        },
    )

    # Execute specialist agent with logging
    agent_context = {
        "handoff_from": "Orchestrator",
        "specialist_type": specialist_name.lower().replace(" ", "_"),
    }

    try:
        with logger.log_agent_execution(
            specialist_name, full_query, agent_context
        ) as agent_id:
            result = await Runner.run(specialist_agent, full_query, context=deps)
            logger.log_agent_complete(agent_id, result.final_output)

        return f"{emoji} **{specialist_name} Response:**\n\n{result.final_output}"

    except Exception as e:
        logger.log_error(f"Specialist handoff failed: {specialist_name}", error=e)
        raise


ORCHESTRATOR_INSTRUCTIONS = """You are the Financial Agent Orchestrator - the intelligent coordinator of specialized financial agents.

Your role is to:
• Understand user requests and determine which specialist(s) can best help
• Route complex queries to the appropriate specialist agents  
• Coordinate multiple agents for comprehensive financial advice
• Handle general queries and data ingestion directly
• Synthesize insights from multiple specialists when needed

Specialist Agents Available:
1. **Tax Specialist** - Tax optimization, deductions, compliance, tax reports
2. **Budget Specialist** - Spending analysis, budget management, subscription optimization
3. **Goal Specialist** - Financial planning, savings goals, motivation coaching
4. **Investment Specialist** - Portfolio analysis, investment readiness, wealth building
5. **Debt Specialist** - Debt elimination, consolidation, credit improvement

Routing Guidelines:
• Tax questions → Tax Specialist
• Budget, spending, subscription questions → Budget Specialist  
• Financial goals, savings plans, motivation → Goal Specialist
• Investment, portfolio, wealth building → Investment Specialist
• Debt, loans, credit questions → Debt Specialist
• Data ingestion (CSV/PDF/Excel) → Handle directly
• Monthly spending/cost questions → Use monthly_cost_summary first (fast, deterministic)
• Multi-domain questions → Use multiple specialists and synthesize

Key principles:
- Route to specialists for expert-level advice in their domain
- Handle simple data queries yourself (transactions, summaries)
- When in doubt, explain options and let user choose
- Always provide specific, actionable advice
- Coordinate specialists for comprehensive planning
- Use multi-specialist analysis for complex financial situations
- Synthesize specialist recommendations into coherent action plans

You have access to core tools for data management and can handoff to specialists for advanced analysis.

COORDINATION NOTES:
- For holistic financial planning: combine Goal + Investment + Tax specialists
- For debt-to-wealth transitions: coordinate Debt + Investment + Goal specialists  
- For spending optimization: combine Budget + Tax specialists
- Always consider tax implications when coordinating Investment or Goal specialists"""


@function_tool
async def handoff_to_tax_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None,
) -> str:
    """Handoff tax-related queries to the Tax Specialist agent.

    Args:
        user_query: The user's tax-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    return await _logged_handoff(
        "Tax Specialist",
        build_tax_agent(),
        user_query,
        additional_context,
        ctx.context,
        "🏛️",
        "User query contains tax-related terms or requirements",
    )


@function_tool
async def handoff_to_budget_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None,
) -> str:
    """Handoff budget and spending analysis queries to the Budget Specialist agent.

    Args:
        user_query: The user's budget/spending-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    return await _logged_handoff(
        "Budget Specialist",
        build_budget_agent(),
        user_query,
        additional_context,
        ctx.context,
        "💰",
        "User query contains budget or spending analysis requirements",
    )


@function_tool
async def handoff_to_goal_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None,
) -> str:
    """Handoff financial goal and planning queries to the Goal Specialist agent.

    Args:
        user_query: The user's goal/planning-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    return await _logged_handoff(
        "Goal Specialist",
        build_goal_agent(),
        user_query,
        additional_context,
        ctx.context,
        "🎯",
        "User query contains goal setting or financial planning requirements",
    )


@function_tool
async def handoff_to_investment_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None,
) -> str:
    """Handoff investment and portfolio queries to the Investment Specialist agent.

    Args:
        user_query: The user's investment-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    return await _logged_handoff(
        "Investment Specialist",
        build_investment_agent(),
        user_query,
        additional_context,
        ctx.context,
        "📈",
        "User query contains investment or portfolio analysis requirements",
    )


@function_tool
async def handoff_to_debt_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None,
) -> str:
    """Handoff debt and loan queries to the Debt Management Specialist agent.

    Args:
        user_query: The user's debt-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    return await _logged_handoff(
        "Debt Specialist",
        build_debt_agent(),
        user_query,
        additional_context,
        ctx.context,
        "💳",
        "User query contains debt management or loan-related requirements",
    )


@function_tool
async def coordinate_multi_specialist_analysis(
    ctx: RunContextWrapper[RunDeps], user_query: str, specialists_needed: List[str]
) -> str:
    """Coordinate multiple specialists for comprehensive financial analysis.

    Args:
        user_query: The user's complex query requiring multiple specialists
        specialists_needed: List of specialists to involve (tax, budget, goal)
    """
    deps = ctx.context

    results = []
    results.append("🤝 **Multi-Specialist Analysis**\n")
    results.append(f"Query: {user_query}\n")
    results.append("=" * 50)

    # Run each requested specialist
    if "tax" in specialists_needed:
        tax_agent = build_tax_agent()
        tax_result = await Runner.run(tax_agent, user_query, context=deps)
        results.append("\n🏛️ **Tax Specialist Perspective:**")
        results.append(str(tax_result.final_output))

    if "budget" in specialists_needed:
        budget_agent = build_budget_agent()
        budget_result = await Runner.run(budget_agent, user_query, context=deps)
        results.append("\n💰 **Budget Specialist Perspective:**")
        results.append(str(budget_result.final_output))

    if "goal" in specialists_needed:
        goal_agent = build_goal_agent()
        goal_result = await Runner.run(goal_agent, user_query, context=deps)
        results.append("\n🎯 **Goal Specialist Perspective:**")
        results.append(str(goal_result.final_output))

    if "investment" in specialists_needed:
        investment_agent = build_investment_agent()
        investment_result = await Runner.run(investment_agent, user_query, context=deps)
        results.append("\n📈 **Investment Specialist Perspective:**")
        results.append(str(investment_result.final_output))

    if "debt" in specialists_needed:
        debt_agent = build_debt_agent()
        debt_result = await Runner.run(debt_agent, user_query, context=deps)
        results.append("\n💳 **Debt Specialist Perspective:**")
        results.append(str(debt_result.final_output))

    # Synthesize recommendations
    results.append("\n🔄 **Coordinated Recommendations:**")
    results.append("Based on all specialist inputs:")
    results.append("• Review each specialist's advice carefully")
    results.append("• Look for synergies between tax, budget, and goal strategies")
    results.append("• Prioritize actions that benefit multiple areas")
    results.append("• Consider scheduling follow-ups with specific specialists")

    return "\n".join(results)


def analyze_query_intent(user_query: str) -> Dict[str, Any]:
    """Analyze user query to determine routing intent."""
    query_lower = user_query.lower()

    # Keywords for each specialist
    tax_keywords = [
        "tax",
        "deduction",
        "deductible",
        "irs",
        "filing",
        "write-off",
        "tax report",
        "tax category",
        "tax optimization",
        "refund",
    ]

    budget_keywords = [
        "budget",
        "spending",
        "expense",
        "subscription",
        "recurring",
        "overspend",
        "spending pattern",
        "categories",
        "monthly cost",
    ]

    goal_keywords = [
        "goal",
        "save",
        "saving",
        "target",
        "plan",
        "financial plan",
        "emergency fund",
        "retirement",
        "motivation",
        "progress",
    ]

    investment_keywords = [
        "invest",
        "investment",
        "portfolio",
        "stock",
        "bond",
        "etf",
        "index fund",
        "wealth",
        "compound",
        "return",
        "dividend",
        "asset allocation",
        "diversification",
        "risk",
    ]

    debt_keywords = [
        "debt",
        "loan",
        "credit",
        "payoff",
        "consolidation",
        "interest",
        "mortgage",
        "student loan",
        "credit card",
        "payment",
        "balance",
        "refinance",
        "snowball",
        "avalanche",
    ]

    # Monthly cost intent keywords
    monthly_cost_keywords = [
        "monthly cost",
        "monthly costs",
        "monthly spending",
        "spending last month",
        "last month spend",
        "expenses last month",
        "monthly expenses",
        "costs last month",
    ]

    # Score each category
    tax_score = sum(1 for keyword in tax_keywords if keyword in query_lower)
    budget_score = sum(1 for keyword in budget_keywords if keyword in query_lower)
    goal_score = sum(1 for keyword in goal_keywords if keyword in query_lower)
    investment_score = sum(
        1 for keyword in investment_keywords if keyword in query_lower
    )
    debt_score = sum(1 for keyword in debt_keywords if keyword in query_lower)
    monthly_cost_intent = any(k in query_lower for k in monthly_cost_keywords)

    # Determine primary intent
    scores = {
        "tax": tax_score,
        "budget": budget_score,
        "goal": goal_score,
        "investment": investment_score,
        "debt": debt_score,
    }
    primary_intent = (
        max(scores, key=lambda k: scores[k]) if max(scores.values()) > 0 else "general"
    )

    # Check for multi-specialist needs
    specialists_needed = []
    if tax_score > 0:
        specialists_needed.append("tax")
    if budget_score > 0:
        specialists_needed.append("budget")
    if goal_score > 0:
        specialists_needed.append("goal")
    if investment_score > 0:
        specialists_needed.append("investment")
    if debt_score > 0:
        specialists_needed.append("debt")

    is_multi_specialist = len(specialists_needed) > 1

    # Check for general/data operations
    data_keywords = [
        "transactions",
        "import",
        "csv",
        "pdf",
        "export",
        "recent",
        "search",
    ]
    is_data_operation = any(keyword in query_lower for keyword in data_keywords)

    return {
        "primary_intent": primary_intent,
        "specialists_needed": specialists_needed,
        "is_multi_specialist": is_multi_specialist,
        "is_data_operation": is_data_operation,
        "is_monthly_cost": monthly_cost_intent,
        "confidence_scores": scores,
    }


def build_orchestrator_agent() -> Agent[RunDeps]:
    """Build the main orchestrator agent with handoff capabilities."""
    logger = get_logger()

    # Configure ModelSettings for GPT-5 with reasoning and text verbosity
    # Use proper Agents SDK format for reasoning parameters
    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high"),  # minimal | low | medium | high
        verbosity="high",  # low | medium | high
    )

    handoff_agents = [
        build_tax_agent(),
        build_budget_agent(),
        build_goal_agent(),
        build_investment_agent(),
        build_debt_agent(),
    ]

    tools_list = [
        # Handoff tools
        handoff_to_tax_specialist,
        handoff_to_budget_specialist,
        handoff_to_goal_specialist,
        handoff_to_investment_specialist,
        handoff_to_debt_specialist,
        coordinate_multi_specialist_analysis,
        route_user_query,
        # Core data management tools
        ingest_csv,
        ingest_excel,
        list_excel_sheets,
        ingest_pdfs,
        list_recent_transactions,
        search_transactions,
        list_memories,
        summarize_file,
        summarize_overview,
        monthly_cost_summary,
        add_transaction,
        export_transactions,
        export_recurring_payments,
        export_clean_monthly_recurring,
        # Debug and logging tools
        # create_debug_tool(),  # TODO: Fix type issues
    ]

    logger.log_agent_init(
        "FinancialOrchestrator",
        {
            "model": "gpt-5",
            "model_settings": {"reasoning_effort": "high", "verbosity": "high"},
            "handoff_agents": [agent.name for agent in handoff_agents],
            "tool_count": len(tools_list),
            "tool_names": [
                getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                for tool in tools_list
            ],
            "specialist_capabilities": ["tax", "budget", "goal", "investment", "debt"],
        },
    )

    return Agent[RunDeps](
        name="FinancialOrchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model="gpt-5",
        model_settings=model_settings,
        handoffs=cast(List[Any], handoff_agents),  # type: ignore[arg-type]
        tools=cast(List[Any], tools_list),  # type: ignore[arg-type]
    )


@function_tool
def route_user_query(ctx: RunContextWrapper[RunDeps], user_query: str) -> str:
    """Analyze user query and provide routing recommendations.

    Args:
        user_query: The user's question or request
    """
    intent = analyze_query_intent(user_query)

    results = ["🧠 **Query Analysis & Routing**\n"]
    results.append(f'Query: "{user_query}"')
    results.append(f"Primary Intent: {intent['primary_intent'].title()}")

    if intent["is_multi_specialist"]:
        results.append(
            f"Multi-Specialist Query: {', '.join(intent['specialists_needed'])}"
        )
        results.append(
            "\n💡 **Recommendation:** Use coordinate_multi_specialist_analysis"
        )
    elif intent["primary_intent"] == "tax":
        results.append("\n💡 **Recommendation:** Route to Tax Specialist")
    elif intent["primary_intent"] == "budget":
        results.append("\n💡 **Recommendation:** Route to Budget Specialist")
    elif intent["primary_intent"] == "goal":
        results.append("\n💡 **Recommendation:** Route to Goal Specialist")
    elif intent["primary_intent"] == "investment":
        results.append("\n💡 **Recommendation:** Route to Investment Specialist")
    elif intent["primary_intent"] == "debt":
        results.append("\n💡 **Recommendation:** Route to Debt Specialist")
    elif intent["is_monthly_cost"]:
        results.append(
            "\n💡 **Recommendation:** Call monthly_cost_summary (fast, deterministic)"
        )
    elif intent["is_data_operation"]:
        results.append("\n💡 **Recommendation:** Handle directly (data operation)")
    else:
        results.append(
            "\n💡 **Recommendation:** General query - orchestrator can handle"
        )

    results.append(f"\nConfidence Scores: {intent['confidence_scores']}")

    return "\n".join(results)
