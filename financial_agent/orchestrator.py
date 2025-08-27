from __future__ import annotations
from typing import Dict, List, Optional
from agents import Agent, ModelSettings, function_tool, RunContextWrapper, Runner
from .context import RunDeps
from .specialists.tax_agent import build_tax_agent
from .specialists.budget_agent import build_budget_agent  
from .specialists.goal_agent import build_goal_agent
from .specialists.investment_agent import build_investment_agent
from .specialists.debt_agent import build_debt_agent
from .tools.ingest import ingest_csv
from .tools.excel_ingest import ingest_excel, list_excel_sheets
from .tools.query import list_recent_transactions, search_transactions
from .tools.pdf_ingest import ingest_pdfs
from .tools.memory import list_memories
from .tools.summarize import summarize_file, summarize_overview
from .tools.records import add_transaction
from .tools.export import export_transactions, export_recurring_payments


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
    additional_context: Optional[str] = None
) -> str:
    """Handoff tax-related queries to the Tax Specialist agent.
    
    Args:
        user_query: The user's tax-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    deps = ctx.context
    
    # Build tax specialist
    tax_agent = build_tax_agent()
    
    # Prepare query with context
    full_query = user_query
    if additional_context:
        full_query += f"\n\nAdditional context: {additional_context}"
    
    # Run the specialist agent
    result = await Runner.run(tax_agent, full_query, context=deps)
    
    # Return with specialist attribution
    return f"🏛️ **Tax Specialist Response:**\n\n{result.final_output}"


@function_tool
async def handoff_to_budget_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None
) -> str:
    """Handoff budget and spending analysis queries to the Budget Specialist agent.
    
    Args:
        user_query: The user's budget/spending-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    deps = ctx.context
    
    # Build budget specialist
    budget_agent = build_budget_agent()
    
    # Prepare query with context
    full_query = user_query
    if additional_context:
        full_query += f"\n\nAdditional context: {additional_context}"
    
    # Run the specialist agent
    result = await Runner.run(budget_agent, full_query, context=deps)
    
    # Return with specialist attribution
    return f"💰 **Budget Specialist Response:**\n\n{result.final_output}"


@function_tool
async def handoff_to_goal_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None
) -> str:
    """Handoff financial goal and planning queries to the Goal Specialist agent.
    
    Args:
        user_query: The user's goal/planning-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    deps = ctx.context
    
    # Build goal specialist
    goal_agent = build_goal_agent()
    
    # Prepare query with context
    full_query = user_query
    if additional_context:
        full_query += f"\n\nAdditional context: {additional_context}"
    
    # Run the specialist agent
    result = await Runner.run(goal_agent, full_query, context=deps)
    
    # Return with specialist attribution
    return f"🎯 **Goal Specialist Response:**\n\n{result.final_output}"


@function_tool
async def handoff_to_investment_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None
) -> str:
    """Handoff investment and portfolio queries to the Investment Specialist agent.
    
    Args:
        user_query: The user's investment-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    deps = ctx.context
    
    # Build investment specialist
    investment_agent = build_investment_agent()
    
    # Prepare query with context
    full_query = user_query
    if additional_context:
        full_query += f"\n\nAdditional context: {additional_context}"
    
    # Run the specialist agent
    result = await Runner.run(investment_agent, full_query, context=deps)
    
    # Return with specialist attribution
    return f"📈 **Investment Specialist Response:**\n\n{result.final_output}"


@function_tool
async def handoff_to_debt_specialist(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    additional_context: Optional[str] = None
) -> str:
    """Handoff debt and loan queries to the Debt Management Specialist agent.
    
    Args:
        user_query: The user's debt-related question or request
        additional_context: Any additional context to provide to the specialist
    """
    deps = ctx.context
    
    # Build debt specialist
    debt_agent = build_debt_agent()
    
    # Prepare query with context
    full_query = user_query
    if additional_context:
        full_query += f"\n\nAdditional context: {additional_context}"
    
    # Run the specialist agent
    result = await Runner.run(debt_agent, full_query, context=deps)
    
    # Return with specialist attribution
    return f"💳 **Debt Specialist Response:**\n\n{result.final_output}"


@function_tool
async def coordinate_multi_specialist_analysis(
    ctx: RunContextWrapper[RunDeps],
    user_query: str,
    specialists_needed: List[str]
) -> str:
    """Coordinate multiple specialists for comprehensive financial analysis.
    
    Args:
        user_query: The user's complex query requiring multiple specialists
        specialists_needed: List of specialists to involve (tax, budget, goal)
    """
    deps = ctx.context
    
    results = []
    results.append(f"🤝 **Multi-Specialist Analysis**\n")
    results.append(f"Query: {user_query}\n")
    results.append("=" * 50)
    
    # Run each requested specialist
    if "tax" in specialists_needed:
        tax_agent = build_tax_agent()
        tax_result = await Runner.run(tax_agent, user_query, context=deps)
        results.append(f"\n🏛️ **Tax Specialist Perspective:**")
        results.append(str(tax_result.final_output))
    
    if "budget" in specialists_needed:
        budget_agent = build_budget_agent()  
        budget_result = await Runner.run(budget_agent, user_query, context=deps)
        results.append(f"\n💰 **Budget Specialist Perspective:**")
        results.append(str(budget_result.final_output))
    
    if "goal" in specialists_needed:
        goal_agent = build_goal_agent()
        goal_result = await Runner.run(goal_agent, user_query, context=deps)
        results.append(f"\n🎯 **Goal Specialist Perspective:**")
        results.append(str(goal_result.final_output))
    
    if "investment" in specialists_needed:
        investment_agent = build_investment_agent()
        investment_result = await Runner.run(investment_agent, user_query, context=deps)
        results.append(f"\n📈 **Investment Specialist Perspective:**")
        results.append(str(investment_result.final_output))
    
    if "debt" in specialists_needed:
        debt_agent = build_debt_agent()
        debt_result = await Runner.run(debt_agent, user_query, context=deps)
        results.append(f"\n💳 **Debt Specialist Perspective:**")
        results.append(str(debt_result.final_output))
    
    # Synthesize recommendations
    results.append(f"\n🔄 **Coordinated Recommendations:**")
    results.append("Based on all specialist inputs:")
    results.append("• Review each specialist's advice carefully")
    results.append("• Look for synergies between tax, budget, and goal strategies")
    results.append("• Prioritize actions that benefit multiple areas")
    results.append("• Consider scheduling follow-ups with specific specialists")
    
    return "\n".join(results)


def analyze_query_intent(user_query: str) -> Dict[str, any]:
    """Analyze user query to determine routing intent."""
    query_lower = user_query.lower()
    
    # Keywords for each specialist
    tax_keywords = [
        'tax', 'deduction', 'deductible', 'irs', 'filing', 'write-off', 
        'tax report', 'tax category', 'tax optimization', 'refund'
    ]
    
    budget_keywords = [
        'budget', 'spending', 'expense', 'subscription', 'recurring', 
        'overspend', 'spending pattern', 'categories', 'monthly cost'
    ]
    
    goal_keywords = [
        'goal', 'save', 'saving', 'target', 'plan', 'financial plan',
        'emergency fund', 'retirement', 'motivation', 'progress'
    ]
    
    investment_keywords = [
        'invest', 'investment', 'portfolio', 'stock', 'bond', 'etf',
        'index fund', 'wealth', 'compound', 'return', 'dividend',
        'asset allocation', 'diversification', 'risk'
    ]
    
    debt_keywords = [
        'debt', 'loan', 'credit', 'payoff', 'consolidation', 'interest',
        'mortgage', 'student loan', 'credit card', 'payment', 'balance',
        'refinance', 'snowball', 'avalanche'
    ]
    
    # Score each category
    tax_score = sum(1 for keyword in tax_keywords if keyword in query_lower)
    budget_score = sum(1 for keyword in budget_keywords if keyword in query_lower)
    goal_score = sum(1 for keyword in goal_keywords if keyword in query_lower)
    investment_score = sum(1 for keyword in investment_keywords if keyword in query_lower)
    debt_score = sum(1 for keyword in debt_keywords if keyword in query_lower)
    
    # Determine primary intent
    scores = {
        'tax': tax_score, 
        'budget': budget_score, 
        'goal': goal_score,
        'investment': investment_score,
        'debt': debt_score
    }
    primary_intent = max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    # Check for multi-specialist needs
    specialists_needed = []
    if tax_score > 0:
        specialists_needed.append('tax')
    if budget_score > 0:
        specialists_needed.append('budget')
    if goal_score > 0:
        specialists_needed.append('goal')
    if investment_score > 0:
        specialists_needed.append('investment')
    if debt_score > 0:
        specialists_needed.append('debt')
    
    is_multi_specialist = len(specialists_needed) > 1
    
    # Check for general/data operations
    data_keywords = ['transactions', 'import', 'csv', 'pdf', 'export', 'recent', 'search']
    is_data_operation = any(keyword in query_lower for keyword in data_keywords)
    
    return {
        'primary_intent': primary_intent,
        'specialists_needed': specialists_needed,
        'is_multi_specialist': is_multi_specialist,
        'is_data_operation': is_data_operation,
        'confidence_scores': scores
    }


def build_orchestrator_agent() -> Agent[RunDeps]:
    """Build the main orchestrator agent with handoff capabilities."""
    
    return Agent[RunDeps](
        name="FinancialOrchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model="gps-5",
        model_settings=ModelSettings(),
        handoffs=[
            build_tax_agent(),
            build_budget_agent(),
            build_goal_agent(),
            build_investment_agent(),
            build_debt_agent()
        ],
        tools=[
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
            add_transaction,
            export_transactions,
            export_recurring_payments,
        ]
    )


@function_tool 
def route_user_query(
    ctx: RunContextWrapper[RunDeps],
    user_query: str
) -> str:
    """Analyze user query and provide routing recommendations.
    
    Args:
        user_query: The user's question or request
    """
    intent = analyze_query_intent(user_query)
    
    results = ["🧠 **Query Analysis & Routing**\n"]
    results.append(f"Query: \"{user_query}\"")
    results.append(f"Primary Intent: {intent['primary_intent'].title()}")
    
    if intent['is_multi_specialist']:
        results.append(f"Multi-Specialist Query: {', '.join(intent['specialists_needed'])}")
        results.append("\n💡 **Recommendation:** Use coordinate_multi_specialist_analysis")
    elif intent['primary_intent'] == 'tax':
        results.append("\n💡 **Recommendation:** Route to Tax Specialist")
    elif intent['primary_intent'] == 'budget':
        results.append("\n💡 **Recommendation:** Route to Budget Specialist")
    elif intent['primary_intent'] == 'goal':
        results.append("\n💡 **Recommendation:** Route to Goal Specialist")
    elif intent['primary_intent'] == 'investment':
        results.append("\n💡 **Recommendation:** Route to Investment Specialist")
    elif intent['primary_intent'] == 'debt':
        results.append("\n💡 **Recommendation:** Route to Debt Specialist")
    elif intent['is_data_operation']:
        results.append("\n💡 **Recommendation:** Handle directly (data operation)")
    else:
        results.append("\n💡 **Recommendation:** General query - orchestrator can handle")
    
    results.append(f"\nConfidence Scores: {intent['confidence_scores']}")
    
    return "\n".join(results)