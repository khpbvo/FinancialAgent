"""Advanced agent with SDK optimizations."""

from __future__ import annotations
from typing import Any, Optional
import os
import re
from datetime import datetime

from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    RunResult,
    WebSearchTool,
)
from agents.agent import StopAtTools
from agents.guardrails import Guardrail, GuardrailError  # pylint: disable=E0611
from agents.tool import ToolChoice  # pylint: disable=E0611
from agents.hooks import AsyncAgentHooks  # pylint: disable=E0611

from .context import AppConfig, RunDeps
from .tools.ingest import ingest_csv
from .tools.query import list_recent_transactions, search_transactions
from .tools.advice import analyze_and_advise
from .tools.pdf_ingest import ingest_pdfs
from .tools.memory import list_memories
from .tools.summarize import summarize_file, summarize_overview
from .tools.records import add_transaction
from .models import FinancialSummary, BudgetAdvice, SpendingAnalysis


def tool_choice_for_critical_operations(
    context: RunContextWrapper[RunDeps],
) -> Optional[ToolChoice]:
    """Determine tool choice based on context for critical operations."""
    deps = context.context

    # Check if database is empty - force ingestion
    cur = deps.db.conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM transactions")
    tx_count = cur.fetchone()["count"]

    if tx_count == 0:
        # Force user to ingest data first
        return ToolChoice(type="function", function={"name": "ingest_csv"})

    return None  # Let the model choose


def is_web_search_enabled(ctx: RunContextWrapper[RunDeps], agent: Agent) -> bool:
    """Enable web search only when user explicitly requests real-time data."""
    # This could be enhanced to check user preferences or subscription level
    return os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"


def is_ingestion_enabled(ctx: RunContextWrapper[RunDeps], agent: Agent) -> bool:
    """Enable ingestion tools based on context."""
    deps = ctx.context
    # Check if documents directory has files
    docs_dir = deps.config.documents_dir
    has_files = any(docs_dir.glob("*.csv")) or any(docs_dir.glob("*.pdf"))
    return has_files


class PIIGuardrail(Guardrail):
    """Guardrail to detect and prevent PII exposure."""

    name = "pii_protection"
    description = "Prevents exposure of sensitive personal information"

    async def guard(self, context: RunContextWrapper[RunDeps], input: str) -> str:
        # Patterns for common PII
        patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b",
            "phone": r"\b\+?[\d\s-()]{10,}\b",
        }

        for pii_type, pattern in patterns.items():
            if re.search(pattern, input, re.IGNORECASE):
                raise GuardrailError(
                    f"Detected potential {pii_type} in input. Please remove sensitive information."
                )

        return input


class TransactionValidationGuardrail(Guardrail):
    """Guardrail to validate transaction data."""

    name = "transaction_validation"
    description = "Validates transaction amounts and dates"

    async def guard(self, context: RunContextWrapper[RunDeps], input: str) -> str:
        # Check for unrealistic transaction amounts
        amount_pattern = r"€?\s*([\d,]+\.?\d*)"
        amounts = re.findall(amount_pattern, input)

        for amount_str in amounts:
            try:
                amount = float(amount_str.replace(",", ""))
                if amount > 1000000:  # Flag transactions over 1M
                    raise GuardrailError(
                        f"Transaction amount €{amount:,.2f} seems unusually high. Please verify."
                    )
            except ValueError:
                pass

        return input


class FinancialAgentHooks(AsyncAgentHooks):
    """Lifecycle hooks for monitoring and metrics."""

    async def on_run_start(self, context: RunContextWrapper[RunDeps]) -> None:
        """Log run start with metadata."""
        print(f"🚀 Starting agent run at {datetime.now().isoformat()}")

    async def on_run_end(
        self, context: RunContextWrapper[RunDeps], result: RunResult
    ) -> None:
        """Log run completion with metrics."""
        print(f"✅ Agent run completed in {result.elapsed_time:.2f}s")
        print(f"   Model calls: {len(result.model_calls)}")
        print(
            f"   Tool calls: {len([i for i in result.new_items if hasattr(i, 'tool_name')])}"
        )

    async def on_tool_start(
        self, context: RunContextWrapper[RunDeps], tool_name: str, args: dict
    ) -> None:
        """Log tool execution start."""
        print(f"🔧 Executing tool: {tool_name}")

    async def on_tool_end(
        self, context: RunContextWrapper[RunDeps], tool_name: str, result: Any
    ) -> None:
        """Log tool execution completion."""
        print(f"   ✓ {tool_name} completed")

    async def on_error(
        self, context: RunContextWrapper[RunDeps], error: Exception
    ) -> None:
        """Log errors for debugging."""
        print(f"❌ Error occurred: {error}")


def extract_financial_insights(result: RunResult) -> str:
    """Custom output extractor for financial summaries."""
    # Look for structured financial data in the result
    for item in reversed(result.new_items):
        # Check if we have a structured output
        if hasattr(item, "output") and isinstance(
            item.output, (FinancialSummary, BudgetAdvice, SpendingAnalysis)
        ):
            # Convert structured output to readable format
            if isinstance(item.output, FinancialSummary):
                return f"""
📊 Financial Summary for {item.output.period}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Income: €{item.output.total_income:,.2f}
💸 Expenses: €{item.output.total_expenses:,.2f}
💵 Net: €{item.output.net_savings:,.2f}

Key Insights:
{chr(10).join(f'• {insight}' for insight in item.output.key_insights)}

Next Steps:
{chr(10).join(f'✓ {step}' for step in item.output.next_steps)}
"""
            elif isinstance(item.output, BudgetAdvice):
                return f"""
💡 Budget Analysis
━━━━━━━━━━━━━━━
{item.output.summary}

Recommendations:
{chr(10).join(f'• {rec}' for rec in item.output.recommendations)}

Savings Potential: €{item.output.savings_potential:,.2f}/month
"""

    # Fallback to default output
    return str(result.final_output)


def build_advanced_agent(
    enable_web_search: bool = True,
    enable_guardrails: bool = True,
    enable_hooks: bool = True,
    tool_choice_mode: Optional[str] = None,
) -> Agent[RunDeps]:
    """Build an advanced agent with all SDK optimizations."""

    cfg = AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("FIN_AGENT_MODEL", "gpt-4o"),
    )

    # Model settings with tool choice control
    model_settings = ModelSettings()
    if tool_choice_mode == "critical":
        model_settings.tool_choice = tool_choice_for_critical_operations
    elif tool_choice_mode == "required":
        model_settings.tool_choice = ToolChoice(type="required")

    # Build tools list with conditional enabling
    tools = [
        # Ingestion tools (always available; enable at runtime via prompts)
        ingest_csv,
        ingest_pdfs,
        # Query tools (always enabled)
        list_recent_transactions,
        search_transactions,
        list_memories,
        # Analysis tools
        analyze_and_advise,
        summarize_file,
        summarize_overview,
        add_transaction,
    ]

    # Add web search if enabled
    if enable_web_search:
        tools.append(WebSearchTool())

    # Filter out None values
    tools = [t for t in tools if t is not None]

    # Guardrails are defined above; wiring can be added when supported by Agent

    # Build hooks
    hooks = FinancialAgentHooks() if enable_hooks else None

    instructions = """You are an advanced financial assistant with real-time capabilities.
    You can analyze transactions, provide budget advice, and access current market data.
    Always prioritize user privacy and validate data accuracy.
    
    Key capabilities:
    - Transaction analysis and categorization
    - Budget planning and optimization
    - Real-time financial data access (when enabled)
    - Secure handling of financial information
    """

    return Agent[RunDeps](
        name="AdvancedFinancialAgent",
        instructions=instructions,
        model=cfg.model,
        model_settings=model_settings,
        tool_use_behavior=StopAtTools(
            stop_at_tool_names=["add_transaction", "ingest_csv"]
        ),
        tools=tools,
        hooks=hooks,
    )
