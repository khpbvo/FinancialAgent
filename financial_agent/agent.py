from __future__ import annotations
from typing import Any, Generator
from contextlib import contextmanager
import os

from agents import Agent, ModelSettings, Runner, set_default_openai_key, RunContextWrapper
from agents.agent import StopAtTools

from .context import AppConfig, DB, RunDeps
from .tools.ingest import ingest_csv
from .tools.excel_ingest import ingest_excel, list_excel_sheets
from .tools.query import list_recent_transactions, search_transactions
from .tools.advice import analyze_and_advise
from .tools.pdf_ingest import ingest_pdfs
from .tools.memory import list_memories
from .tools.summarize import summarize_file, summarize_overview
from .tools.records import add_transaction
from .tools.budgets import set_budget, check_budget, list_budgets, suggest_budgets, delete_budget
from .tools.goals import create_goal, update_goal_progress, check_goals, suggest_savings_plan, complete_goal, pause_goal
from .tools.recurring import detect_recurring, list_subscriptions, analyze_subscription_value, predict_next_recurring
from .tools.export import export_transactions, generate_tax_report, export_budget_report, export_recurring_payments
from .orchestrator import build_orchestrator_agent


def build_app_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("FIN_AGENT_MODEL", "gps-5"),
    )


@contextmanager
def build_deps() -> Generator[RunDeps, None, None]:
    cfg = build_app_config()
    if cfg.openai_api_key:
        try:
            set_default_openai_key(cfg.openai_api_key)
        except Exception:
            pass
    with DB(cfg.db_path) as db:
        deps = RunDeps(config=cfg, db=db)
        deps.ensure_ready()
        yield deps


def dynamic_instructions(context: RunContextWrapper[RunDeps], agent: Agent[RunDeps]) -> str:
    """Generate dynamic instructions based on current context."""
    deps = context.context
    
    # Check if we have data in the database
    cur = deps.db.conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM transactions")
    tx_count = cur.fetchone()["count"]
    
    cur.execute("SELECT COUNT(*) as count FROM memories")
    mem_count = cur.fetchone()["count"]
    
    base_instructions = (
        "You are a comprehensive financial assistant with advanced capabilities:\n"
        "• Ingest CSV/PDF/Excel files and manage transactions\n"
        "• Set and track budgets with spending alerts\n"
        "• Create and monitor financial goals (savings, debt reduction)\n"
        "• Analyze investment readiness and portfolio strategies\n"
        "• Develop debt elimination and consolidation plans\n"
        "• Detect recurring transactions and subscriptions automatically\n"
        "• Export data to CSV/Excel/PDF for reports and tax preparation\n"
        "• Generate professional tax reports with categorized deductions\n"
        "• Provide expert analysis and personalized financial advice\n"
        "When giving advice, be concise and quantify when possible.\n"
        "\nNote: You coordinate 5 specialist agents for advanced analysis when needed."
    )
    
    if tx_count > 0:
        base_instructions += f"\n\nYou currently have {tx_count} transactions in the database."
        
        # Get spending summary
        cur.execute("SELECT SUM(amount) as total, MIN(date) as start_date, MAX(date) as end_date FROM transactions WHERE amount < 0")
        summary = cur.fetchone()
        if summary and summary["total"]:
            base_instructions += f" Total spending from {summary['start_date']} to {summary['end_date']}: €{abs(summary['total']):.2f}"
    else:
        base_instructions += "\n\nNo transactions loaded yet. Suggest using 'bootstrap' or ingesting CSV/PDF files from the documents folder."
    
    if mem_count > 0:
        base_instructions += f" You have {mem_count} memories/insights stored."
    
    return base_instructions


def build_agent(use_orchestrator: bool = True) -> Agent[RunDeps]:
    """Build the main financial agent.
    
    Args:
        use_orchestrator: If True, use the orchestrator with specialist handoffs.
                         If False, use the original monolithic agent.
    """
    if use_orchestrator:
        return build_orchestrator_agent()
    
    # Original monolithic agent (kept for backwards compatibility)
    return build_legacy_agent()


def build_legacy_agent() -> Agent[RunDeps]:

    # Create minimal ModelSettings for GPT-5 compatibility 
    # GPT-5 doesn't support max_tokens, temperature, or many other parameters
    model_settings = ModelSettings()

    return Agent[RunDeps](
        name="FinancialAgent",
        instructions=dynamic_instructions,  # Use dynamic instructions
        model=build_app_config().model,
        model_settings=model_settings,
        # Stop immediately after these tools to avoid unnecessary LLM calls
        tool_use_behavior=StopAtTools(stop_at_tool_names=["add_transaction", "ingest_csv", "ingest_pdfs"]),
        tools=[
            # Ingestion tools
            ingest_csv,
            ingest_excel,
            list_excel_sheets,
            ingest_pdfs,
            # Query tools
            list_recent_transactions,
            search_transactions,
            list_memories,
            # Budget tools
            set_budget,
            check_budget,
            list_budgets,
            suggest_budgets,
            delete_budget,
            # Goal tracking tools
            create_goal,
            update_goal_progress,
            check_goals,
            suggest_savings_plan,
            complete_goal,
            pause_goal,
            # Recurring transaction tools
            detect_recurring,
            list_subscriptions,
            analyze_subscription_value,
            predict_next_recurring,
            # Export tools
            export_transactions,
            generate_tax_report,
            export_budget_report,
            export_recurring_payments,
            # Analysis tools
            analyze_and_advise,
            summarize_file,
            summarize_overview,
            add_transaction,
        ],
    )


def run_once(user_input: str) -> str:
    with build_deps() as deps:
        agent = build_agent()
        result = Runner.run_sync(agent, user_input, context=deps)  # type: ignore[arg-type]
        return str(result.final_output)
