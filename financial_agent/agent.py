from __future__ import annotations
from typing import Any, Generator
from contextlib import contextmanager
import os

from agents import Agent, ModelSettings, Runner, set_default_openai_key, RunContextWrapper
from agents.agent import StopAtTools
from openai.types.shared import Reasoning

from .context import AppConfig, DB, RunDeps
from .logging_utils import get_logger, log_agent_execution, log_agent_result, log_info, log_error
from .openai_logger import get_openai_logger, set_openai_logger, OpenAILogger
from .enhanced_openai_logger import get_enhanced_openai_logger, set_enhanced_openai_logger, EnhancedOpenAILogger
from .logged_tools import (
    logged_ingest_csv as ingest_csv,
    logged_ingest_excel as ingest_excel,
    logged_list_excel_sheets as list_excel_sheets,
    logged_ingest_pdfs as ingest_pdfs,
    logged_list_recent_transactions as list_recent_transactions,
    logged_search_transactions as search_transactions,
    logged_list_memories as list_memories,
    logged_analyze_and_advise as analyze_and_advise,
    logged_summarize_file as summarize_file,
    logged_summarize_overview as summarize_overview,
    logged_add_transaction as add_transaction,
    logged_set_budget as set_budget,
    logged_check_budget as check_budget,
    logged_list_budgets as list_budgets,
    logged_suggest_budgets as suggest_budgets,
    logged_delete_budget as delete_budget,
    logged_create_goal as create_goal,
    logged_update_goal_progress as update_goal_progress,
    logged_check_goals as check_goals,
    logged_suggest_savings_plan as suggest_savings_plan,
    logged_complete_goal as complete_goal,
    logged_pause_goal as pause_goal,
    logged_detect_recurring as detect_recurring,
    logged_list_subscriptions as list_subscriptions,
    logged_analyze_subscription_value as analyze_subscription_value,
    logged_predict_next_recurring as predict_next_recurring,
    logged_export_clean_monthly_recurring as export_clean_monthly_recurring,
    logged_export_transactions as export_transactions,
    logged_generate_tax_report as generate_tax_report,
    logged_export_budget_report as export_budget_report,
    logged_export_recurring_payments as export_recurring_payments,
    logged_monthly_cost_summary as monthly_cost_summary,
)
from .orchestrator import build_orchestrator_agent


def build_app_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("FIN_AGENT_MODEL", "gpt-5"),
    )


@contextmanager
def build_deps() -> Generator[RunDeps, None, None]:
    cfg = build_app_config()
    logger = get_logger()
    
    log_info("Building dependencies", model=cfg.model, db_path=str(cfg.db_path))
    
    # Initialize OpenAI loggers with the same session ID
    openai_logger = OpenAILogger(session_id=logger.session_id)
    set_openai_logger(openai_logger)
    
    # Initialize enhanced OpenAI logger for deeper HTTP-level interception
    # TEMPORARILY DISABLED: Can cause hanging issues
    # enhanced_logger = EnhancedOpenAILogger(session_id=logger.session_id)
    # set_enhanced_openai_logger(enhanced_logger)
    
    if cfg.openai_api_key:
        try:
            set_default_openai_key(cfg.openai_api_key)
            log_info("OpenAI API key configured successfully")
            openai_logger.log_system_event("OpenAI API key configured", {"model": cfg.model})
            # enhanced_logger.log_system_event("Enhanced OpenAI logger initialized", {"model": cfg.model})
        except Exception as e:
            log_error("Failed to set OpenAI API key", error=e)
            pass
    
    with DB(cfg.db_path) as db:
        deps = RunDeps(config=cfg, db=db)
        deps.ensure_ready()
        
        # Log database state
        cur = db.conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM transactions")
        tx_count = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM memories")
        mem_count = cur.fetchone()["count"]
        
        log_info("Dependencies ready", 
                 transactions=tx_count, 
                 memories=mem_count,
                 db_tables=["transactions", "memories", "budgets", "goals"])
        
        openai_logger.log_system_event("Financial Agent context ready", {
            "transactions": tx_count,
            "memories": mem_count,
            "model": cfg.model
        })
        # enhanced_logger.log_system_event("Financial Agent context ready", {
        #     "transactions": tx_count,
        #     "memories": mem_count,
        #     "model": cfg.model
        # })
        
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
    logger = get_logger()
    
    if use_orchestrator:
        logger.log_agent_init("OrchestratorAgent", {
            "type": "orchestrator",
            "specialists": ["tax", "budget", "goal", "investment", "debt"],
            "use_orchestrator": True
        })
        return build_orchestrator_agent()
    
    # Original monolithic agent (kept for backwards compatibility)
    logger.log_agent_init("FinancialAgent", {
        "type": "monolithic",
        "use_orchestrator": False
    })
    return build_legacy_agent()


def build_legacy_agent() -> Agent[RunDeps]:
    logger = get_logger()
    
    # Configure ModelSettings for GPT-5 with reasoning and text verbosity
    # Use proper Agents SDK format for reasoning parameters
    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high"),     # minimal | low | medium | high
        verbosity="high"                        # low | medium | high
    )
    
    tools_list = [
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
        export_clean_monthly_recurring,
        # Analysis tools
        analyze_and_advise,
        summarize_file,
        summarize_overview,
        add_transaction,
        # Fast cost summary
        monthly_cost_summary,
    ]
    
    logger.log_agent_init("FinancialAgent", {
        "model": build_app_config().model,
        "model_settings": {
            "reasoning_effort": "high",
            "verbosity": "high"
        },
        "tool_count": len(tools_list),
        "tool_names": [getattr(tool, 'name', getattr(tool, '__name__', str(tool))) for tool in tools_list],
        "stop_at_tools": ["add_transaction", "ingest_csv", "ingest_pdfs"]
    })

    return Agent[RunDeps](
        name="FinancialAgent",
        instructions=dynamic_instructions,  # Use dynamic instructions
        model=build_app_config().model,
        model_settings=model_settings,
        # Stop immediately after these tools to avoid unnecessary LLM calls
        tool_use_behavior=StopAtTools(stop_at_tool_names=["add_transaction", "ingest_csv", "ingest_pdfs"]),
        tools=tools_list,
    )


def run_once(user_input: str) -> str:
    with build_deps() as deps:
        agent = build_agent()
        
        # Log execution with comprehensive context
        context_info = {
            "db_transactions": deps.db.conn.cursor().execute("SELECT COUNT(*) FROM transactions").fetchone()[0],
            "db_memories": deps.db.conn.cursor().execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "model": deps.config.model
        }
        
        with log_agent_execution(agent.name, user_input, context_info) as agent_id:
            try:
                result = Runner.run_sync(agent, user_input, context=deps)  # type: ignore[arg-type]
                log_agent_result(agent_id, result.final_output)
                return str(result.final_output)
            except Exception as e:
                get_logger().log_agent_error(agent_id, e)
                raise
