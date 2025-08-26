from __future__ import annotations
from typing import Any
import os

from agents import Agent, ModelSettings, Runner, set_default_openai_key, RunContextWrapper
from agents.agent import StopAtTools

from .context import AppConfig, DB, RunDeps
from .tools.ingest import ingest_csv
from .tools.query import list_recent_transactions, search_transactions
from .tools.advice import analyze_and_advise
from .tools.pdf_ingest import ingest_pdfs
from .tools.memory import list_memories
from .tools.summarize import summarize_file, summarize_overview
from .tools.records import add_transaction


def build_app_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("FIN_AGENT_MODEL", "gpt-5"),
    )


def build_deps() -> RunDeps:
    cfg = build_app_config()
    if cfg.openai_api_key:
        try:
            set_default_openai_key(cfg.openai_api_key)
        except Exception:
            pass
    db = DB(cfg.db_path)
    deps = RunDeps(config=cfg, db=db)
    deps.ensure_ready()
    return deps


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
        "You are a financial assistant. You can ingest CSV files, list/search transactions, "
        "and provide expert analysis and advice grounded in the user's stored data. "
        "When giving advice, be concise and quantify when possible."
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


def build_agent() -> Agent[RunDeps]:

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
            ingest_csv,
            ingest_pdfs,
            list_recent_transactions,
            search_transactions,
            list_memories,
            analyze_and_advise,
            summarize_file,
            summarize_overview,
            add_transaction,
        ],
    )


def run_once(user_input: str) -> str:
    deps = build_deps()
    agent = build_agent()
    result = Runner.run_sync(agent, user_input, context=deps)  # type: ignore[arg-type]
    return str(result.final_output)
