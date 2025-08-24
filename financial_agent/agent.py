from __future__ import annotations
from typing import Any
import os

from agents import Agent, ModelSettings, Runner, set_default_openai_key

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


def build_agent() -> Agent[RunDeps]:
    instructions = (
        "You are a financial assistant. You can ingest CSV files, list/search transactions, "
        "and provide expert analysis and advice grounded in the user's stored data."
        "When giving advice, be concise and quantify when possible."
    )

    return Agent[RunDeps](
        name="FinancialAgent",
        instructions=instructions,
        model=build_app_config().model,
        model_settings=ModelSettings(),
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
