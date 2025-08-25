from __future__ import annotations
from pathlib import Path

from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool

from ..context import RunDeps
from ..db.sql import INSERT_MEMORY


SUMMARIZER_PROMPT = (
    "You are a financial document summarizer. Extract key amounts, entities, dates, and classify the document. "
    "Provide a concise summary with bullet points and a one-line takeaway."
)


@function_tool
async def summarize_file(ctx: RunContextWrapper[RunDeps], relative_path: str) -> str:
    """Summarize a single file under documents/ and store in memory.

    Args:
        relative_path: Path relative to the documents directory
    """
    deps = ctx.context
    path = deps.config.documents_dir / relative_path
    if not path.exists():
        return f"File not found: {relative_path}"

    # read limited text
    content = ""
    try:
        if path.suffix.lower() == ".pdf":
            from PyPDF2 import PdfReader
            txt = []
            reader = PdfReader(str(path))
            for p in reader.pages[:5]:
                try:
                    txt.append(p.extract_text() or "")
                except Exception:
                    pass
            content = "\n".join(txt)
        else:
            content = path.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception as e:
        return f"Failed to read file: {e}"

    agent = Agent[RunDeps](
        name="Summarizer",
        instructions=SUMMARIZER_PROMPT,
        model=deps.config.model,
        model_settings=ModelSettings(),
    )

    result = await Runner.run(agent, f"Summarize this:\n\n{content}", context=deps)
    output = str(result.final_output)

    cur = deps.db.conn.cursor()
    cur.execute(INSERT_MEMORY, ("summary", output, f"file,{path.name}"))
    deps.db.conn.commit()
    return output


@function_tool
async def summarize_overview(ctx: RunContextWrapper[RunDeps], last_n_memories: int = 20) -> str:
    """Produce an overview summary across the latest stored summaries and transactions."""
    deps = ctx.context
    # pull some content from memories and transactions
    cur = deps.db.conn.cursor()
    cur.execute("SELECT content FROM memories ORDER BY created_at DESC, id DESC LIMIT ?", (last_n_memories,))
    mems = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT date, description, amount, currency FROM transactions ORDER BY date DESC, id DESC LIMIT 30")
    tx_lines = [f"{r['date']}: {r['description']} ({r['amount']} {r['currency']})" for r in cur.fetchall()]

    corpus = "\n\n".join([
        "Recent memories:",
        *mems,
        "\nRecent transactions:",
        *tx_lines,
    ])

    agent = Agent[RunDeps](
        name="Portfolio Overview",
        instructions=(
            "You are a financial analyst. Create a short overview of the user's recent financial activity, "
            "detect spending patterns and risks, and suggest top 3 actions."
        ),
        model=deps.config.model,
        model_settings=ModelSettings(),
    )

    result = await Runner.run(agent, f"Create an overview based on:\n{corpus[:6000]}", context=deps)
    output = str(result.final_output)
    cur = deps.db.conn.cursor()
    cur.execute(INSERT_MEMORY, ("insight", output, "overview"))
    deps.db.conn.commit()
    return output
