from __future__ import annotations
from dataclasses import dataclass

from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    Runner,
    function_tool,
)

from ..context import RunDeps
from ..db.sql import INSERT_MEMORY


FINANCIAL_INSTRUCTIONS = (
    "You are a financial expert. You analyze transactions and documents to provide insights, summaries, and practical advice. "
    "Be explicit about assumptions, quantify when possible, and show short bullet points. If the user asks for a plan, propose steps with estimated impact."
)


@dataclass
class AnalysisPrompt:
    question: str
    extra_context: str | None = None

    def to_text(self) -> str:
        base = self.question.strip()
        if self.extra_context:
            base += f"\n\nAdditional context:\n{self.extra_context.strip()}"
        return base


@function_tool
async def analyze_and_advise(
    ctx: RunContextWrapper[RunDeps], question: str, extra_context: str | None = None
) -> str:
    """Ask the financial expert to analyze your data and provide advice.

    Args:
        question: What you'd like to know (e.g., "How can I reduce monthly expenses?")
        extra_context: Optional additional notes (e.g., preferences)
    """
    deps = ctx.context

    expert = Agent[RunDeps](
        name="Financial Expert",
        instructions=FINANCIAL_INSTRUCTIONS,
        model=deps.config.model,
        model_settings=ModelSettings(),
    )

    prompt = AnalysisPrompt(question=question, extra_context=extra_context)
    result = await Runner.run(
        expert,
        prompt.to_text(),
        context=deps,
    )
    output = result.final_output

    # store in memories
    cur = deps.db.conn.cursor()
    cur.execute(INSERT_MEMORY, ("advice", str(output), "analysis"))
    deps.db.conn.commit()
    return str(output)
