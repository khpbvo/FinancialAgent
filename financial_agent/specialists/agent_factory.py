from __future__ import annotations

from typing import Any, Sequence

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning

from ..context import RunDeps


def build_specialist_agent(
    name: str, instructions: str, tools: Sequence[Any]
) -> Agent[RunDeps]:
    """Factory to build specialist agents with consistent model settings.

    This centralizes the repeated configuration used across specialist agents
    (model name, reasoning effort, verbosity), reducing duplication and keeping
    settings consistent.
    """

    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high"),  # minimal | low | medium | high
        verbosity="high",  # low | medium | high
    )

    return Agent[RunDeps](
        name=name,
        instructions=instructions,
        model="gpt-5",
        model_settings=model_settings,
        tools=list(tools),
    )
