"""Compatibility wrapper for agent builders.

This module provides a stable import path `financial_agent.tools.agent`
that re-exports the primary build/run helpers from `financial_agent.agent`.
Some codebases or docs may reference this path, so we keep a thin shim
here to avoid import errors.
"""

from __future__ import annotations

from ..agent import build_agent, build_legacy_agent, build_deps, run_once

__all__ = [
    "build_agent",
    "build_legacy_agent",
    "build_deps",
    "run_once",
]
