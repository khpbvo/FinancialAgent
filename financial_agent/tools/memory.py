from __future__ import annotations
from agents import RunContextWrapper, function_tool

from ..context import RunDeps
from ..db.sql import LIST_MEMORIES


@function_tool
def list_memories(ctx: RunContextWrapper[RunDeps], limit: int = 10) -> str:
    """List recent saved memories (summaries, insights, advice)."""
    deps = ctx.context
    cur = deps.db.conn.cursor()
    cur.execute(LIST_MEMORIES, (limit,))
    rows = cur.fetchall()
    if not rows:
        return "No memories saved yet."
    lines = [f"[{r['created_at']}] {r['kind']}: {r['content'][:140]}... (tags: {r['tags']})" for r in rows]
    return "\n".join(lines)
