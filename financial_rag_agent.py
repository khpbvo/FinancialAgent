#!/usr/bin/env python3
"""RAG-enabled financial chat agent using OpenAI Agents SDK.

This module adds an agent that can ingest financial documents, store them as
3072-dimension embeddings, and answer questions through a streaming chat
interface. Standard tools like read_file, web browsing, file search and shell
commands (e.g. ``cat`` or ``sed``) are available through the OpenAI Agents SDK.
"""

from __future__ import annotations

import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional

from agents import Agent, Runner, function_tool, SQLiteSession
from agents import RunContextWrapper
from agents.tool import WebSearchTool, FileSearchTool
from openai import AsyncOpenAI
from pydantic import BaseModel
from openai.types.responses import ResponseTextDeltaEvent
from pypdf import PdfReader  # type: ignore
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # Fallback when pandas isn't available

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings:
    """Application configuration"""

    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    STORE_PATH: Path = Path(__file__).parent / "documents" / "embeddings.json"
    # Optional: OpenAI Vector Store IDs for hosted FileSearchTool
    # Configure via env var OPENAI_VECTOR_STORE_IDS as a comma-separated list
    _VEC_IDS_RAW: str = os.getenv("OPENAI_VECTOR_STORE_IDS", "")
    FILESEARCH_VECTOR_STORE_IDS: list[str] = [
        s.strip() for s in _VEC_IDS_RAW.split(",") if s.strip()
    ]


settings = Settings()

# OpenAI async client (dummy embeddings allowed when API key missing)
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Embedding Store
# ---------------------------------------------------------------------------

class EmbeddingRecord(BaseModel):
    path: str
    content: str
    embedding: List[float]


class EmbeddingsStore:
    """Simple JSON based embeddings store"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.records: List[EmbeddingRecord] = [EmbeddingRecord(**r) for r in data]
        else:
            self.records = []

    def save(self) -> None:
        self.path.write_text(json.dumps([r.dict() for r in self.records], indent=2))

    async def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI, fallback to zeros for tests"""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "test-key-placeholder":
            # Return zero vector when key is missing (for tests)
            return np.zeros(settings.EMBEDDING_DIMENSION)

        resp = await openai_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return np.array(resp.data[0].embedding)

    async def add_document(self, path: str, content: str) -> None:
        embedding = await self.create_embedding(content)
        record = EmbeddingRecord(path=path, content=content, embedding=embedding.tolist())
        self.records.append(record)
        self.save()

    async def search(self, query: str) -> str:
        if not self.records:
            return "No documents indexed."

        query_embedding = await self.create_embedding(query)
        q_norm = np.linalg.norm(query_embedding) or 1.0

        best_score = -1.0
        best_record: EmbeddingRecord | None = None
        for record in self.records:
            emb = np.array(record.embedding)
            denom = (np.linalg.norm(emb) or 1.0) * q_norm
            score = float(np.dot(emb, query_embedding) / denom)
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None:
            return "No relevant documents found."

        snippet = best_record.content[:1000]
        return f"From {best_record.path}:\n{snippet}"


store = EmbeddingsStore(settings.STORE_PATH)


# ---------------------------------------------------------------------------
# Function Tools
# ---------------------------------------------------------------------------

async def ingest_document_fn(file_path: str) -> str:
    """Read a document from disk and store its embedding"""
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    content = path.read_text(encoding="utf-8", errors="ignore")[:20000]
    await store.add_document(file_path, content)
    return f"Ingested {file_path}"


async def retrieve_financial_context_fn(query: str) -> str:
    """Retrieve relevant document context for the query"""
    return await store.search(query)


# Wrap raw functions as FunctionTool instances for the agent
ingest_document = function_tool(ingest_document_fn)
retrieve_financial_context = function_tool(retrieve_financial_context_fn)


# ---- Filesystem context and helpers ----

class FileContext(BaseModel):
    base_dir: str
    cwd: str


def _safe_resolve(ctx: RunContextWrapper[FileContext], path: str) -> Path:
    """Resolve path safely within base_dir, supporting relative to cwd."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(ctx.context.cwd) / p
    p = p.resolve()
    base = Path(ctx.context.base_dir).resolve()
    try:
        # Python 3.9+: emulate is_relative_to
        p.relative_to(base)
    except Exception:
        raise ValueError(f"Access denied outside base directory: {p}")
    return p


@function_tool
def change_directory(ctx: RunContextWrapper[FileContext], path: str) -> str:
    """Change the current working directory for subsequent file operations.

    Args:
        path: The directory to switch to. Relative paths resolve from the current cwd.
    """
    target = _safe_resolve(ctx, path)
    if not target.exists() or not target.is_dir():
        return f"Directory not found: {target}"
    ctx.context.cwd = str(target)
    return f"Changed directory to: {ctx.context.cwd}"


@function_tool
def read_file(ctx: RunContextWrapper[FileContext], path: str, max_bytes: int = 50000) -> str:
    """Read a file's contents (truncated).

    Args:
        path: Path to the file to read (relative to cwd or absolute within base_dir).
        max_bytes: Maximum bytes to return to avoid huge outputs.
    """
    fp = _safe_resolve(ctx, path)
    if not fp.exists() or not fp.is_file():
        return f"File not found: {fp}"
    try:
        data = fp.read_bytes()[: max(0, max_bytes)]
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = data.decode("latin-1", errors="ignore")
        suffix = " (truncated)" if fp.stat().st_size > max_bytes else ""
        return f"=== {fp} ===\n{text}{suffix}"
    except Exception as e:  # pragma: no cover
        return f"Error reading {fp}: {e}"


@function_tool
def list_directory(
    ctx: RunContextWrapper[FileContext],
    path: Optional[str] = None,
    recursive: bool = False,
    max_entries: int = 200,
) -> str:
    """List directory contents.

    Args:
        path: Directory to list. Defaults to current cwd.
        recursive: Recurse into subdirectories.
        max_entries: Max entries to return.
    """
    root = _safe_resolve(ctx, path or ctx.context.cwd)
    if not root.exists() or not root.is_dir():
        return f"Directory not found: {root}"
    lines: List[str] = []
    count = 0
    if recursive:
        for dpath, dnames, fnames in os.walk(root):
            rel = Path(dpath).resolve().relative_to(Path(ctx.context.base_dir))
            lines.append(f"[{rel}]")
            for name in sorted(dnames + fnames):
                if count >= max_entries:
                    break
                lines.append(f"- {name}")
                count += 1
            if count >= max_entries:
                break
    else:
        for entry in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if count >= max_entries:
                break
            marker = "/" if entry.is_dir() else ""
            lines.append(entry.name + marker)
            count += 1
    if count >= max_entries:
        lines.append("... (truncated)")
    return "\n".join(lines)


@function_tool
def search_text(
    ctx: RunContextWrapper[FileContext],
    query: str,
    path: Optional[str] = None,
    file_glob: str = "**/*",
    case_sensitive: bool = False,
    max_results: int = 100,
    context_lines: int = 1,
) -> str:
    """Search for text in files under a directory.

    Args:
        query: The text or regex to search for.
        path: Directory root to search. Defaults to cwd.
        file_glob: Glob pattern for files, e.g. "**/*.py".
        case_sensitive: Whether search is case sensitive.
        max_results: Maximum number of matches to return.
        context_lines: Number of context lines around a match.
    """
    import re

    root = _safe_resolve(ctx, path or ctx.context.cwd)
    if not root.is_dir():
        return f"Not a directory: {root}"
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query, flags)

    results: List[str] = []
    for file_path in root.glob(file_glob):
        if len(results) >= max_results:
            break
        if not file_path.is_file():
            continue
        # Skip large files
        try:
            if file_path.stat().st_size > 512 * 1024:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines()):
            if pattern.search(line):
                start = max(0, i - context_lines)
                end = min(len(text.splitlines()), i + context_lines + 1)
                snippet = "\n".join(text.splitlines()[start:end])
                rel = file_path.resolve().relative_to(Path(ctx.context.base_dir))
                results.append(f"{rel}:{i+1}:\n{snippet}")
                if len(results) >= max_results:
                    break
    if not results:
        return "No matches found."
    return "\n\n".join(results)


@function_tool
def open_all_files_in_directory(
    ctx: RunContextWrapper[FileContext],
    path: str,
    file_glob: str = "*",
    max_files: int = 50,
    max_bytes_per_file: int = 20000,
) -> str:
    """Open and return contents of all files in a directory (truncated per file).

    Args:
        path: Directory to open files from.
        file_glob: Glob pattern to select files.
        max_files: Maximum files to open.
        max_bytes_per_file: Max bytes per file to include.
    """
    root = _safe_resolve(ctx, path)
    if not root.exists() or not root.is_dir():
        return f"Directory not found: {root}"
    out: Dict[str, str] = {}
    count = 0
    for fp in sorted(root.glob(file_glob)):
        if count >= max_files:
            break
        if not fp.is_file():
            continue
        try:
            data = fp.read_bytes()[: max(0, max_bytes_per_file)]
            text = data.decode("utf-8", errors="ignore")
            rel = str(fp.resolve().relative_to(Path(ctx.context.base_dir)))
            out[rel] = text
            count += 1
        except Exception:
            continue
    return json.dumps(out, indent=2)


@function_tool
def analyze_file(ctx: RunContextWrapper[FileContext], path: str, max_bytes: int = 20000) -> str:
    """Analyze a file and return a concise summary.

    Supports CSV (basic stats), PDF (text extract), and text files.
    """
    fp = _safe_resolve(ctx, path)
    if not fp.exists() or not fp.is_file():
        return f"File not found: {fp}"
    suffix = fp.suffix.lower()
    try:
        if suffix == ".csv":
            if pd is None:
                return "pandas not available to analyze CSV."
            df = pd.read_csv(fp)
            info = {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": list(map(str, df.columns.tolist())),
                "head": df.head(5).to_dict(orient="records"),
                "summary": df.describe(include="all").fillna(0).to_dict(),
            }
            return json.dumps(info, indent=2, default=str)
        elif suffix == ".pdf":
            reader = PdfReader(str(fp))
            pages = []
            for i, page in enumerate(reader.pages[:5]):  # limit pages
                pages.append(page.extract_text() or "")
            text = "\n\n".join(pages)[:max_bytes]
            return f"PDF Extract (truncated):\n{text}"
        else:
            data = fp.read_bytes()[: max(0, max_bytes)]
            text = data.decode("utf-8", errors="ignore")
            return f"Text extract (truncated):\n{text}"
    except Exception as e:
        return f"Error analyzing {fp}: {e}"


# ---------------------------------------------------------------------------
# Agent Creation and Streaming Chat
# ---------------------------------------------------------------------------

def create_chat_agent() -> Agent:
    """Create a chat agent with standard tools and retrieval"""
    tools = [
        ingest_document,
        retrieve_financial_context,
        # Filesystem tools
        change_directory,
        list_directory,
        read_file,
        search_text,
        open_all_files_in_directory,
        analyze_file,
        # Hosted tools
        WebSearchTool(),
    ]

    # FileSearchTool requires vector_store_ids per Docs/tools.md; include only if configured
    if settings.FILESEARCH_VECTOR_STORE_IDS:
        tools.append(
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=settings.FILESEARCH_VECTOR_STORE_IDS,
            )
        )

    # Local shell tool (optional utility)
    # Note: LocalShellTool requires an executor; omitted here per Docs to keep setup simple

    instructions = (
        "You are a financial assistant. Use the provided tools to read files, "
        "search the web, and fetch context from documents. Always call "
        "retrieve_financial_context before answering user financial questions."
    )

    return Agent(
        name="financial_chat_agent",
        model=settings.OPENAI_MODEL,
        instructions=instructions,
        tools=tools,
    )


async def stream_chat(query: str) -> None:
    """Run the chat agent with streaming output"""
    agent = create_chat_agent()
    # Provide filesystem context (base is project root; start cwd at documents/)
    base_dir = str(Path(__file__).parent.resolve())
    start_cwd = str((Path(__file__).parent / "documents").resolve())
    fs_context = FileContext(base_dir=base_dir, cwd=start_cwd)
    result = Runner.run_streamed(agent, input=query, context=fs_context)
    async for event in result.stream_events():
        # Stream raw text deltas as they arrive
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()


async def interactive_chat(session_id: str = "financial_chat") -> None:
    """Interactive REPL chat with streaming output and session memory."""
    agent = create_chat_agent()
    # Persist conversation history in a lightweight SQLite file under documents/
    db_path = str((Path(__file__).parent / "documents" / "conversation_history.db").resolve())
    session = SQLiteSession(session_id, db_path)
    base_dir = str(Path(__file__).parent.resolve())
    start_cwd = str((Path(__file__).parent / "documents").resolve())
    fs_context = FileContext(base_dir=base_dir, cwd=start_cwd)

    print("\nInteractive financial chat. Type 'exit' or 'quit' to leave. Type 'clear' to reset memory.\n")
    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(None, lambda: input("You: ").strip())
        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "clear":
            await session.clear_session()
            print("(Memory cleared)\n")
            continue

        # Stream the response for this turn, preserving conversation via session
        try:
            result = Runner.run_streamed(agent, input=user_input, session=session, context=fs_context)
            print("Assistant:", end=" ", flush=True)
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
            print("\n")
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n(Interrupted)\n")
            break


if __name__ == "__main__":
    import sys
    # If first arg is 'chat', start interactive mode; otherwise do single-turn
    if len(sys.argv) > 1 and sys.argv[1].lower() == "chat":
        sess_id = sys.argv[2] if len(sys.argv) > 2 else "financial_chat"
        try:
            asyncio.run(interactive_chat(sess_id))
        except KeyboardInterrupt:
            pass
    else:
        user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are my recent expenses?"
        asyncio.run(stream_chat(user_query))
