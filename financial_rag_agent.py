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
from typing import Any, List, Dict

from agents import Agent, Runner, function_tool
from agents.tool import WebSearchTool, FileSearchTool, LocalShellTool
# Import model class explicitly to avoid __all__ restrictions
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI
from pydantic import BaseModel

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


# ---------------------------------------------------------------------------
# Agent Creation and Streaming Chat
# ---------------------------------------------------------------------------

def create_chat_agent() -> Agent:
    """Create a chat agent with standard tools and retrieval"""
    tools = [
        ingest_document,
        retrieve_financial_context,
        WebSearchTool(),
        FileSearchTool(),
        LocalShellTool(),
    ]

    instructions = (
        "You are a financial assistant. Use the provided tools to read files, "
        "search the web, and fetch context from documents. Always call "
        "retrieve_financial_context before answering user financial questions."
    )

    return Agent(
        name="financial_chat_agent",
        model=OpenAIResponsesModel(model=settings.OPENAI_MODEL),
        instructions=instructions,
        tools=tools,
    )


async def stream_chat(query: str) -> None:
    """Run the chat agent with streaming output"""
    agent = create_chat_agent()
    async for event in Runner.stream(agent, query):
        if event.get("type") == "response.output_text.delta":
            print(event["delta"], end="", flush=True)
    print()


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are my recent expenses?"
    asyncio.run(stream_chat(user_query))
