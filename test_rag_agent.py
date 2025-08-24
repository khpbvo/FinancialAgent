#!/usr/bin/env python3
"""Tests for the RAG-enabled financial chat agent."""

import os
import asyncio
import tempfile
from pathlib import Path

# Ensure predictable environment for embeddings
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")

from financial_rag_agent import (
    ingest_document_fn,
    retrieve_financial_context_fn,
    store,
)


def _create_temp_doc(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


async def test_ingest_and_retrieve() -> None:
    # Clear existing records
    store.records.clear()
    store.save()

    path = _create_temp_doc("Salary payment of 5000 and rent 1200 this month.")
    try:
        result = await ingest_document_fn(path)
        assert "Ingested" in result

        context = await retrieve_financial_context_fn("salary")
        assert "salary" in context.lower() or "no documents" not in context.lower()
    finally:
        Path(path).unlink(missing_ok=True)
        store.records.clear()
        store.save()


if __name__ == "__main__":
    asyncio.run(test_ingest_and_retrieve())
