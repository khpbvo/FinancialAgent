# Financial Agent

A Python agent that analyzes and summarizes financial documents (PDF/CSV) and stores structured memory in SQLite for retrieval and future advice. Built against the OpenAI Agents SDK patterns documented in `Docs/`.

## Features
- Ingest PDFs/CSVs from `documents/` into a SQLite DB
- Tools: ingest files, query transactions, summarize findings, and provide budget advice
- Uses streaming to show progress
- Context object wires dependencies (DB, file paths, OpenAI model config)

## Quick start
1. Set environment:
   - `export OPENAI_API_KEY=...`
2. Install:
   - `pip install -e .`
3. Run the agent (interactive):
   - `financial-agent`  

## Notes
- Model: set via env `FIN_AGENT_MODEL` (defaults to `gps-5` as per request).
- DB path: `financial_agent/db/finance.db` by default.
