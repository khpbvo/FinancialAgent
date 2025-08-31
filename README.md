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

## Bills-only recurring exports
- Parse PDFs into transactions with LLM classification (GPT-5):
  - Interactive: type `bootstrap` to ingest `documents/` (CSV/PDF/Excel). PDFs are classified and parsed into transactions (credit card + bank statements). Credit card repayments are skipped to avoid double counting.
  - Programmatic: use `ingest_pdfs` (defaults to transactions mode, GPT-5 classification on).

- Export only monthly bills (exclude credit-card repayment, POS-like entries):
  - Interactive commands:
    - `bills` → CSV
    - `bills-pdf` → PDF
    - `bills-excel` → Excel
  - CLI flags:
    - `financial-agent --export-bills --bills-format pdf|excel|csv|json`

- Notes:
  - The export uses confidence thresholds and pattern matching to keep subscriptions/utilities/insurance/telecom/government bills, and excludes POS-like merchants and fees/cash withdrawals by default.
  - Amount tolerance is higher (40%) for variable bills (e.g., utilities/insurance/telecom) to capture month-to-month fluctuations.


## Notes
- Model: set via env `FIN_AGENT_MODEL` (defaults to `gpt-5` as per request).
- DB path: `financial_agent/db/finance.db` by default.
