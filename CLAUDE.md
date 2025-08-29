# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial Agent is a Python-based AI assistant that analyzes and summarizes financial documents (PDFs and CSVs) and stores structured memory in SQLite for retrieval and future advice. The agent is built using the OpenAI Agents SDK patterns.

## Development Commands

### Installation
```bash
# Install in development mode
pip install -e .
```

### Running the Application
```bash
# Interactive mode (default)
financial-agent

# With specific input
financial-agent -i "Show me recent transactions"

# Streaming mode
financial-agent --stream -i "Analyze my spending"

# Bootstrap documents from documents/ folder
financial-agent --bootstrap

# Alternative: using shell script
./run_financial_agent.sh
```

### Testing
```bash
# Run smoke test
python financial_agent/smoke_test.py

# Test GPT-5 integration
python test_gpt5.py

# Test OpenAI connection
python test_openai.py
```

## Architecture

### Core Components

**Dependency Injection Pattern**: The codebase uses a context-based dependency injection pattern where `RunDeps` carries configuration and database connections through the system.

**Agent Construction**: The `build_agent()` function in `financial_agent/agent.py` creates an agent with specific tools and minimal ModelSettings for GPT-5 compatibility.

**Database Layer**: SQLite database (`financial_agent/db/finance.db`) stores:
- `transactions` table: financial transaction records
- `memories` table: AI-generated summaries, insights, and advice

### Tool System

Tools are implemented as functions decorated with `@function_tool` that accept a `RunContextWrapper[RunDeps]`:
- **ingest_csv**: Processes CSV files (auto-detects ING bank format)
- **ingest_pdfs**: Extracts text from PDF bank statements
- **query tools**: list_recent_transactions, search_transactions
- **analysis tools**: analyze_and_advise, summarize_file, summarize_overview
- **memory tools**: list_memories, add_transaction

### CLI Modes

The CLI (`financial_agent/cli.py`) supports three execution modes:
1. **Interactive Mode**: Real-time chat with streaming responses
2. **Streaming Mode**: Single command with progressive output
3. **Sync Mode**: Simple request-response without streaming

### Document Processing

Documents are expected in the `documents/` directory. The system handles:
- ING bank CSV exports (auto-detected by specific column headers)
- PDF bank statements (text extraction via PyPDF2)
- Generic CSV formats with standard columns

## Key Implementation Details

- Model configuration via `FIN_AGENT_MODEL` environment variable (defaults to "gpt-5")
- OpenAI API key required via `OPENAI_API_KEY` environment variable
- Database auto-initialization on first run
- Streaming events handled through async iteration with ResponseTextDeltaEvent
- Transaction amounts parsed with European format support (comma as decimal separator)