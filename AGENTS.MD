**Overview**
- Purpose: Comprehensive financial assistant that ingests CSV/PDF/Excel, analyzes transactions, manages budgets/goals, detects recurring payments, generates reports, and coordinates specialist agents for expert advice.
- Architecture: Orchestrator agent with five specialists (Tax, Budget, Goal, Investment, Debt), a legacy monolithic agent kept for compatibility, and document analyzers for CSV/PDF. Built on the Agents SDK patterns (see Docs/).

**Core Agents**
- Financial Orchestrator: Primary entrypoint that routes queries, runs core tools, and hands off to specialists when appropriate. Uses `gpt-5` with reasoning effort high and verbosity high.
- Legacy Financial Agent: A single agent variant with the full toolset and dynamic instructions; stops after certain tools via `StopAtTools` to return tool output directly.

**Orchestrator Details**
- Instructions: Coordinates specialists, handles general data operations, synthesizes multi-specialist outputs, and follows domain routing rules (tax, budget/spending, goals, investments, debt).
- Handoffs: Provides function tools to hand off to: TaxSpecialist, BudgetSpecialist, GoalSpecialist, InvestmentSpecialist, DebtSpecialist.
- Routing Tools:
  - `route_user_query(user_query)`: Analyzes intent and recommends routing; scores domains by keyword.
  - `coordinate_multi_specialist_analysis(user_query, specialists_needed)`: Runs multiple specialists and synthesizes recommendations.
- Core Tools (non-exhaustive): `ingest_csv`, `ingest_excel`, `list_excel_sheets`, `ingest_pdfs`, `list_recent_transactions`, `search_transactions`, `list_memories`, `summarize_file`, `summarize_overview`, `add_transaction`, `export_transactions`, `export_recurring_payments`, `export_clean_monthly_recurring`.

**Legacy Agent Details**
- Dynamic Instructions: Built from current DB state (transactions/memories), adds context such as spending totals and guidance when the DB is empty.
- Tool Behavior: `StopAtTools(stop_at_tool_names=["add_transaction", "ingest_csv", "ingest_pdfs"])` to return first tool results as final output without additional LLM calls.
- Tools: Full suite including ingestion, query, memory, budgets, goals, recurring detection, exports, analysis, summaries, and record edits.

**Specialist Agents**
- TaxSpecialist:
  - Focus: Deductions, tax reports, timing advice, categorization for compliance.
  - Tools: `generate_tax_report`, `analyze_tax_deductions(year, include_estimates)`, `suggest_tax_timing(current_month)`, `export_transactions`.
- BudgetSpecialist:
  - Focus: Spending analysis, budget planning, recurring/subscriptions optimization, alerts.
  - Tools: `set_budget`, `check_budget`, `list_budgets`, `suggest_budgets`, `delete_budget`, `analyze_spending_patterns`, `create_smart_budget_plan`, `budget_alert_system`, `detect_recurring`, `list_subscriptions`, `analyze_subscription_value`.
- GoalSpecialist:
  - Focus: SMART goals, milestones, motivation, resource allocation, timelines.
  - Tools: `create_goal`, `update_goal_progress`, `check_goals`, `suggest_savings_plan`, `complete_goal`, `pause_goal`, `create_comprehensive_financial_plan`, `goal_motivation_coach`, `optimize_goal_strategy`.
- InvestmentSpecialist:
  - Focus: Readiness checks, projections, allocation/rebalancing, education; coordinates with Tax/Goal.
  - Tools: `analyze_investment_readiness`, `calculate_investment_projections`, `portfolio_rebalancing_advisor`.
- DebtSpecialist:
  - Focus: Avalanche/snowball planning, consolidation analysis, payoff timelines, motivation.
  - Tools: `analyze_debt_situation(debts, monthly_payment_capacity)`, `create_debt_freedom_plan(target_date, aggressive_mode)`, `debt_consolidation_analyzer(...)`.

**Document Analyzers**
- CSVAnalyzer Agent: Specialized analysis of CSV structure, column mapping, formats; returns `CSVAnalysisResult` structured output.
- PDFAnalyzer Agent: Extracts information from bank statement PDFs; returns `PDFAnalysisResult` structured output.
- Helper Tools: `analyze_csv_with_agent(file_path, preview_rows)`, `analyze_pdf_with_agent(file_path, max_pages)` to invoke analyzers via tools.

**Key Tools By Domain**
- Ingestion: `ingest_csv(path, csv_map)`, `ingest_excel(path)`, `list_excel_sheets(path)`, `ingest_pdfs(path_or_dir)`.
- Query & Memory: `list_recent_transactions(limit)`, `search_transactions(...)`, `list_memories()`.
- Summaries & Advice: `summarize_file(path)`, `summarize_overview()`, `analyze_and_advise()`.
- Records: `add_transaction(date, description, amount, currency, category)`.
- Budgets: `set_budget`, `check_budget(period_days)`, `list_budgets`, `suggest_budgets(months_back)`, `delete_budget(category)`.
- Goals: `create_goal`, `update_goal_progress`, `check_goals`, `suggest_savings_plan`, `complete_goal`, `pause_goal`.
- Recurring/Subscriptions: `detect_recurring(min_occurrences, lookback_months)`, `list_subscriptions()`, `analyze_subscription_value()`, `predict_next_recurring(days_ahead)`, `export_clean_monthly_recurring(format, months_required, include_all_bills)`.
- Export & Reports: `export_transactions(format, start_date, end_date, category, output_file)`, `export_recurring_payments`, `generate_tax_report(year, format)`, `export_budget_report`.

**Context & Storage**
- AppConfig: `OPENAI_API_KEY`, `FIN_AGENT_MODEL` (default `gpt-5`), `db_path` (defaults to `financial_agent/db/finance.db`), `documents_dir` (defaults to `documents/`).
- Database Tables:
  - `transactions(id, date, description, amount, currency, category, source_file)`
  - `memories(id, created_at, kind, content, tags)`
  - `budgets(id, category UNIQUE, amount, period, created_at, updated_at)`
  - `goals(id, name, target_amount, current_amount, target_date, category, status, created_at, updated_at)`
  - `recurring_transactions(id, description_pattern, amount, frequency, category, last_seen, confidence, created_at)`
  - `agent_messages`, `agent_sessions` (Agents SDK session storage)
- Initialization: `RunDeps.ensure_ready()` prepares DB and documents dir; ingestion tools write into `transactions` and associated tables.

**Models & Settings**
- Model: All agents default to `gpt-5` with `ModelSettings(reasoning=Reasoning(effort="high"), verbosity="high")` for rich reasoning traces and clearer outputs.
- Tool Behavior:
  - Orchestrator: Standard “run LLM again” after tools; decides on handoffs and synthesis.
  - Legacy Agent: Uses `StopAtTools` for selected tools to minimize extra LLM calls.
- Dynamic Instructions: Main agent instructions are generated at runtime to incorporate DB state and give actionable guidance when empty.

**Handoffs & Routing**
- Handoff Tools: `handoff_to_tax_specialist`, `handoff_to_budget_specialist`, `handoff_to_goal_specialist`, `handoff_to_investment_specialist`, `handoff_to_debt_specialist` — each logs routing, passes optional additional context, executes the specialist, and returns a formatted response.
- Routing Heuristics: `analyze_query_intent(user_query)` scores keywords per domain; `route_user_query` reports intent, scores, and a recommended path; multi-domain queries can call `coordinate_multi_specialist_analysis`.

**Running & CLI**
- Entry Points: `financial_agent/cli.py` and `run_financial_agent.sh`.
- Modes:
  - Interactive: `financial-agent` (streams reasoning and tool progress; session memory on by default).
  - Single command streaming: `financial-agent --stream -i "Analyze my spending"`.
  - Single command non-streaming: `financial-agent -i "Generate a tax report for 2024"`.
- Helpers:
  - Bootstrap docs: `financial-agent --bootstrap` (ingest from `documents/`).
  - Sessions: `--no-session` disables; session DB stored under `~/.financial_agent/sessions.db`.
  - Maintenance: `--clear-db`, `--clear-sessions`, `--clear-all`, and interactive commands `clear`, `clear-db`, `clear-sessions`, `clear-all`, `history`.
- Env Vars: `OPENAI_API_KEY` required; `FIN_AGENT_MODEL` optional; see README.md for quick start.

**Error Handling & Reliability**
- Tool Error Handlers: Custom failure handlers for CSV ingestion and DB queries provide user-friendly messages and prevent crashes.
- Logging: Central logger plus OpenAI request/response loggers (`OpenAILogger` and `EnhancedOpenAILogger`) annotate runs with session IDs and system events.
- Indices: DB schema includes indices for date/category/amount to keep queries responsive.

**Best Practices (from Docs/)**
- Use specialized agents with clear instructions; rely on handoffs for domain depth.
- Prefer structured outputs for extractors (CSV/PDF analyzers use Pydantic output types).
- Keep prompts explicit about tool availability and expected parameters.
- Iterate with tracing and results inspection; use Sessions for conversation continuity.

**Common Flows**
- New user setup:
  - Bootstrap documents or run `ingest_csv`/`ingest_pdfs`.
  - Ask Orchestrator to “summarize my finances and recommend next actions”.
- Spending optimization:
  - Orchestrator routes to BudgetSpecialist → run `analyze_spending_patterns` → set budgets and `detect_recurring` → `analyze_subscription_value` → export clean recurring.
- Tax prep:
  - Route to TaxSpecialist → `analyze_tax_deductions(year)` → `generate_tax_report(year, pdf|excel)` → `export_transactions` if needed.
- Goal planning:
  - GoalSpecialist → `create_comprehensive_financial_plan` → create goals and schedule reviews.
- Debt strategy:
  - DebtSpecialist → `analyze_debt_situation` and compare avalanche vs snowball → optional consolidation analysis → action plan.

**References**
- SDK Concepts: see Docs/agents.md, Docs/handoffs.md, Docs/tools.md, Docs/running_agents.md, Docs/sessions.md, Docs/tracing.md, Docs/streaming.md for deeper details.

