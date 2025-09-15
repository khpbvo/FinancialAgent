"""
Logged versions of all Financial Agent tools.

This module wraps all tools with logging decorators to capture detailed
execution information for debugging and performance analysis.
"""

# Import and wrap all tools with logging
from .tools.ingest import ingest_csv
from .tools.excel_ingest import ingest_excel, list_excel_sheets
from .tools.query import list_recent_transactions, search_transactions
from .tools.advice import analyze_and_advise
from .tools.pdf_ingest import ingest_pdfs
from .tools.memory import list_memories
from .tools.summarize import summarize_file, summarize_overview
from .tools.records import add_transaction
from .tools.budgets import (
    set_budget,
    check_budget,
    list_budgets,
    suggest_budgets,
    delete_budget,
)
from .tools.goals import (
    create_goal,
    update_goal_progress,
    check_goals,
    suggest_savings_plan,
    complete_goal,
    pause_goal,
)
from .tools.recurring import (
    detect_recurring,
    list_subscriptions,
    analyze_subscription_value,
    predict_next_recurring,
    export_clean_monthly_recurring,
)
from .tools.export import (
    export_transactions,
    generate_tax_report,
    export_budget_report,
    export_recurring_payments,
)
from .tools.costs import monthly_cost_summary

# Apply logging decorators to all tools
# TEMPORARILY DISABLED: logging decorators break function_tool attributes
logged_ingest_csv = ingest_csv  # log_tool_calls(ingest_csv)
logged_ingest_excel = ingest_excel  # log_tool_calls(ingest_excel)
logged_list_excel_sheets = list_excel_sheets  # log_tool_calls(list_excel_sheets)
logged_ingest_pdfs = ingest_pdfs  # log_tool_calls(ingest_pdfs)

logged_list_recent_transactions = (
    list_recent_transactions  # log_tool_calls(list_recent_transactions)
)
logged_search_transactions = search_transactions  # log_tool_calls(search_transactions)
logged_list_memories = list_memories  # log_tool_calls(list_memories)

logged_set_budget = set_budget  # log_tool_calls(set_budget)
logged_check_budget = check_budget  # log_tool_calls(check_budget)
logged_list_budgets = list_budgets  # log_tool_calls(list_budgets)
logged_suggest_budgets = suggest_budgets  # log_tool_calls(suggest_budgets)
logged_delete_budget = delete_budget  # log_tool_calls(delete_budget)

logged_create_goal = create_goal  # log_tool_calls(create_goal)
logged_update_goal_progress = (
    update_goal_progress  # log_tool_calls(update_goal_progress)
)
logged_check_goals = check_goals  # log_tool_calls(check_goals)
logged_suggest_savings_plan = (
    suggest_savings_plan  # log_tool_calls(suggest_savings_plan)
)
logged_complete_goal = complete_goal  # log_tool_calls(complete_goal)
logged_pause_goal = pause_goal  # log_tool_calls(pause_goal)

logged_detect_recurring = detect_recurring  # log_tool_calls(detect_recurring)
logged_list_subscriptions = list_subscriptions  # log_tool_calls(list_subscriptions)
logged_analyze_subscription_value = (
    analyze_subscription_value  # log_tool_calls(analyze_subscription_value)
)
logged_predict_next_recurring = (
    predict_next_recurring  # log_tool_calls(predict_next_recurring)
)
logged_export_clean_monthly_recurring = (
    export_clean_monthly_recurring  # log_tool_calls(export_clean_monthly_recurring)
)

logged_export_transactions = export_transactions  # log_tool_calls(export_transactions)
logged_generate_tax_report = generate_tax_report  # log_tool_calls(generate_tax_report)
logged_export_budget_report = (
    export_budget_report  # log_tool_calls(export_budget_report)
)
logged_export_recurring_payments = (
    export_recurring_payments  # log_tool_calls(export_recurring_payments)
)
logged_monthly_cost_summary = (
    monthly_cost_summary  # log_tool_calls(monthly_cost_summary)
)

logged_analyze_and_advise = analyze_and_advise  # log_tool_calls(analyze_and_advise)
logged_summarize_file = summarize_file  # log_tool_calls(summarize_file)
logged_summarize_overview = summarize_overview  # log_tool_calls(summarize_overview)
logged_add_transaction = add_transaction  # log_tool_calls(add_transaction)

# Create a list of all logged tools for easy import
ALL_LOGGED_TOOLS = [
    # Ingestion tools
    logged_ingest_csv,
    logged_ingest_excel,
    logged_list_excel_sheets,
    logged_ingest_pdfs,
    # Query tools
    logged_list_recent_transactions,
    logged_search_transactions,
    logged_list_memories,
    # Budget tools
    logged_set_budget,
    logged_check_budget,
    logged_list_budgets,
    logged_suggest_budgets,
    logged_delete_budget,
    # Goal tracking tools
    logged_create_goal,
    logged_update_goal_progress,
    logged_check_goals,
    logged_suggest_savings_plan,
    logged_complete_goal,
    logged_pause_goal,
    # Recurring transaction tools
    logged_detect_recurring,
    logged_list_subscriptions,
    logged_analyze_subscription_value,
    logged_predict_next_recurring,
    logged_export_clean_monthly_recurring,
    # Export tools
    logged_export_transactions,
    logged_generate_tax_report,
    logged_export_budget_report,
    logged_export_recurring_payments,
    # Analysis tools
    logged_analyze_and_advise,
    logged_summarize_file,
    logged_summarize_overview,
    logged_add_transaction,
    logged_monthly_cost_summary,
]

# Create dictionary mapping for easy lookup
# TEMPORARILY DISABLED: FunctionTool objects don't have __name__ attribute
# LOGGED_TOOLS_MAP = {
#     tool.__name__.replace('logged_', ''): tool for tool in ALL_LOGGED_TOOLS
# }
