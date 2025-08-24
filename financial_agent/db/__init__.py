"""Database module for financial_agent.

Contains SQL constants and the default SQLite database file.
"""

# Expose SQL constants at package level if helpful
from .sql import (
    LIST_RECENT_TRANSACTIONS,
    INSERT_TRANSACTION,
    SEARCH_TRANSACTIONS,
    INSERT_MEMORY,
    LIST_MEMORIES,
)
