from __future__ import annotations

LIST_RECENT_TRANSACTIONS = """
SELECT date, description, amount, currency, category, source_file
FROM transactions
ORDER BY date DESC, id DESC
LIMIT ?
"""

INSERT_TRANSACTION = """
INSERT INTO transactions (date, description, amount, currency, category, source_file)
VALUES (?, ?, ?, COALESCE(?, 'EUR'), ?, ?)
"""

SEARCH_TRANSACTIONS = """
SELECT date, description, amount, currency, category, source_file
FROM transactions
WHERE (
    (? IS NULL OR date >= ?) AND
    (? IS NULL OR date <= ?) AND
    (? IS NULL OR category = ?) AND
    (? IS NULL OR description LIKE '%' || ? || '%')
)
ORDER BY date ASC, id ASC
LIMIT ?
"""

INSERT_MEMORY = """
INSERT INTO memories (kind, content, tags) VALUES (?, ?, ?)
"""

LIST_MEMORIES = """
SELECT id, created_at, kind, content, tags
FROM memories
ORDER BY created_at DESC, id DESC
LIMIT ?
"""
