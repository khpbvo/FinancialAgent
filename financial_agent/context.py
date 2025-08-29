from pathlib import Path
import sqlite3
from typing import Optional
from pydantic import BaseModel, Field


DEFAULT_DB_PATH = Path(__file__).parent / "db" / "finance.db"
DOCUMENTS_DIR = Path(__file__).parents[1] / "documents"


class AppConfig(BaseModel):
    """Application configuration with validation."""
    openai_api_key: str = Field(description="OpenAI API key")
    model: str = Field(default="gpt-5", description="Model to use")
    db_path: Path = Field(default=DEFAULT_DB_PATH, description="Database path")
    documents_dir: Path = Field(default=DOCUMENTS_DIR, description="Documents directory")
    
    class Config:
        arbitrary_types_allowed = True


class DB:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                description TEXT,
                amount REAL,
                currency TEXT DEFAULT 'EUR',
                category TEXT,
                source_file TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                kind TEXT, -- summary|insight|advice
                content TEXT,
                tags TEXT
            )
            """
        )
        
        # Create budgets table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                period TEXT DEFAULT 'monthly', -- monthly|weekly|yearly
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        
        # Create financial goals table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL DEFAULT 0,
                target_date TEXT,
                category TEXT, -- savings|debt_reduction|investment
                status TEXT DEFAULT 'active', -- active|completed|paused
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        
        # Create recurring transactions table for detection
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recurring_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description_pattern TEXT NOT NULL,
                amount REAL,
                frequency TEXT, -- daily|weekly|monthly|yearly
                category TEXT,
                last_seen TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        
        # Create agent_messages table required by Agents SDK for session management
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL, -- user|assistant|system|tool
                content TEXT,
                tool_calls TEXT, -- JSON array of tool calls
                tool_call_id TEXT,
                name TEXT, -- tool name for tool messages
                created_at TEXT DEFAULT (datetime('now')),
                metadata TEXT -- JSON for additional data
            )
            """
        )
        
        # Create agent_sessions table for session management
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                metadata TEXT -- JSON for session metadata
            )
            """
        )
        
        # Add indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_budgets_category ON budgets(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_messages_created ON agent_messages(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated ON agent_sessions(updated_at)")
        
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection if it's open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RunDeps(BaseModel):
    """Runtime dependencies with validation."""
    config: AppConfig = Field(description="Application configuration")
    db: DB = Field(description="Database connection")
    
    class Config:
        arbitrary_types_allowed = True

    def ensure_ready(self) -> None:
        """Ensure all dependencies are ready."""
        self.db.init()
        self.config.documents_dir.mkdir(parents=True, exist_ok=True)
