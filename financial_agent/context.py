from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Optional


DEFAULT_DB_PATH = Path(__file__).parent / "db" / "finance.db"
DOCUMENTS_DIR = Path(__file__).parents[1] / "documents"


@dataclass
class AppConfig:
    openai_api_key: str
    model: str = "gpt-5"
    db_path: Path = DEFAULT_DB_PATH
    documents_dir: Path = DOCUMENTS_DIR


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
        self.conn.commit()


@dataclass
class RunDeps:
    config: AppConfig
    db: DB

    def ensure_ready(self) -> None:
        self.db.init()
        self.config.documents_dir.mkdir(parents=True, exist_ok=True)
