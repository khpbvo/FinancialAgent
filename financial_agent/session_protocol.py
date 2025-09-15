"""Memory abstraction and session protocols for pluggable storage backends."""

from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum


class StorageBackend(str, Enum):
    """Available storage backend types."""

    SQLITE = "sqlite"
    JSON = "json"
    REDIS = "redis"
    MEMORY = "memory"
    POSTGRESQL = "postgresql"


@dataclass
class SessionMessage:
    """A message in a session."""

    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class SessionProtocol(Protocol):
    """Protocol for session storage implementations."""

    async def save_message(self, message: SessionMessage) -> None:
        """Save a message to the session."""
        ...

    async def load_messages(self, limit: Optional[int] = None) -> List[SessionMessage]:
        """Load messages from the session."""
        ...

    async def clear(self) -> None:
        """Clear all messages from the session."""
        ...

    async def get_summary(self) -> str:
        """Get a summary of the session."""
        ...


class BaseSessionStorage(ABC):
    """Base class for session storage implementations."""

    def __init__(self, session_id: str, **kwargs):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.metadata = kwargs

    @abstractmethod
    async def save_message(self, message: SessionMessage) -> None:
        """Save a message to storage."""
        pass

    @abstractmethod
    async def load_messages(self, limit: Optional[int] = None) -> List[SessionMessage]:
        """Load messages from storage."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear the session."""
        pass

    async def get_summary(self) -> str:
        """Generate a summary of the session."""
        messages = await self.load_messages(limit=10)
        if not messages:
            return "Empty session"

        return f"Session {self.session_id}: {len(messages)} messages, started {self.created_at}"


class SQLiteSessionStorage(BaseSessionStorage):
    """SQLite-based session storage."""

    def __init__(self, session_id: str, db_path: str = "sessions.db", **kwargs):
        super().__init__(session_id, **kwargs)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                metadata TEXT
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (id, created_at, metadata)
            VALUES (?, ?, ?)
        """,
            (self.session_id, self.created_at.isoformat(), json.dumps(self.metadata)),
        )
        conn.commit()
        conn.close()

    async def save_message(self, message: SessionMessage) -> None:
        """Save a message to SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                self.session_id,
                message.role,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.metadata) if message.metadata else None,
            ),
        )
        conn.commit()
        conn.close()

    async def load_messages(self, limit: Optional[int] = None) -> List[SessionMessage]:
        """Load messages from SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT role, content, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query, (self.session_id,))
        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            messages.append(
                SessionMessage(
                    role=row["role"],
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                )
            )

        return messages

    async def clear(self) -> None:
        """Clear session messages."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        conn.commit()
        conn.close()


class JSONSessionStorage(BaseSessionStorage):
    """JSON file-based session storage."""

    def __init__(self, session_id: str, storage_dir: str = "sessions", **kwargs):
        super().__init__(session_id, **kwargs)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{session_id}.json"

        if not self.file_path.exists():
            self._save_data([])

    def _save_data(self, messages: List[Dict[str, Any]]) -> None:
        """Save data to JSON file."""
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "messages": messages,
        }
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                return json.load(f)
        return {"messages": []}

    async def save_message(self, message: SessionMessage) -> None:
        """Save a message to JSON."""
        data = self._load_data()
        messages = data.get("messages", [])
        messages.append(
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata,
            }
        )
        self._save_data(messages)

    async def load_messages(self, limit: Optional[int] = None) -> List[SessionMessage]:
        """Load messages from JSON."""
        data = self._load_data()
        messages_data = data.get("messages", [])

        if limit:
            messages_data = messages_data[-limit:]

        messages = []
        for msg in messages_data:
            messages.append(
                SessionMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(msg["timestamp"]),
                    metadata=msg.get("metadata"),
                )
            )

        return messages

    async def clear(self) -> None:
        """Clear session messages."""
        self._save_data([])


class InMemorySessionStorage(BaseSessionStorage):
    """In-memory session storage (for testing/development)."""

    _storage: Dict[str, List[SessionMessage]] = {}

    def __init__(self, session_id: str, **kwargs):
        super().__init__(session_id, **kwargs)
        if session_id not in self._storage:
            self._storage[session_id] = []

    async def save_message(self, message: SessionMessage) -> None:
        """Save a message to memory."""
        self._storage[self.session_id].append(message)

    async def load_messages(self, limit: Optional[int] = None) -> List[SessionMessage]:
        """Load messages from memory."""
        messages = self._storage.get(self.session_id, [])
        if limit:
            return messages[-limit:]
        return messages

    async def clear(self) -> None:
        """Clear session messages."""
        self._storage[self.session_id] = []


class SessionFactory:
    """Factory for creating session storage instances."""

    _backends = {
        StorageBackend.SQLITE: SQLiteSessionStorage,
        StorageBackend.JSON: JSONSessionStorage,
        StorageBackend.MEMORY: InMemorySessionStorage,
    }

    @classmethod
    def create(
        cls, backend: StorageBackend, session_id: str, **kwargs
    ) -> BaseSessionStorage:
        """Create a session storage instance.

        Args:
            backend: The storage backend to use
            session_id: Unique identifier for the session
            **kwargs: Backend-specific configuration

        Returns:
            A session storage instance
        """
        if backend not in cls._backends:
            raise ValueError(f"Unknown backend: {backend}")

        storage_class = cls._backends[backend]
        return storage_class(session_id, **kwargs)

    @classmethod
    def register_backend(
        cls, backend: StorageBackend, storage_class: type[BaseSessionStorage]
    ) -> None:
        """Register a new storage backend.

        Args:
            backend: The backend identifier
            storage_class: The storage class implementation
        """
        cls._backends[backend] = storage_class


class FinancialSessionManager:
    """Manager for financial agent sessions with context preservation."""

    def __init__(
        self, backend: StorageBackend = StorageBackend.SQLITE, **backend_kwargs
    ):
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.sessions: Dict[str, BaseSessionStorage] = {}

    def get_or_create_session(self, session_id: str) -> BaseSessionStorage:
        """Get an existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionFactory.create(
                self.backend, session_id, **self.backend_kwargs
            )
        return self.sessions[session_id]

    async def save_interaction(
        self,
        session_id: str,
        user_input: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a complete interaction to the session."""
        session = self.get_or_create_session(session_id)

        # Save user message
        await session.save_message(
            SessionMessage(
                role="user",
                content=user_input,
                timestamp=datetime.now(),
                metadata=metadata,
            )
        )

        # Save agent response
        await session.save_message(
            SessionMessage(
                role="assistant",
                content=agent_response,
                timestamp=datetime.now(),
                metadata=metadata,
            )
        )

    async def get_context(self, session_id: str, max_messages: int = 10) -> str:
        """Get context from previous messages."""
        session = self.get_or_create_session(session_id)
        messages = await session.load_messages(limit=max_messages)

        if not messages:
            return ""

        context_lines = ["Previous conversation:"]
        for msg in messages:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            context_lines.append(f"{prefix} {msg.content[:100]}...")

        return "\n".join(context_lines)

    async def export_session(self, session_id: str, format: str = "json") -> str:
        """Export a session in the specified format."""
        session = self.get_or_create_session(session_id)
        messages = await session.load_messages()

        if format == "json":
            data = {
                "session_id": session_id,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata,
                    }
                    for msg in messages
                ],
            }
            return json.dumps(data, indent=2)

        elif format == "text":
            lines = [f"Session: {session_id}", "=" * 50]
            for msg in messages:
                lines.append(
                    f"\n[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {msg.role.upper()}"
                )
                lines.append(msg.content)
                lines.append("-" * 30)
            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown export format: {format}")


# Example of extending with Redis backend
class RedisSessionStorage(BaseSessionStorage):
    """Redis-based session storage (example implementation)."""

    def __init__(
        self, session_id: str, redis_url: str = "redis://localhost:6379", **kwargs
    ):
        super().__init__(session_id, **kwargs)
        # This would require redis-py package
        # import redis
        # self.redis = redis.from_url(redis_url)
        self.redis_url = redis_url
        self.messages = []  # Simplified for example

    async def save_message(self, message: SessionMessage) -> None:
        """Save a message to Redis."""
        # Actual implementation would use Redis
        self.messages.append(message)

    async def load_messages(self, limit: Optional[int] = None) -> List[SessionMessage]:
        """Load messages from Redis."""
        # Actual implementation would query Redis
        if limit:
            return self.messages[-limit:]
        return self.messages

    async def clear(self) -> None:
        """Clear session messages from Redis."""
        # Actual implementation would clear Redis keys
        self.messages = []


# Register the Redis backend
SessionFactory.register_backend(StorageBackend.REDIS, RedisSessionStorage)
