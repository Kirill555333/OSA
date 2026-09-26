"""Persistent long-term memory for OSA."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import unicodedata


class MemoryError(RuntimeError):
    """Base exception for memory-related errors."""


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One persistent OSA memory."""

    id: int
    content: str
    category: str
    importance: int
    created_at: str
    updated_at: str


class LongTermMemory:
    """Store and retrieve persistent memories using SQLite."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path.expanduser()

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the SQLite database path."""
        return self._database_path

    def save(
        self,
        content: str,
        *,
        category: str = "general",
        importance: int = 5,
    ) -> MemoryRecord:
        """Save a new memory."""
        normalized_content = content.strip()
        normalized_category = category.strip() or "general"

        if not normalized_content:
            raise ValueError("Memory content cannot be empty.")

        if not 1 <= importance <= 10:
            raise ValueError("importance must be between 1 and 10.")

        timestamp = self._timestamp()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    content,
                    category,
                    importance,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    normalized_content,
                    normalized_category,
                    importance,
                    timestamp,
                    timestamp,
                ),
            )

            memory_id = cursor.lastrowid

        if memory_id is None:
            raise MemoryError("Failed to create memory.")

        return self.get(memory_id)

    def get(self, memory_id: int) -> MemoryRecord:
        """Return a memory by ID."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    content,
                    category,
                    importance,
                    created_at,
                    updated_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Memory '{memory_id}' does not exist.")

        return self._row_to_record(row)

    def recent(
        self,
        limit: int = 10,
    ) -> tuple[MemoryRecord, ...]:
        """Return the most recently updated memories."""
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    content,
                    category,
                    importance,
                    created_at,
                    updated_at
                FROM memories
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return tuple(
            self._row_to_record(row)
            for row in rows
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[MemoryRecord, ...]:
        """
        Search memories using Unicode-aware keyword matching.

        Every meaningful word in the query must be present in either
        the memory content or its category.
        """
        normalized_query = self._normalize_text(query)

        if not normalized_query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        terms = tuple(
            dict.fromkeys(
                term
                for term in normalized_query.split()
                if term
            )
        )

        if not terms:
            raise ValueError(
                "Search query must contain at least one word."
            )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    content,
                    category,
                    importance,
                    created_at,
                    updated_at
                FROM memories
                ORDER BY importance DESC, updated_at DESC, id DESC
                """
            ).fetchall()

        matches: list[MemoryRecord] = []

        for row in rows:
            record = self._row_to_record(row)

            searchable_text = (
                f"{record.content} {record.category}"
            )

            normalized_text = self._normalize_text(
                searchable_text
            )

            if all(
                term in normalized_text
                for term in terms
            ):
                matches.append(record)

                if len(matches) >= limit:
                    break

        return tuple(matches)

    def delete(self, memory_id: int) -> None:
        """Delete a memory by ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )

        if cursor.rowcount == 0:
            raise KeyError(
                f"Memory '{memory_id}' does not exist."
            )

    def count(self) -> int:
        """Return the number of stored memories."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()

        return int(row[0])

    def _initialize_database(self) -> None:
        """Create the memory database schema."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_updated_at
                ON memories(updated_at)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        connection = sqlite3.connect(
            self._database_path,
        )

        connection.row_factory = sqlite3.Row

        return connection

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for Unicode-aware case-insensitive search."""
        return unicodedata.normalize(
            "NFKC",
            value,
        ).casefold()

    @staticmethod
    def _timestamp() -> str:
        """Return the current UTC timestamp."""
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> MemoryRecord:
        """Convert a SQLite row into a MemoryRecord."""
        return MemoryRecord(
            id=int(row["id"]),
            content=str(row["content"]),
            category=str(row["category"]),
            importance=int(row["importance"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
