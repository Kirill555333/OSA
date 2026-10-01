"""Session model and store for OSA conversation runtime."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping
import uuid

from osa.models import ChatMessage, ToolCall


class SessionError(RuntimeError):
    """Base exception for session operations."""


class SessionNotFoundError(SessionError, KeyError):
    """Raised when a requested session is not found."""


class Session:
    """Represents one discrete conversation session with message history and metadata."""

    def __init__(
        self,
        session_id: str | None = None,
        *,
        system_prompt: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        raw_id = session_id if session_id is not None else f"session-{uuid.uuid4().hex[:12]}"
        normalized_id = raw_id.strip()

        if not normalized_id:
            raise SessionError("session_id cannot be empty.")

        self._session_id = normalized_id
        self._system_prompt = system_prompt.strip() if system_prompt else None
        now_ts = self._current_timestamp()
        self._created_at = created_at or now_ts
        self._updated_at = updated_at or now_ts

        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self._messages: list[ChatMessage] = []

        if self._system_prompt:
            self._messages.append(
                ChatMessage(
                    role="system",
                    content=self._system_prompt,
                )
            )

    @property
    def session_id(self) -> str:
        """Unique identifier of the conversation session."""
        return self._session_id

    @property
    def system_prompt(self) -> str | None:
        """Initial system prompt of the session, if configured."""
        return self._system_prompt

    @property
    def created_at(self) -> str:
        """UTC ISO timestamp of session creation."""
        return self._created_at

    @property
    def updated_at(self) -> str:
        """UTC ISO timestamp of last update."""
        return self._updated_at

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Immutable view of session metadata."""
        return MappingProxyType(self._metadata)

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session message history."""
        self._messages.append(message)
        self._updated_at = self._current_timestamp()

    def remove_last_message(self) -> ChatMessage:
        """Remove and return the most recent message."""
        if not self._messages:
            raise IndexError("Session messages context is empty.")

        msg = self._messages.pop()
        self._updated_at = self._current_timestamp()
        return msg

    def messages(self) -> tuple[ChatMessage, ...]:
        """Return an immutable snapshot of all messages in this session."""
        return tuple(self._messages)

    def restore_messages(self, messages: Iterable[ChatMessage]) -> None:
        """Restore message history from an iterable snapshot."""
        self._messages = list(messages)
        self._updated_at = self._current_timestamp()

    def clear(self) -> None:
        """Clear conversation messages while preserving the system prompt if configured."""
        self._messages.clear()
        if self._system_prompt:
            self._messages.append(
                ChatMessage(
                    role="system",
                    content=self._system_prompt,
                )
            )
        self._updated_at = self._current_timestamp()

    def update_metadata(self, updates: Mapping[str, Any]) -> None:
        """Merge updates into session metadata."""
        self._metadata.update(updates)
        self._updated_at = self._current_timestamp()

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state into a plain dictionary."""
        return {
            "session_id": self._session_id,
            "system_prompt": self._system_prompt,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "metadata": dict(self._metadata),
            "messages": [self._serialize_message(msg) for msg in self._messages],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Session:
        """Restore a Session instance from serialized data."""
        session_id = data.get("session_id")
        if not isinstance(session_id, str):
            raise SessionError("Serialized session missing string 'session_id'.")

        system_prompt = data.get("system_prompt")
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        metadata = data.get("metadata")

        session = cls(
            session_id=session_id,
            system_prompt=system_prompt if isinstance(system_prompt, str) else None,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
            created_at=str(created_at) if created_at else None,
            updated_at=str(updated_at) if updated_at else None,
        )

        raw_messages = data.get("messages", [])
        if isinstance(raw_messages, list):
            deserialized = [cls._deserialize_message(m) for m in raw_messages]
            session.restore_messages(deserialized)

        return session

    def __len__(self) -> int:
        return len(self._messages)

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_call_id is not None:
            data["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": dict(tc.arguments),
                }
                for tc in message.tool_calls
            ]
        return data

    @staticmethod
    def _deserialize_message(data: Any) -> ChatMessage:
        if not isinstance(data, Mapping):
            raise SessionError("Message item in serialized data must be an object.")

        role = data.get("role")
        content = data.get("content", "")
        tool_call_id = data.get("tool_call_id")
        raw_tool_calls = data.get("tool_calls", [])

        parsed_tool_calls: list[ToolCall] = []
        if isinstance(raw_tool_calls, list):
            for tc in raw_tool_calls:
                if isinstance(tc, Mapping):
                    parsed_tool_calls.append(
                        ToolCall(
                            id=str(tc.get("id", "")),
                            name=str(tc.get("name", "")),
                            arguments=dict(tc.get("arguments", {})),
                        )
                    )

        return ChatMessage(
            role=role,
            content=str(content),
            tool_call_id=str(tool_call_id) if tool_call_id is not None else None,
            tool_calls=tuple(parsed_tool_calls),
        )


class SessionStore:
    """In-memory store managing active and historical conversation sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        session_id: str | None = None,
        *,
        system_prompt: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        """Create and register a new session."""
        session = Session(
            session_id=session_id,
            system_prompt=system_prompt,
            metadata=metadata,
        )

        if session.session_id in self._sessions:
            raise SessionError(f"Session '{session.session_id}' already exists.")

        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        """Retrieve a session by its identifier."""
        norm_id = session_id.strip()
        session = self._sessions.get(norm_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{norm_id}' was not found.")
        return session

    def get_or_create(
        self,
        session_id: str,
        *,
        system_prompt: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        """Retrieve an existing session or create a new one if absent."""
        norm_id = session_id.strip()
        if not norm_id:
            raise SessionError("session_id cannot be empty.")

        if norm_id in self._sessions:
            return self._sessions[norm_id]

        return self.create(
            session_id=norm_id,
            system_prompt=system_prompt,
            metadata=metadata,
        )

    def has(self, session_id: str) -> bool:
        """Check whether a session exists."""
        return session_id.strip() in self._sessions

    def list(self) -> tuple[Session, ...]:
        """Return all sessions sorted by updated_at descending."""
        return tuple(
            sorted(
                self._sessions.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )
        )

    def delete(self, session_id: str) -> None:
        """Delete a session by ID."""
        norm_id = session_id.strip()
        if norm_id not in self._sessions:
            raise SessionNotFoundError(f"Session '{norm_id}' was not found.")
        del self._sessions[norm_id]

    def count(self) -> int:
        """Return the number of stored sessions."""
        return len(self._sessions)
