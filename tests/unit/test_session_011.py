"""Unit tests for Session and SessionStore (0.8.11.2)."""

from __future__ import annotations

import pytest

from osa.context import (
    Session,
    SessionError,
    SessionNotFoundError,
    SessionStore,
)
from osa.models import ChatMessage, ToolCall


def test_session_creation_defaults_and_validation() -> None:
    session = Session()

    assert session.session_id.startswith("session-")
    assert session.system_prompt is None
    assert len(session.messages()) == 0
    assert len(session) == 0
    assert session.created_at
    assert session.updated_at

    with pytest.raises(SessionError, match="session_id cannot be empty"):
        Session("")

    with pytest.raises(SessionError, match="session_id cannot be empty"):
        Session("   ")


def test_session_with_system_prompt_and_metadata() -> None:
    session = Session(
        session_id="custom-1",
        system_prompt="You are OSA assistant.",
        metadata={"user_id": "kirill", "mode": "chat"},
    )

    assert session.session_id == "custom-1"
    assert session.system_prompt == "You are OSA assistant."
    assert session.metadata["user_id"] == "kirill"
    assert len(session.messages()) == 1
    assert session.messages()[0].role == "system"
    assert session.messages()[0].content == "You are OSA assistant."


def test_session_messages_lifecycle() -> None:
    session = Session(session_id="s1", system_prompt="Sys")
    initial_updated = session.updated_at

    user_msg = ChatMessage(role="user", content="Hello!")
    session.add_message(user_msg)

    assert len(session.messages()) == 2
    assert session.messages()[1] == user_msg
    assert session.updated_at >= initial_updated

    popped = session.remove_last_message()
    assert popped == user_msg
    assert len(session.messages()) == 1

    session.clear()
    # System prompt is preserved on clear
    assert len(session.messages()) == 1
    assert session.messages()[0].role == "system"


def test_session_metadata_immutable_proxy() -> None:
    session = Session(metadata={"tag": "dev"})
    proxy = session.metadata

    with pytest.raises(TypeError):
        proxy["new_tag"] = "prod"  # type: ignore[index]

    session.update_metadata({"new_tag": "prod"})
    assert session.metadata["new_tag"] == "prod"


def test_session_serialization_round_trip() -> None:
    session = Session(
        session_id="roundtrip-session",
        system_prompt="System prompt.",
        metadata={"priority": 1},
    )

    call = ToolCall(id="c1", name="calculator", arguments={"expr": "2+2"})
    session.add_message(ChatMessage(role="assistant", content="Calculating", tool_calls=(call,)))
    session.add_message(ChatMessage(role="tool", content="4", tool_call_id="c1"))

    data = session.to_dict()
    restored = Session.from_dict(data)

    assert restored.session_id == session.session_id
    assert restored.system_prompt == session.system_prompt
    assert restored.metadata == session.metadata
    assert len(restored.messages()) == 3

    tool_msg = restored.messages()[1]
    assert len(tool_msg.tool_calls) == 1
    assert tool_msg.tool_calls[0].name == "calculator"
    assert tool_msg.tool_calls[0].arguments == {"expr": "2+2"}


def test_session_store_crud() -> None:
    store = SessionStore()
    assert store.count() == 0

    s1 = store.create("session-1", system_prompt="System 1")
    assert store.count() == 1
    assert store.has("session-1")
    assert not store.has("session-2")

    retrieved = store.get("session-1")
    assert retrieved is s1

    # Duplicate creation rejected
    with pytest.raises(SessionError, match="already exists"):
        store.create("session-1")

    # Nonexistent get raises SessionNotFoundError
    with pytest.raises(SessionNotFoundError):
        store.get("nonexistent")

    # get_or_create
    s2 = store.get_or_create("session-2", system_prompt="System 2")
    assert s2.session_id == "session-2"
    assert store.count() == 2

    s2_again = store.get_or_create("session-2")
    assert s2_again is s2
    assert store.count() == 2

    # Listing
    sessions = store.list()
    assert len(sessions) == 2

    # Deletion
    store.delete("session-1")
    assert store.count() == 1
    assert not store.has("session-1")

    with pytest.raises(SessionNotFoundError):
        store.delete("session-1")


def test_session_isolation_from_each_other() -> None:
    store = SessionStore()
    s1 = store.create("user-session-a")
    s2 = store.create("user-session-b")

    s1.add_message(ChatMessage(role="user", content="Secret for session A"))
    s2.add_message(ChatMessage(role="user", content="Public message for session B"))

    assert len(s1.messages()) == 1
    assert len(s2.messages()) == 1
    assert s1.messages()[0].content == "Secret for session A"
    assert s2.messages()[0].content == "Public message for session B"

    s1.clear()
    assert len(s1.messages()) == 0
    assert len(s2.messages()) == 1
