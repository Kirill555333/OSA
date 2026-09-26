"""Conversation context management for OSA."""

from __future__ import annotations

from osa.models import ChatMessage


class ConversationContext:
    """Store the messages belonging to one OSA conversation."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._messages: list[ChatMessage] = []

        if system_prompt:
            self._messages.append(
                ChatMessage(
                    role="system",
                    content=system_prompt,
                )
            )

    def add(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        self._messages.append(message)

    def remove_last(self) -> ChatMessage:
        """Remove and return the most recent message."""
        if not self._messages:
            raise IndexError("Conversation context is empty.")

        return self._messages.pop()

    def messages(self) -> tuple[ChatMessage, ...]:
        """Return an immutable snapshot of the conversation."""
        return tuple(self._messages)

    def clear(self) -> None:
        """Clear the conversation."""
        self._messages.clear()

    def __len__(self) -> int:
        """Return the number of stored messages."""
        return len(self._messages)
