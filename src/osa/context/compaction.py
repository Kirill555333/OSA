"""Context compaction and sliding window management for OSA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from osa.context.budget import ConservativeTokenEstimator
from osa.models import ChatMessage


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Result of conversation context compaction."""

    retained_messages: tuple[ChatMessage, ...]
    evicted_messages: tuple[ChatMessage, ...]
    compacted_tool_count: int
    estimated_tokens: int


class ContextCompactor:
    """Compacts conversation history into a target token budget.

    Preserves atomic tool execution turns, compacts verbose historical tool outputs,
    and slides older messages out into an evicted partition for summarization.
    """

    def __init__(
        self,
        token_estimator: ConservativeTokenEstimator | None = None,
        *,
        min_recent_messages: int = 2,
        max_tool_content_tokens: int = 120,
        preserve_recent_tool_rounds: int = 1,
    ) -> None:
        if min_recent_messages < 1:
            raise ValueError("min_recent_messages must be at least 1.")

        if max_tool_content_tokens < 10:
            raise ValueError("max_tool_content_tokens must be at least 10.")

        if preserve_recent_tool_rounds < 0:
            raise ValueError("preserve_recent_tool_rounds cannot be negative.")

        self._estimator = token_estimator or ConservativeTokenEstimator()
        self._min_recent_messages = min_recent_messages
        self._max_tool_content_tokens = max_tool_content_tokens
        self._preserve_recent_tool_rounds = preserve_recent_tool_rounds

    def compact(
        self,
        messages: Sequence[ChatMessage],
        budget: int,
    ) -> CompactionResult:
        """Compact message sequence strictly within history token budget."""
        if budget <= 0:
            raise ValueError("budget must be greater than zero.")

        if not messages:
            return CompactionResult(
                retained_messages=(),
                evicted_messages=(),
                compacted_tool_count=0,
                estimated_tokens=0,
            )

        system_msg: ChatMessage | None = None
        dialogue_messages: list[ChatMessage] = []

        if messages[0].role == "system":
            system_msg = messages[0]
            dialogue_messages = list(messages[1:])
        else:
            dialogue_messages = list(messages)

        if not dialogue_messages:
            tokens = self._estimator.estimate_message(system_msg) if system_msg else 0
            return CompactionResult(
                retained_messages=tuple(messages),
                evicted_messages=(),
                compacted_tool_count=0,
                estimated_tokens=tokens,
            )

        system_tokens = self._estimator.estimate_message(system_msg) if system_msg else 0
        available_dialogue_budget = max(0, budget - system_tokens)

        # Step 1: Group dialogue messages into atomic units
        units = self._group_into_units(dialogue_messages)

        # Step 2: Phase 1 - Compact older verbose tool outputs
        units, compacted_tool_count = self._compact_tool_units(units)

        # Step 3: Phase 2 - Slide out older units if dialogue exceeds budget
        retained_units, evicted_units = self._slide_window(
            units,
            available_dialogue_budget,
        )

        retained_dialogue: list[ChatMessage] = []
        for unit in retained_units:
            retained_dialogue.extend(unit)

        evicted_messages: list[ChatMessage] = []
        for unit in evicted_units:
            evicted_messages.extend(unit)

        final_retained: list[ChatMessage] = []
        if system_msg is not None:
            final_retained.append(system_msg)
        final_retained.extend(retained_dialogue)

        total_tokens = self._estimator.estimate_messages(tuple(final_retained))

        return CompactionResult(
            retained_messages=tuple(final_retained),
            evicted_messages=tuple(evicted_messages),
            compacted_tool_count=compacted_tool_count,
            estimated_tokens=total_tokens,
        )

    def _group_into_units(
        self,
        messages: Sequence[ChatMessage],
    ) -> list[list[ChatMessage]]:
        """Group messages into atomic turns so tool calls and results are never decoupled."""
        units: list[list[ChatMessage]] = []
        i = 0
        n = len(messages)

        while i < n:
            msg = messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                unit = [msg]
                i += 1
                while i < n and messages[i].role == "tool":
                    unit.append(messages[i])
                    i += 1
                units.append(unit)
            else:
                units.append([msg])
                i += 1

        return units

    def _compact_tool_units(
        self,
        units: list[list[ChatMessage]],
    ) -> tuple[list[list[ChatMessage]], int]:
        """Compact oversized tool outputs in older tool rounds."""
        tool_unit_indices = [
            idx
            for idx, unit in enumerate(units)
            if any(m.role == "tool" for m in unit)
        ]

        if not tool_unit_indices:
            return units, 0

        # Protect the most recent tool rounds
        protected_count = self._preserve_recent_tool_rounds
        if protected_count > 0:
            compactable_indices = set(tool_unit_indices[:-protected_count])
        else:
            compactable_indices = set(tool_unit_indices)

        compacted_count = 0
        new_units: list[list[ChatMessage]] = []

        for idx, unit in enumerate(units):
            if idx not in compactable_indices:
                new_units.append(unit)
                continue

            new_unit: list[ChatMessage] = []
            for msg in unit:
                if msg.role == "tool":
                    compacted_msg, was_compacted = self._compact_single_tool_message(msg)
                    new_unit.append(compacted_msg)
                    if was_compacted:
                        compacted_count += 1
                else:
                    new_unit.append(msg)
            new_units.append(new_unit)

        return new_units, compacted_count

    def _compact_single_tool_message(
        self,
        message: ChatMessage,
    ) -> tuple[ChatMessage, bool]:
        """Truncate content of a tool message if it exceeds the tool token threshold."""
        content = message.content
        est_tokens = self._estimator.estimate_text(content)

        if est_tokens <= self._max_tool_content_tokens:
            return message, False

        # Approximate character limit based on token threshold
        char_limit = self._max_tool_content_tokens * 3
        truncated_content = (
            content[:char_limit].rstrip()
            + "\n...[output truncated for context budget]"
        )

        compacted_msg = ChatMessage(
            role=message.role,
            content=truncated_content,
            tool_call_id=message.tool_call_id,
            tool_calls=message.tool_calls,
        )
        return compacted_msg, True

    def _slide_window(
        self,
        units: list[list[ChatMessage]],
        available_budget: int,
    ) -> tuple[list[list[ChatMessage]], list[list[ChatMessage]]]:
        """Slide older units out until dialogue units fit into available budget."""
        if not units:
            return [], []

        unit_tokens = [
            self._estimator.estimate_messages(tuple(unit))
            for unit in units
        ]

        total_tokens = sum(unit_tokens)

        if total_tokens <= available_budget:
            return list(units), []

        evicted_units: list[list[ChatMessage]] = []
        retained_units = list(units)
        retained_tokens = unit_tokens[:]

        # Evict from oldest to newest
        while len(retained_units) > 1 and sum(retained_tokens) > available_budget:
            total_messages_retained = sum(len(u) for u in retained_units)
            if total_messages_retained <= self._min_recent_messages:
                break

            evicted_units.append(retained_units.pop(0))
            retained_tokens.pop(0)

        # If still over budget, keep evicting down to the last single unit if needed
        while len(retained_units) > 1 and sum(retained_tokens) > available_budget:
            evicted_units.append(retained_units.pop(0))
            retained_tokens.pop(0)

        return retained_units, evicted_units
