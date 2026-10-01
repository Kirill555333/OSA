"""ContextManager orchestrating budget, compaction, memory, and summary for OSA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from osa.context.budget import (
    ConservativeTokenEstimator,
    ContextBudget,
    ContextOverflowError,
)
from osa.context.compaction import ContextCompactor
from osa.context.memory import BudgetedMemoryResult, BudgetedMemoryRetriever
from osa.context.summary import ConversationSummary, SummaryBuilder
from osa.models import ChatMessage, ToolDefinition


class ContextManagerError(RuntimeError):
    """Raised when context assembly fails."""


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The fully assembled, bounded context ready for model invocation."""

    messages: tuple[ChatMessage, ...]
    summary: ConversationSummary
    memory_result: BudgetedMemoryResult | None
    estimated_input_tokens: int
    compacted_tool_count: int
    evicted_messages_count: int


class ContextManager:
    """Orchestrates bounded working context assembly for OSA language models."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        compactor: ContextCompactor | None = None,
        summary_builder: SummaryBuilder | None = None,
        memory_retriever: BudgetedMemoryRetriever | None = None,
        token_estimator: ConservativeTokenEstimator | None = None,
        default_system_prompt: str | None = None,
    ) -> None:
        self._budget = budget or ContextBudget.create()
        self._estimator = token_estimator or ConservativeTokenEstimator()
        self._compactor = compactor or ContextCompactor(token_estimator=self._estimator)
        self._summary_builder = summary_builder or SummaryBuilder()
        self._memory_retriever = memory_retriever
        self._default_system_prompt = default_system_prompt.strip() if default_system_prompt else None

        self._current_summary: ConversationSummary = ConversationSummary.empty()

    @property
    def budget(self) -> ContextBudget:
        """Return the active ContextBudget configuration."""
        return self._budget

    @property
    def current_summary(self) -> ConversationSummary:
        """Return the current conversation summary."""
        return self._current_summary

    def set_summary(self, summary: ConversationSummary) -> None:
        """Restore or set active conversation summary."""
        self._current_summary = summary

    def reset(self) -> None:
        """Reset conversation summary and transient state."""
        self._current_summary = ConversationSummary.empty()

    def assemble(
        self,
        session_messages: Sequence[ChatMessage],
        *,
        query: str | None = None,
        task_context: str | None = None,
        tools: Sequence[ToolDefinition] = (),
        override_memory_context: str | None = None,
    ) -> AssembledContext:
        """Assemble a bounded, structured tuple of ChatMessages for the model."""
        # 1. Base system prompt extraction
        system_content = self._default_system_prompt
        dialogue_messages: list[ChatMessage] = []

        for idx, msg in enumerate(session_messages):
            if idx == 0 and msg.role == "system":
                system_content = msg.content
            else:
                dialogue_messages.append(msg)

        # 2. Compact dialogue messages into history_budget
        compaction = self._compactor.compact(
            dialogue_messages,
            budget=self._budget.history_budget,
        )

        # 3. If messages were evicted, update the conversation summary
        if compaction.evicted_messages:
            self._current_summary = self._summary_builder.build_summary(
                compaction.evicted_messages,
                previous_summary=self._current_summary,
            )

        # 4. Long-term memory retrieval within memory_budget
        mem_result: BudgetedMemoryResult | None = None
        memory_prompt_text: str | None = None

        if override_memory_context:
            memory_prompt_text = override_memory_context.strip()
        elif query and self._memory_retriever is not None:
            mem_result = self._memory_retriever.retrieve(
                query,
                token_budget=self._budget.memory_budget,
            )
            memory_prompt_text = mem_result.prompt

        # 5. Build assembled prompt blocks
        assembled: list[ChatMessage] = []

        if system_content:
            assembled.append(ChatMessage(role="system", content=system_content))

        # Structured summary block
        if not self._current_summary.is_empty:
            summary_text = self._current_summary.format_for_prompt()
            if summary_text:
                assembled.append(ChatMessage(role="system", content=summary_text))

        # Memory context block
        if memory_prompt_text:
            assembled.append(ChatMessage(role="system", content=memory_prompt_text))

        # Task context block
        if task_context and task_context.strip():
            task_msg = f"[Current Task State]\n{task_context.strip()}"
            assembled.append(ChatMessage(role="system", content=task_msg))

        # Retained dialogue messages
        assembled.extend(compaction.retained_messages)

        final_tuple = tuple(assembled)

        # 6. Estimate tokens and enforce strict budget invariant
        total_msg_tokens = self._estimator.estimate_messages(final_tuple)
        tool_tokens = self._estimator.estimate_tools(tuple(tools)) if tools else 0
        total_input_tokens = total_msg_tokens + tool_tokens

        try:
            self._budget.validate_input_tokens(total_input_tokens)
        except ContextOverflowError as exc:
            raise ContextManagerError(
                f"Assembled context exceeds budget: {exc}"
            ) from exc

        return AssembledContext(
            messages=final_tuple,
            summary=self._current_summary,
            memory_result=mem_result,
            estimated_input_tokens=total_input_tokens,
            compacted_tool_count=compaction.compacted_tool_count,
            evicted_messages_count=len(compaction.evicted_messages),
        )
