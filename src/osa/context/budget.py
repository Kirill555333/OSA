"""Context budget management and conservative token estimation for OSA."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping

from osa.models import ChatMessage, ToolDefinition


class ContextBudgetError(ValueError):
    """Raised when context budget configuration is invalid."""


class ContextOverflowError(RuntimeError):
    """Raised when content exceeds the allocated token budget."""


_WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]", re.UNICODE)


class ConservativeTokenEstimator:
    """Conservative token estimator for multilingual and tool-enabled prompts.

    Ensures prompt token counts are conservatively estimated so that the model
    context limit is never inadvertently exceeded on the inference backend.
    """

    MESSAGE_OVERHEAD_TOKENS = 4
    TOOL_CALL_OVERHEAD_TOKENS = 6
    TOOL_DEFINITION_OVERHEAD_TOKENS = 8

    def estimate_text(self, text: str) -> int:
        """Estimate token count for a text string conservatively."""
        if not text:
            return 0

        tokens = 0
        tokens_found = _WORD_PATTERN.findall(text)

        for token in tokens_found:
            cyrillic_chars = len(_CYRILLIC_PATTERN.findall(token))
            total_chars = len(token)
            latin_or_other_chars = total_chars - cyrillic_chars

            token_cost = 0.0

            if cyrillic_chars > 0:
                # Cyrillic BPE tokens typically encode 1.2 to 2 characters per token.
                # Conservatively estimate 1 token per 1.5 characters, minimum 1.
                token_cost += max(1.0, cyrillic_chars / 1.5)

            if latin_or_other_chars > 0:
                # Latin/code BPE tokens typically encode 3 to 4 characters per token.
                # Conservatively estimate 1 token per 3 characters, minimum 1.
                token_cost += max(1.0, latin_or_other_chars / 3.0)

            tokens += math.ceil(token_cost)

        # Baseline safety factor: ensure at least len(text) / 3.5 for whitespace/punctuations
        safety_floor = math.ceil(len(text) / 3.5)
        return max(tokens, safety_floor)

    def estimate_message(self, message: ChatMessage) -> int:
        """Estimate token count for one ChatMessage including metadata."""
        count = self.MESSAGE_OVERHEAD_TOKENS
        count += self.estimate_text(message.role)
        count += self.estimate_text(message.content)

        if message.tool_call_id:
            count += self.estimate_text(message.tool_call_id) + 2

        if message.tool_calls:
            for tool_call in message.tool_calls:
                count += self.TOOL_CALL_OVERHEAD_TOKENS
                count += self.estimate_text(tool_call.id)
                count += self.estimate_text(tool_call.name)
                serialized_args = json.dumps(
                    dict(tool_call.arguments),
                    ensure_ascii=False,
                )
                count += self.estimate_text(serialized_args)

        return count

    def estimate_messages(self, messages: tuple[ChatMessage, ...]) -> int:
        """Estimate token count for a collection of ChatMessages."""
        return sum(self.estimate_message(msg) for msg in messages)

    def estimate_tool_definition(self, tool: ToolDefinition) -> int:
        """Estimate token count for one ToolDefinition schema."""
        count = self.TOOL_DEFINITION_OVERHEAD_TOKENS
        count += self.estimate_text(tool.name)
        count += self.estimate_text(tool.description)
        serialized_params = json.dumps(
            dict(tool.parameters),
            ensure_ascii=False,
        )
        count += self.estimate_text(serialized_params)
        return count

    def estimate_tools(self, tools: tuple[ToolDefinition, ...]) -> int:
        """Estimate token count for a collection of ToolDefinitions."""
        return sum(self.estimate_tool_definition(tool) for tool in tools)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Immutable token budget allocation for model context management."""

    total_context_limit: int
    reserved_output_tokens: int
    system_budget: int
    memory_budget: int
    summary_budget: int
    task_budget: int
    history_budget: int
    tool_budget: int

    def __post_init__(self) -> None:
        if self.total_context_limit <= 0:
            raise ContextBudgetError("total_context_limit must be greater than zero.")

        if self.reserved_output_tokens <= 0:
            raise ContextBudgetError("reserved_output_tokens must be greater than zero.")

        if self.reserved_output_tokens >= self.total_context_limit:
            raise ContextBudgetError(
                "reserved_output_tokens must be strictly less than total_context_limit."
            )

        sub_budgets = {
            "system_budget": self.system_budget,
            "memory_budget": self.memory_budget,
            "summary_budget": self.summary_budget,
            "task_budget": self.task_budget,
            "history_budget": self.history_budget,
            "tool_budget": self.tool_budget,
        }

        for name, value in sub_budgets.items():
            if value < 0:
                raise ContextBudgetError(f"{name} cannot be negative.")

        if self.allocated_input_tokens > self.max_input_tokens:
            raise ContextBudgetError(
                f"Sum of allocated sub-budgets ({self.allocated_input_tokens}) "
                f"exceeds max_input_tokens ({self.max_input_tokens})."
            )

    @property
    def max_input_tokens(self) -> int:
        """Maximum tokens allowed for all input components combined."""
        return self.total_context_limit - self.reserved_output_tokens

    @property
    def allocated_input_tokens(self) -> int:
        """Total tokens explicitly partitioned among sub-budgets."""
        return (
            self.system_budget
            + self.memory_budget
            + self.summary_budget
            + self.task_budget
            + self.history_budget
            + self.tool_budget
        )

    @property
    def unallocated_headroom(self) -> int:
        """Unallocated tokens available as safety margin."""
        return self.max_input_tokens - self.allocated_input_tokens

    @classmethod
    def create(
        cls,
        total_context_limit: int = 13568,
        *,
        reserved_output_tokens: int = 512,
        system_budget: int | None = None,
        memory_budget: int | None = None,
        summary_budget: int | None = None,
        task_budget: int | None = None,
        history_budget: int | None = None,
        tool_budget: int | None = None,
    ) -> ContextBudget:
        """Factory creating a balanced ContextBudget with sensible defaults."""
        if total_context_limit <= 0:
            raise ContextBudgetError("total_context_limit must be greater than zero.")

        if reserved_output_tokens <= 0:
            raise ContextBudgetError("reserved_output_tokens must be greater than zero.")

        max_input = total_context_limit - reserved_output_tokens
        if max_input <= 0:
            raise ContextBudgetError(
                "total_context_limit must exceed reserved_output_tokens."
            )

        # Default distribution proportions for unspecified sections
        # system: ~8%, tool: ~10%, memory: ~8%, summary: ~12%, task: ~8%, history: remainder
        sys_b = (
            system_budget
            if system_budget is not None
            else max(64, int(max_input * 0.08))
        )
        tool_b = (
            tool_budget
            if tool_budget is not None
            else max(64, int(max_input * 0.10))
        )
        mem_b = (
            memory_budget
            if memory_budget is not None
            else max(64, int(max_input * 0.08))
        )
        sum_b = (
            summary_budget
            if summary_budget is not None
            else max(64, int(max_input * 0.12))
        )
        task_b = (
            task_budget
            if task_budget is not None
            else max(64, int(max_input * 0.08))
        )

        explicit_non_history = sys_b + tool_b + mem_b + sum_b + task_b

        if history_budget is not None:
            hist_b = history_budget
        else:
            hist_b = max(0, max_input - explicit_non_history)

        return cls(
            total_context_limit=total_context_limit,
            reserved_output_tokens=reserved_output_tokens,
            system_budget=sys_b,
            memory_budget=mem_b,
            summary_budget=sum_b,
            task_budget=task_b,
            history_budget=hist_b,
            tool_budget=tool_b,
        )

    def validate_input_tokens(self, estimated_tokens: int) -> None:
        """Verify that total estimated input tokens strictly respect max_input_tokens."""
        if estimated_tokens > self.max_input_tokens:
            raise ContextOverflowError(
                f"Estimated input tokens ({estimated_tokens}) exceed "
                f"max_input_tokens ({self.max_input_tokens}) for limit {self.total_context_limit}."
            )
