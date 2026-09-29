from __future__ import annotations

import pytest

from osa.core.agent_runtime import AgentRuntimeError
from osa.core.agent_stream import AgentStreamAccumulator
from osa.models import (
    ModelStreamEvent,
    ToolCall,
)


def test_accumulator_concatenates_stream_content() -> None:
    accumulator = AgentStreamAccumulator()

    accumulator.add(
        ModelStreamEvent(
            content="Привет, "
        )
    )
    accumulator.add(
        ModelStreamEvent(
            content="мир!"
        )
    )

    result = accumulator.result()

    assert accumulator.content == "Привет, мир!"
    assert result.content == "Привет, мир!"
    assert result.tool_calls == ()
    assert result.completed is True


def test_accumulator_preserves_multiple_tool_calls() -> None:
    accumulator = AgentStreamAccumulator()

    call_a = ToolCall(
        id="call_a",
        name="calculator",
        arguments={"expression": "1 + 1"},
    )
    call_b = ToolCall(
        id="call_b",
        name="calculator",
        arguments={"expression": "2 + 2"},
    )

    accumulator.add(
        ModelStreamEvent(
            tool_calls=(call_a,)
        )
    )
    accumulator.add(
        ModelStreamEvent(
            tool_calls=(call_b,)
        )
    )

    result = accumulator.result()

    assert result.tool_calls == (
        call_a,
        call_b,
    )
    assert result.completed is False


def test_accumulator_preserves_finish_reason() -> None:
    accumulator = AgentStreamAccumulator()

    accumulator.add(
        ModelStreamEvent(
            content="done",
            finish_reason="stop",
        )
    )

    result = accumulator.result()

    assert accumulator.finish_reason == "stop"
    assert result.metadata == {
        "finish_reason": "stop",
    }


def test_accumulator_uses_latest_finish_reason() -> None:
    accumulator = AgentStreamAccumulator()

    accumulator.add(
        ModelStreamEvent(
            finish_reason="tool_calls"
        )
    )
    accumulator.add(
        ModelStreamEvent(
            finish_reason="stop"
        )
    )

    assert accumulator.finish_reason == "stop"


def test_accumulator_rejects_invalid_event() -> None:
    accumulator = AgentStreamAccumulator()

    with pytest.raises(
        AgentRuntimeError,
        match="ModelStreamEvent",
    ):
        accumulator.add(
            object(),  # type: ignore[arg-type]
        )
