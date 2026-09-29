from __future__ import annotations

import pytest

from osa.core.agent_runtime import (
    AgentRoundExecutor,
    AgentRoundResult,
    AgentRoundRuntime,
    AgentRuntimeError,
)
from osa.models import (
    ModelResponse,
    ToolCall,
)
from osa.tools import ToolResult


def test_from_model_response_normalizes_completed_round() -> None:
    response = ModelResponse(
        content="Готово.",
        model_name="fake",
    )

    result = AgentRoundRuntime.from_model_response(
        response
    )

    assert result.content == "Готово."
    assert result.tool_calls == ()
    assert result.model_response is response
    assert result.completed is True
    assert result.has_tool_calls is False


def test_from_model_response_preserves_multiple_tool_calls() -> None:
    calls = (
        ToolCall(
            id="call_1",
            name="calculator",
            arguments={"expression": "2 + 2"},
        ),
        ToolCall(
            id="call_2",
            name="calculator",
            arguments={"expression": "3 * 3"},
        ),
    )

    response = ModelResponse(
        content="",
        model_name="fake",
        tool_calls=calls,
    )

    result = AgentRoundRuntime.from_model_response(
        response
    )

    assert result.content == ""
    assert result.tool_calls == calls
    assert result.model_response is response
    assert result.completed is False
    assert result.has_tool_calls is True


def test_from_stream_event_normalizes_accumulated_content() -> None:
    result = AgentRoundRuntime.from_stream_event(
        "part onepart two",
    )

    assert result.content == "part onepart two"
    assert result.tool_calls == ()
    assert result.metadata is None
    assert result.completed is True


def test_from_stream_event_preserves_tool_calls_and_metadata() -> None:
    calls = (
        ToolCall(
            id="stream_1",
            name="calculator",
            arguments={"expression": "5 + 7"},
        ),
    )

    result = AgentRoundRuntime.from_stream_event(
        "partial",
        calls,
        metadata={
            "finish_reason": "tool_calls",
        },
    )

    assert result.content == "partial"
    assert result.tool_calls == calls
    assert result.metadata == {
        "finish_reason": "tool_calls",
    }
    assert result.has_tool_calls is True
    assert result.completed is False


def test_round_result_rejects_invalid_content() -> None:
    with pytest.raises(
        AgentRuntimeError,
        match="content",
    ):
        AgentRoundResult(
            content=None,  # type: ignore[arg-type]
        )


def test_round_result_rejects_invalid_tool_calls_container() -> None:
    with pytest.raises(
        AgentRuntimeError,
        match="tool_calls",
    ):
        AgentRoundResult(
            content="content",
            tool_calls=[  # type: ignore[arg-type]
                ToolCall(
                    id="call_1",
                    name="demo",
                    arguments={},
                )
            ],
        )


def test_round_result_rejects_invalid_tool_call_value() -> None:
    with pytest.raises(
        AgentRuntimeError,
        match="ToolCall",
    ):
        AgentRoundResult(
            content="content",
            tool_calls=(
                object(),  # type: ignore[arg-type]
            ),
        )


def test_round_result_rejects_invalid_model_response() -> None:
    with pytest.raises(
        AgentRuntimeError,
        match="model_response",
    ):
        AgentRoundResult(
            content="content",
            model_response=object(),  # type: ignore[arg-type]
        )


def test_round_result_rejects_invalid_metadata() -> None:
    with pytest.raises(
        AgentRuntimeError,
        match="metadata",
    ):
        AgentRoundResult(
            content="content",
            metadata=[],  # type: ignore[arg-type]
        )


def test_round_executor_preserves_tool_call_order() -> None:
    calls = (
        ToolCall(
            id="call_1",
            name="first",
            arguments={},
        ),
        ToolCall(
            id="call_2",
            name="second",
            arguments={},
        ),
    )

    executed: list[str] = []

    def execute(tool_call: ToolCall) -> ToolResult:
        executed.append(tool_call.id)

        return ToolResult(
            success=True,
            output=tool_call.id,
        )

    executor = AgentRoundExecutor(execute)
    results = executor.execute(calls)

    assert executed == [
        "call_1",
        "call_2",
    ]
    assert tuple(
        item.tool_call.id
        for item in results
    ) == (
        "call_1",
        "call_2",
    )
    assert tuple(
        item.result.output
        for item in results
        if item.result is not None
    ) == (
        "call_1",
        "call_2",
    )
    assert all(
        item.error is None
        for item in results
    )


def test_round_executor_captures_one_error_and_continues() -> None:
    calls = (
        ToolCall(
            id="call_1",
            name="first",
            arguments={},
        ),
        ToolCall(
            id="call_2",
            name="second",
            arguments={},
        ),
    )

    second_result = ToolResult(
        success=True,
        output="second-output",
    )

    def execute(tool_call: ToolCall) -> ToolResult:
        if tool_call.id == "call_1":
            raise PermissionError("denied")

        return second_result

    executor = AgentRoundExecutor(execute)
    results = executor.execute(calls)

    assert len(results) == 2
    assert isinstance(
        results[0].error,
        PermissionError,
    )
    assert results[0].result is None

    assert results[1].error is None
    assert results[1].result is second_result


def test_round_executor_rejects_invalid_callback() -> None:
    with pytest.raises(
        AgentRuntimeError,
        match="execute_tool_call",
    ):
        AgentRoundExecutor(
            None,  # type: ignore[arg-type]
        )


def test_round_executor_rejects_invalid_tool_result() -> None:
    tool_call = ToolCall(
        id="call_invalid",
        name="demo",
        arguments={},
    )

    def execute(_: ToolCall):
        return "invalid"

    executor = AgentRoundExecutor(execute)

    with pytest.raises(
        AgentRuntimeError,
        match="ToolResult",
    ):
        executor.execute(
            (tool_call,)
        )


def test_round_executor_rejects_non_iterable_tool_calls() -> None:
    executor = AgentRoundExecutor(
        lambda _: ToolResult(
            success=True,
            output="ok",
        )
    )

    with pytest.raises(
        AgentRuntimeError,
        match="iterable",
    ):
        executor.execute(
            None,  # type: ignore[arg-type]
        )
