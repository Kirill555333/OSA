"""0.8.9.6 Text/Streaming round execution-context parity tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from osa.actions.contracts import ActionKind, ActionRequest
from osa.core import Agent
from osa.core.execution_context import ExecutionContext
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.actions.contracts import ActionResult
from osa.tools import ToolRegistry


class RecordingRecoveryExecutor:
    """Capture action requests reaching the unified core."""

    max_attempts = 1

    def __init__(self) -> None:
        self.requests: list[ActionRequest] = []

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.requests.append(request)

        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="42",
            ),
        )


class TextRoundModel(ModelInterface):
    """Text model that emits one tool call on round one."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "text-round-context-089"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="text-round-089",
                        name="demo",
                        arguments={
                            "value": 21,
                        },
                    ),
                ),
            )

        return ModelResponse(
            content="42",
            model_name=self.model_name,
            finish_reason="stop",
        )

    def health_check(self) -> bool:
        return True


class StreamingRoundModel(ModelInterface):
    """Streaming model that emits one tool call on round one."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stream-round-context-089"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Streaming path must not call generate()."
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        self.calls += 1

        if self.calls == 1:
            yield ModelStreamEvent(
                tool_calls=(
                    ToolCall(
                        id="stream-round-089",
                        name="demo",
                        arguments={
                            "value": 21,
                        },
                    ),
                )
            )
            return

        yield ModelStreamEvent(
            content="42"
        )

    def health_check(self) -> bool:
        return True


def _agent(
    model: ModelInterface,
    executor: RecordingRecoveryExecutor,
) -> Agent:
    registry = ToolRegistry()

    class DemoTool:
        name = "demo"
        description = "Round context test tool"
        parameters = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        }

        def execute(
            self,
            arguments: dict[str, Any],
        ):
            raise AssertionError(
                "Direct tool execution must not occur in this recovery test."
            )

    registry.register(
        DemoTool()
    )

    return Agent(
        model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
    )


def test_text_tool_call_carries_round_one_context() -> None:
    executor = RecordingRecoveryExecutor()

    agent = _agent(
        TextRoundModel(),
        executor,
    )

    response = agent.chat(
        "run round context"
    )

    assert response.content == "42"
    assert len(executor.requests) == 1

    request = executor.requests[0]

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.request_id == "text-round-089"
    assert context.round_number == 1
    assert context.round == 1
    assert context.source == "agent"
    assert context.run_id is None
    assert context.task_id is None


def test_streaming_tool_call_carries_round_one_context() -> None:
    executor = RecordingRecoveryExecutor()

    agent = _agent(
        StreamingRoundModel(),
        executor,
    )

    chunks = list(
        agent.chat_stream(
            "run round context"
        )
    )

    assert chunks == ["42"]
    assert len(executor.requests) == 1

    request = executor.requests[0]

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.request_id == "stream-round-089"
    assert context.round_number == 1
    assert context.round == 1
    assert context.source == "agent"
    assert context.run_id is None
    assert context.task_id is None


def test_text_and_streaming_have_identical_round_context_semantics() -> None:
    text_executor = RecordingRecoveryExecutor()
    text_agent = _agent(
        TextRoundModel(),
        text_executor,
    )

    text_agent.chat(
        "run round context"
    )

    stream_executor = RecordingRecoveryExecutor()
    stream_agent = _agent(
        StreamingRoundModel(),
        stream_executor,
    )

    list(
        stream_agent.chat_stream(
            "run round context"
        )
    )

    text_context = ExecutionContext.from_action_request(
        text_executor.requests[0]
    )

    stream_context = ExecutionContext.from_action_request(
        stream_executor.requests[0]
    )

    assert text_context.round_number == (
        stream_context.round_number
    )
    assert text_context.round_number == 1

    assert text_context.source == (
        stream_context.source
    )
    assert text_context.source == "agent"

    assert text_context.run_id is None
    assert stream_context.run_id is None

    assert text_context.task_id is None
    assert stream_context.task_id is None
