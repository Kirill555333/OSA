from __future__ import annotations

import pytest

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.core.agent import (
    Agent,
    AgentError,
)
from osa.core.agent_recovery import (
    AgentRecoveryIntegration,
    AgentRecoveryIntegrationError,
)
from osa.recovery_contracts import (
    RecoveryResult,
)


class FakeModel:
    pass


class FakeRecoveryExecutor:
    def __init__(
        self,
        result: RecoveryResult | None = None,
        *,
        exception: Exception | None = None,
    ) -> None:
        self.result = (
            result
            if result is not None
            else RecoveryResult.succeeded(
                result="ok"
            )
        )
        self.exception = exception
        self.calls: list[
            tuple[
                ActionRequest,
                str | None,
                str | None,
            ]
        ] = []

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls.append(
            (
                request,
                run_id,
                task_id,
            )
        )

        if self.exception is not None:
            raise self.exception

        return self.result


class InvalidResultExecutor:
    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ):
        return object()


def _request() -> ActionRequest:
    return ActionRequest(
        request_id="req-1",
        kind=ActionKind.TOOL,
        name="demo_tool",
        arguments={
            "value": 42,
        },
    )


def test_integration_preserves_executor_identity():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    assert integration.executor is executor


def test_execute_delegates_original_request_and_correlation():
    executor = FakeRecoveryExecutor(
        RecoveryResult.succeeded(
            result={
                "ok": True,
            }
        )
    )

    integration = AgentRecoveryIntegration(
        executor
    )

    request = _request()

    result = integration.execute(
        request,
        run_id="run-1",
        task_id="task-1",
    )

    assert result.success is True
    assert len(executor.calls) == 1

    delegated_request, run_id, task_id = (
        executor.calls[0]
    )

    assert delegated_request is request
    assert run_id == "run-1"
    assert task_id == "task-1"


def test_execute_rejects_invalid_request_before_executor():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    with pytest.raises(
        TypeError,
        match="ActionRequest",
    ):
        integration.execute(
            object()
        )

    assert executor.calls == []


def test_execute_rejects_invalid_recovery_result():
    integration = AgentRecoveryIntegration(
        InvalidResultExecutor()
    )

    with pytest.raises(
        AgentRecoveryIntegrationError,
        match="invalid result",
    ):
        integration.execute(
            _request()
        )


def test_execute_propagates_executor_exception():
    integration = AgentRecoveryIntegration(
        FakeRecoveryExecutor(
            exception=RuntimeError(
                "recovery failure"
            )
        )
    )

    with pytest.raises(
        RuntimeError,
        match="recovery failure",
    ):
        integration.execute(
            _request()
        )


def test_execute_tool_builds_tool_action_request():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    result = integration.execute_tool(
        "demo_tool",
        {
            "value": 42,
        },
        request_id="req-tool-1",
        run_id="run-2",
        task_id="task-2",
    )

    assert result.success is True
    assert len(executor.calls) == 1

    request, run_id, task_id = (
        executor.calls[0]
    )

    assert request.request_id == (
        "req-tool-1"
    )
    assert request.kind == ActionKind.TOOL
    assert request.name == "demo_tool"
    assert dict(request.arguments) == {
        "value": 42,
    }
    assert run_id == "run-2"
    assert task_id == "task-2"


def test_execute_tool_generates_request_id_when_missing():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    integration.execute_tool(
        "demo_tool",
        {},
    )

    request = executor.calls[0][0]

    assert isinstance(
        request.request_id,
        str,
    )
    assert request.request_id


def test_execute_tool_rejects_empty_name():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    with pytest.raises(
        ValueError,
        match="tool_name",
    ):
        integration.execute_tool(
            "   ",
            {},
        )

    assert executor.calls == []


def test_execute_tool_rejects_invalid_arguments():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    with pytest.raises(
        TypeError,
        match="arguments",
    ):
        integration.execute_tool(
            "demo_tool",
            [],
        )

    assert executor.calls == []


def test_execute_tool_rejects_empty_request_id():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    with pytest.raises(
        ValueError,
        match="request_id",
    ):
        integration.execute_tool(
            "demo_tool",
            {},
            request_id="   ",
        )

    assert executor.calls == []


def test_agent_exposes_configured_recovery_integration():
    executor = FakeRecoveryExecutor()

    integration = AgentRecoveryIntegration(
        executor
    )

    agent = Agent(
        FakeModel(),
        recovery_integration=integration,
    )

    assert (
        agent.recovery_integration
        is integration
    )


def test_agent_delegates_action_to_recovery_integration():
    executor = FakeRecoveryExecutor(
        RecoveryResult.succeeded(
            result="recovered"
        )
    )

    integration = AgentRecoveryIntegration(
        executor
    )

    agent = Agent(
        FakeModel(),
        recovery_integration=integration,
    )

    result = agent.execute_action_with_recovery(
        _request(),
        run_id="run-9",
        task_id="task-9",
    )

    assert result.success is True
    assert result.result == "recovered"

    request, run_id, task_id = (
        executor.calls[0]
    )

    assert request.request_id == "req-1"
    assert run_id == "run-9"
    assert task_id == "task-9"


def test_agent_recovery_is_explicitly_opt_in():
    agent = Agent(
        FakeModel()
    )

    assert agent.recovery_integration is None

    with pytest.raises(
        AgentError,
        match="not configured",
    ):
        agent.execute_action_with_recovery(
            _request()
        )


def test_agent_wraps_integration_contract_error():
    integration = AgentRecoveryIntegration(
        InvalidResultExecutor()
    )

    agent = Agent(
        FakeModel(),
        recovery_integration=integration,
    )

    with pytest.raises(
        AgentError,
        match="invalid result",
    ):
        agent.execute_action_with_recovery(
            _request()
        )
