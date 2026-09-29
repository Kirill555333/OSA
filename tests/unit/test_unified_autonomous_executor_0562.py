from __future__ import annotations

import pytest

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_executor import (
    UnifiedAutonomousExecutor,
)


class FakeResolver:
    def __init__(
        self,
        request=None,
        error: Exception | None = None,
    ) -> None:
        self.request = request
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def resolve(
        self,
        run_id: str,
        task_id: str,
    ):
        self.calls.append(
            (
                run_id,
                task_id,
            )
        )

        if self.error is not None:
            raise self.error

        return self.request


class FakeBridge:
    def __init__(
        self,
        result: AutonomousTaskResult | None = None,
    ) -> None:
        self.result = result
        self.calls: list[
            tuple[str, str, ActionRequest]
        ] = []

    def execute(
        self,
        run_id: str,
        task_id: str,
        request: ActionRequest,
    ) -> AutonomousTaskResult:
        self.calls.append(
            (
                run_id,
                task_id,
                request,
            )
        )

        return self.result or AutonomousTaskResult(
            task_id=task_id,
            status="completed",
            output="done",
        )


def action_request() -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="test_tool",
        arguments={
            "value": "hello",
        },
    )


def test_resolves_and_executes_through_bridge() -> None:
    request = action_request()
    expected = AutonomousTaskResult(
        task_id="task-1",
        status="completed",
        output="done",
    )

    resolver = FakeResolver(
        request=request
    )
    bridge = FakeBridge(
        result=expected
    )

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-1",
        "task-1",
    )

    assert result == expected
    assert resolver.calls == [
        (
            "run-1",
            "task-1",
        )
    ]
    assert bridge.calls == [
        (
            "run-1",
            "task-1",
            request,
        )
    ]


def test_resolver_failure_becomes_failed_result() -> None:
    resolver = FakeResolver(
        error=RuntimeError(
            "planner unavailable"
        )
    )
    bridge = FakeBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-1",
        "task-1",
    )

    assert result.task_id == "task-1"
    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_action_resolution_failed: "
            "planner unavailable"
        )
    )
    assert bridge.calls == []


def test_invalid_resolver_result_becomes_failed_result() -> None:
    resolver = FakeResolver(
        request={
            "kind": "tool",
            "name": "test_tool",
        }
    )
    bridge = FakeBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-1",
        "task-1",
    )

    assert result.task_id == "task-1"
    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_action_resolution_failed: "
            "resolver returned an invalid ActionRequest."
        )
    )
    assert bridge.calls == []


def test_failed_bridge_result_is_propagated() -> None:
    request = action_request()

    expected = AutonomousTaskResult(
        task_id="task-1",
        status="failed",
        error="permission denied",
    )

    resolver = FakeResolver(
        request=request
    )
    bridge = FakeBridge(
        result=expected
    )

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-1",
        "task-1",
    )

    assert result == expected


def test_run_id_is_normalized_before_resolution() -> None:
    request = action_request()
    resolver = FakeResolver(
        request=request
    )
    bridge = FakeBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    executor.execute(
        "  run-1  ",
        "  task-1  ",
    )

    assert resolver.calls == [
        (
            "run-1",
            "task-1",
        )
    ]

    assert bridge.calls[0][0:2] == (
        "run-1",
        "task-1",
    )


def test_empty_run_id_is_rejected() -> None:
    executor = UnifiedAutonomousExecutor(
        FakeResolver(),
        FakeBridge(),
    )

    with pytest.raises(
        ValueError,
        match="run_id cannot be empty",
    ):
        executor.execute(
            "   ",
            "task-1",
        )


def test_empty_task_id_is_rejected() -> None:
    executor = UnifiedAutonomousExecutor(
        FakeResolver(),
        FakeBridge(),
    )

    with pytest.raises(
        ValueError,
        match="task_id cannot be empty",
    ):
        executor.execute(
            "run-1",
            "   ",
        )


def test_non_string_run_id_is_rejected() -> None:
    executor = UnifiedAutonomousExecutor(
        FakeResolver(),
        FakeBridge(),
    )

    with pytest.raises(
        TypeError,
        match="run_id must be a string",
    ):
        executor.execute(
            123,
            "task-1",
        )


def test_non_string_task_id_is_rejected() -> None:
    executor = UnifiedAutonomousExecutor(
        FakeResolver(),
        FakeBridge(),
    )

    with pytest.raises(
        TypeError,
        match="task_id must be a string",
    ):
        executor.execute(
            "run-1",
            123,
        )


def test_executor_requires_resolver() -> None:
    with pytest.raises(
        ValueError,
        match="resolver is required",
    ):
        UnifiedAutonomousExecutor(
            None,
            FakeBridge(),
        )


def test_executor_requires_bridge() -> None:
    with pytest.raises(
        ValueError,
        match="bridge is required",
    ):
        UnifiedAutonomousExecutor(
            FakeResolver(),
            None,
        )
