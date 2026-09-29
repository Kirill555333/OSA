from __future__ import annotations

import pytest

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.tasks.action_bridge import (
    AutonomousActionBridge,
    AutonomousActionBridgeError,
    CallbackAutonomousActionResolver,
)


class FakePipeline:
    def __init__(
        self,
        result=None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.requests: list[ActionRequest] = []

    def execute(
        self,
        request: ActionRequest,
    ):
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return self.result


def action_request() -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="test_tool",
        arguments={
            "value": "hello",
        },
    )


def test_success_is_converted_to_completed_task_result() -> None:
    request = action_request()

    pipeline = FakePipeline(
        result=ActionResult.succeeded(
            request.request_id,
            output="done",
        )
    )

    bridge = AutonomousActionBridge(
        pipeline,
    )

    result = bridge.execute(
        "run-1",
        "task-1",
        request,
    )

    assert result.task_id == "task-1"
    assert result.status == "completed"
    assert result.output == "done"
    assert result.error is None
    assert pipeline.requests == [request]


def test_failed_action_is_converted_to_failed_task_result() -> None:
    request = action_request()

    pipeline = FakePipeline(
        result=ActionResult.failed(
            request.request_id,
            error="permission denied",
        )
    )

    bridge = AutonomousActionBridge(
        pipeline,
    )

    result = bridge.execute(
        "run-1",
        "task-1",
        request,
    )

    assert result.task_id == "task-1"
    assert result.status == "failed"
    assert result.error == "permission denied"


def test_pipeline_exception_is_converted_to_failed_task_result() -> None:
    request = action_request()

    pipeline = FakePipeline(
        error=RuntimeError(
            "backend offline"
        )
    )

    bridge = AutonomousActionBridge(
        pipeline,
    )

    result = bridge.execute(
        "run-1",
        "task-1",
        request,
    )

    assert result.task_id == "task-1"
    assert result.status == "failed"
    assert (
        result.error
        == "autonomous_action_execution_failed: backend offline"
    )


def test_invalid_pipeline_result_is_failed() -> None:
    request = action_request()

    pipeline = FakePipeline(
        result={
            "success": True,
        }
    )

    bridge = AutonomousActionBridge(
        pipeline,
    )

    result = bridge.execute(
        "run-1",
        "task-1",
        request,
    )

    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_action_invalid_result: "
            "pipeline returned an invalid result type."
        )
    )


def test_request_id_mismatch_is_failed() -> None:
    request = action_request()

    pipeline = FakePipeline(
        result=ActionResult.succeeded(
            "different-request-id",
            output="done",
        )
    )

    bridge = AutonomousActionBridge(
        pipeline,
    )

    result = bridge.execute(
        "run-1",
        "task-1",
        request,
    )

    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_action_invalid_result: "
            "request_id mismatch."
        )
    )


def test_empty_run_id_is_rejected() -> None:
    bridge = AutonomousActionBridge(
        FakePipeline()
    )

    with pytest.raises(
        ValueError,
        match="run_id cannot be empty",
    ):
        bridge.execute(
            "   ",
            "task-1",
            action_request(),
        )


def test_empty_task_id_is_rejected() -> None:
    bridge = AutonomousActionBridge(
        FakePipeline()
    )

    with pytest.raises(
        ValueError,
        match="task_id cannot be empty",
    ):
        bridge.execute(
            "run-1",
            "   ",
            action_request(),
        )


def test_invalid_request_type_is_rejected() -> None:
    bridge = AutonomousActionBridge(
        FakePipeline()
    )

    with pytest.raises(
        AutonomousActionBridgeError,
        match="request must be an ActionRequest",
    ):
        bridge.execute(
            "run-1",
            "task-1",
            {"name": "test_tool"},
        )


def test_callback_resolver_delegates_exact_arguments() -> None:
    calls = []

    def callback(
        run_id: str,
        task_id: str,
    ) -> ActionRequest:
        calls.append(
            (
                run_id,
                task_id,
            )
        )
        return action_request()

    resolver = CallbackAutonomousActionResolver(
        callback
    )

    request = resolver.resolve(
        "run-7",
        "task-9",
    )

    assert isinstance(
        request,
        ActionRequest,
    )
    assert calls == [
        (
            "run-7",
            "task-9",
        )
    ]


def test_callback_resolver_rejects_non_callable() -> None:
    with pytest.raises(
        TypeError,
        match="callback must be callable",
    ):
        CallbackAutonomousActionResolver(
            None
        )


def test_bridge_requires_pipeline() -> None:
    with pytest.raises(
        ValueError,
        match="pipeline is required",
    ):
        AutonomousActionBridge(
            None
        )
