"""0.8.10.6 Autonomous action observability tests."""

from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.observability import InMemoryObservabilityLogger
from osa.observability.autonomous_action import (
    AutonomousActionObservabilityError,
    ObservableAutonomousActionBridge,
)
from osa.tasks.autonomous import AutonomousTaskResult


class FakeBridge:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls = []

    def execute(
        self,
        run_id: str,
        task_id: str,
        request: ActionRequest,
    ):
        self.calls.append(
            (
                run_id,
                task_id,
                request,
            )
        )

        if self.error is not None:
            raise self.error

        return self.result


def _request() -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="secret_tool",
        arguments={
            "password": "TOP_SECRET_ACTION_ARGUMENT",
        },
        request_id="autonomous-observability-010",
    )


def test_completed_action_emits_canonical_correlation() -> None:
    logger = InMemoryObservabilityLogger()

    bridge = FakeBridge(
        result=AutonomousTaskResult(
            task_id="task-010",
            status="completed",
            output="TOP_SECRET_ACTION_OUTPUT",
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=logger,
    )

    result = observed.execute(
        "run-010",
        "task-010",
        _request(),
    )

    assert result.status == "completed"
    assert bridge.calls[0][0] == "run-010"
    assert bridge.calls[0][1] == "task-010"

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.action.started",
        "autonomous.action.completed",
    ]

    for event in events:
        assert event.source == "autonomous"
        assert event.request_id == (
            "autonomous-observability-010"
        )
        assert event.run_id == "run-010"
        assert event.task_id == "task-010"
        assert event.action_kind == "tool"
        assert event.action_name == "secret_tool"

    assert (
        events[0].metadata["arguments_present"]
        is True
    )
    assert (
        events[-1].metadata["output_present"]
        is True
    )


def test_arguments_and_output_are_never_logged() -> None:
    logger = InMemoryObservabilityLogger()

    argument_secret = "TOP_SECRET_ACTION_ARGUMENT"
    output_secret = "TOP_SECRET_ACTION_OUTPUT"

    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="secret_tool",
        arguments={
            "password": argument_secret,
        },
        request_id="autonomous-redaction-010",
    )

    bridge = FakeBridge(
        result=AutonomousTaskResult(
            task_id="task-redaction",
            status="completed",
            output=output_secret,
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=logger,
    )

    result = observed.execute(
        "run-redaction",
        "task-redaction",
        request,
    )

    assert result.output == output_secret

    for event in logger.events():
        serialized = str(event.to_dict())
        assert argument_secret not in serialized
        assert output_secret not in serialized
        assert "password" not in serialized


def test_failed_action_preserves_identity_and_error() -> None:
    logger = InMemoryObservabilityLogger()

    request = _request()

    bridge = FakeBridge(
        result=AutonomousTaskResult(
            task_id="task-failed",
            status="failed",
            error="execution failed",
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=logger,
    )

    result = observed.execute(
        "run-failed",
        "task-failed",
        request,
    )

    assert result.status == "failed"
    assert result.task_id == "task-failed"
    assert result.error == "execution failed"

    event = logger.events()[-1]

    assert event.event == "autonomous.action.failed"
    assert event.request_id == request.request_id
    assert event.run_id == "run-failed"
    assert event.task_id == "task-failed"
    assert event.error == "execution failed"


def test_bridge_exception_is_logged_and_reraised() -> None:
    logger = InMemoryObservabilityLogger()

    bridge = FakeBridge(
        error=RuntimeError(
            "bridge exploded"
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=logger,
    )

    with pytest.raises(
        RuntimeError,
        match="bridge exploded",
    ):
        observed.execute(
            "run-exception",
            "task-exception",
            _request(),
        )

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.action.started",
        "autonomous.action.failed",
    ]

    assert events[-1].error == "bridge exploded"


def test_invalid_task_status_is_failed_closed() -> None:
    logger = InMemoryObservabilityLogger()

    bridge = FakeBridge(
        result=AutonomousTaskResult(
            task_id="task-invalid",
            status="pending",
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=logger,
    )

    result = observed.execute(
        "run-invalid",
        "task-invalid",
        _request(),
    )

    assert result.status == "failed"
    assert result.error == (
        "wrapped autonomous bridge returned "
        "unsupported task status 'pending'"
    )

    assert logger.events()[-1].event == (
        "autonomous.action.failed"
    )


def test_logging_failure_does_not_break_execution() -> None:
    class BrokenLogger:
        def log(self, event) -> None:
            raise RuntimeError("logger exploded")

    bridge = FakeBridge(
        result=AutonomousTaskResult(
            task_id="task-safe",
            status="completed",
            output="done",
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=BrokenLogger(),
    )

    result = observed.execute(
        "run-safe",
        "task-safe",
        _request(),
    )

    assert result.status == "completed"


def test_none_bridge_is_rejected() -> None:
    with pytest.raises(
        AutonomousActionObservabilityError,
        match="bridge is required",
    ):
        ObservableAutonomousActionBridge(
            None
        )


def test_logger_identity_is_preserved() -> None:
    logger = InMemoryObservabilityLogger()

    bridge = FakeBridge(
        result=AutonomousTaskResult(
            task_id="task-identity",
            status="completed",
        )
    )

    observed = ObservableAutonomousActionBridge(
        bridge,
        logger=logger,
    )

    assert observed.bridge is bridge
    assert observed.logger is logger
