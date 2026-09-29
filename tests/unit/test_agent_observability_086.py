from __future__ import annotations

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core.agent_observability import AgentObservability
from osa.recovery_contracts import RecoveryFailureKind, RecoveryResult


def make_request() -> ActionRequest:
    return ActionRequest(
        request_id="obs-1",
        kind=ActionKind.TOOL,
        name="calculator",
        arguments={
            "expression": "2 + 2",
            "secret": "must-not-be-logged",
        },
    )


def test_action_requested_logs_only_safe_correlation_data() -> None:
    events = []

    observer = AgentObservability(
        lambda event, **data: events.append((event, data))
    )

    observer.action_requested(
        make_request(),
        round_number=3,
    )

    event, data = events[0]

    assert event == "action.requested"
    assert data["request_id"] == "obs-1"
    assert data["action_kind"] == "tool"
    assert data["action_name"] == "calculator"
    assert data["status"] == "requested"
    assert data["round"] == 3

    assert "arguments" not in data
    assert "argument_keys" not in data
    assert "expression" not in str(data)
    assert "secret" not in str(data)
    assert "2 + 2" not in str(data)


def test_action_completed_logs_result_metadata() -> None:
    events = []

    observer = AgentObservability(
        lambda event, **data: events.append((event, data))
    )

    observer.action_completed(
        make_request(),
        ActionResult.succeeded(
            "obs-1",
            output="4",
        ),
        round_number=2,
    )

    event, data = events[0]

    assert event == "action.completed"
    assert data["request_id"] == "obs-1"
    assert data["action_kind"] == "tool"
    assert data["action_name"] == "calculator"
    assert data["status"] == "completed"
    assert data["success"] is True
    assert data["output_length"] == 1
    assert data["round"] == 2


def test_action_denied_logs_decision() -> None:
    events = []

    observer = AgentObservability(
        lambda event, **data: events.append((event, data))
    )

    observer.action_denied(
        make_request(),
        decision="denied",
        reason="permission denied",
        round_number=1,
    )

    event, data = events[0]

    assert event == "action.denied"
    assert data["request_id"] == "obs-1"
    assert data["action_kind"] == "tool"
    assert data["action_name"] == "calculator"
    assert data["status"] == "denied"
    assert data["decision"] == "denied"
    assert data["reason"] == "permission denied"
    assert data["round"] == 1


def test_recovery_events_preserve_correlation_and_failure_kind() -> None:
    events = []

    observer = AgentObservability(
        lambda event, **data: events.append((event, data))
    )

    observer.recovery_started(
        make_request(),
        max_attempts=3,
        round_number=4,
    )

    recovery = RecoveryResult.failed(
        failure_kind=RecoveryFailureKind.BACKEND_ERROR,
        error="backend failed",
    )

    observer.recovery_completed(
        make_request(),
        recovery,
        round_number=4,
    )

    started = events[0]
    completed = events[1]

    assert started[0] == "action.recovery.started"
    assert started[1]["request_id"] == "obs-1"
    assert started[1]["max_attempts"] == 3
    assert started[1]["round"] == 4

    assert completed[0] == "action.recovery.completed"
    assert completed[1]["request_id"] == "obs-1"
    assert completed[1]["status"] == "failed"
    assert completed[1]["failure_kind"] == "backend_error"
    assert completed[1]["round"] == 4
    assert completed[1]["error"] == "backend failed"


def test_observability_without_logger_is_noop() -> None:
    observer = AgentObservability(None)

    observer.action_requested(make_request())
    observer.action_denied(
        make_request(),
        decision="denied",
    )
