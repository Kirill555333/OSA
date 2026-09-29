from __future__ import annotations

from datetime import datetime, timezone

import pytest

from osa.actions.contracts import ActionKind
from osa.observability import (
    ObservabilityEvent,
    ObservabilityEventError,
)


def test_event_generates_id_and_utc_timestamp() -> None:
    event = ObservabilityEvent(
        event="action.started",
        source="action_pipeline",
    )

    assert event.event_id
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() == timezone.utc.utcoffset(
        event.timestamp
    )


def test_event_normalizes_core_string_fields() -> None:
    event = ObservabilityEvent(
        event="  action.started  ",
        source="  pipeline  ",
        request_id="  request-1  ",
        run_id="  run-1  ",
        task_id="  task-1  ",
        action_kind="  tool  ",
        action_name="  integration_tool  ",
        status="  started  ",
        decision="  allow  ",
        error="  failure  ",
    )

    assert event.event == "action.started"
    assert event.source == "pipeline"
    assert event.request_id == "request-1"
    assert event.run_id == "run-1"
    assert event.task_id == "task-1"
    assert event.action_kind == "tool"
    assert event.action_name == "integration_tool"
    assert event.status == "started"
    assert event.decision == "allow"
    assert event.error == "failure"


def test_action_kind_enum_is_normalized_to_value() -> None:
    event = ObservabilityEvent(
        event="action.started",
        source="action_pipeline",
        action_kind=ActionKind.BROWSER,
    )

    assert event.action_kind == ActionKind.BROWSER.value


def test_event_accepts_execution_metadata() -> None:
    timestamp = datetime(
        2026,
        9,
        29,
        10,
        30,
        tzinfo=timezone.utc,
    )

    event = ObservabilityEvent(
        event="action.completed",
        source="router",
        timestamp=timestamp,
        request_id="request-1",
        run_id="run-1",
        task_id="task-1",
        action_kind=ActionKind.DESKTOP,
        action_name="desktop_click_point",
        status="completed",
        duration_ms=125.5,
        decision="allow",
        attempt=1,
        max_attempts=3,
        metadata={
            "backend": "fake",
            "success": True,
        },
    )

    assert event.timestamp == timestamp
    assert event.duration_ms == 125.5
    assert event.attempt == 1
    assert event.max_attempts == 3
    assert dict(event.metadata) == {
        "backend": "fake",
        "success": True,
    }


def test_metadata_is_defensively_copied() -> None:
    metadata = {
        "key": "value",
    }

    event = ObservabilityEvent(
        event="action.completed",
        source="router",
        metadata=metadata,
    )

    metadata["key"] = "changed"

    assert event.metadata["key"] == "value"


def test_metadata_is_immutable() -> None:
    event = ObservabilityEvent(
        event="action.completed",
        source="router",
        metadata={
            "key": "value",
        },
    )

    with pytest.raises(TypeError):
        event.metadata["key"] = "changed"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("event", ""),
        ("source", ""),
        ("request_id", ""),
        ("run_id", ""),
        ("task_id", ""),
        ("action_name", ""),
        ("status", ""),
        ("decision", ""),
        ("error", ""),
    ],
)
def test_empty_string_fields_are_rejected(
    field_name: str,
    value: str,
) -> None:
    kwargs = {
        "event": "action.started",
        "source": "pipeline",
    }

    if field_name in {
        "event",
        "source",
    }:
        kwargs[field_name] = value
    else:
        kwargs[field_name] = value

    with pytest.raises(ObservabilityEventError):
        ObservabilityEvent(
            **kwargs
        )


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(
        ObservabilityEventError,
        match="timezone-aware",
    ):
        ObservabilityEvent(
            event="action.started",
            source="pipeline",
            timestamp=datetime(
                2026,
                9,
                29,
            ),
        )


@pytest.mark.parametrize(
    "duration_ms",
    [
        -1,
        -0.1,
        True,
        False,
        "10",
    ],
)
def test_invalid_duration_is_rejected(
    duration_ms: object,
) -> None:
    with pytest.raises(
        ObservabilityEventError
    ):
        ObservabilityEvent(
            event="action.completed",
            source="pipeline",
            duration_ms=duration_ms,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "attempt",
    [
        0,
        -1,
        True,
        1.5,
        "1",
    ],
)
def test_invalid_attempt_is_rejected(
    attempt: object,
) -> None:
    with pytest.raises(
        ObservabilityEventError
    ):
        ObservabilityEvent(
            event="recovery.attempt",
            source="recovery",
            attempt=attempt,  # type: ignore[arg-type]
        )


def test_attempt_cannot_exceed_max_attempts() -> None:
    with pytest.raises(
        ObservabilityEventError,
        match="attempt cannot exceed max_attempts",
    ):
        ObservabilityEvent(
            event="recovery.attempt",
            source="recovery",
            attempt=4,
            max_attempts=3,
        )


def test_invalid_metadata_is_rejected() -> None:
    with pytest.raises(
        ObservabilityEventError,
        match="metadata must be a mapping",
    ):
        ObservabilityEvent(
            event="action.started",
            source="pipeline",
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_to_dict_contains_serializable_event_shape() -> None:
    timestamp = datetime(
        2026,
        9,
        29,
        10,
        30,
        tzinfo=timezone.utc,
    )

    event = ObservabilityEvent(
        event="action.failed",
        source="browser_adapter",
        timestamp=timestamp,
        request_id="request-1",
        run_id="run-1",
        task_id="task-1",
        action_kind=ActionKind.BROWSER,
        action_name="browser_click",
        status="failed",
        duration_ms=52,
        decision="allow",
        attempt=2,
        max_attempts=3,
        error="backend failure",
        metadata={
            "retryable": True,
        },
    )

    payload = event.to_dict()

    assert payload == {
        "event_id": event.event_id,
        "timestamp": timestamp.isoformat(),
        "event": "action.failed",
        "source": "browser_adapter",
        "request_id": "request-1",
        "run_id": "run-1",
        "task_id": "task-1",
        "action_kind": ActionKind.BROWSER.value,
        "action_name": "browser_click",
        "status": "failed",
        "duration_ms": 52,
        "decision": "allow",
        "attempt": 2,
        "max_attempts": 3,
        "error": "backend failure",
        "metadata": {
            "retryable": True,
        },
    }


def test_event_is_frozen() -> None:
    event = ObservabilityEvent(
        event="action.started",
        source="pipeline",
    )

    with pytest.raises(
        AttributeError
    ):
        event.status = "completed"  # type: ignore[misc]
