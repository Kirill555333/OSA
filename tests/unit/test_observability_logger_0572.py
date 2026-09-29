from __future__ import annotations

from datetime import datetime, timezone
from threading import Thread

import pytest

from osa.actions.contracts import ActionKind
from osa.observability import (
    InMemoryObservabilityLogger,
    NullObservabilityLogger,
    ObservabilityEvent,
    ObservabilityLoggerError,
)


def test_null_logger_does_not_store_events() -> None:
    logger = NullObservabilityLogger()

    assert logger.log(
        "action.started",
        source="pipeline",
    ) is None


def test_in_memory_logger_stores_observability_event() -> None:
    logger = InMemoryObservabilityLogger()

    event = ObservabilityEvent(
        event="action.started",
        source="pipeline",
        request_id="request-1",
    )

    recorded = logger.log(event)

    assert recorded is event
    assert logger.events() == (
        event,
    )
    assert len(logger) == 1


def test_in_memory_logger_accepts_legacy_string_event() -> None:
    logger = InMemoryObservabilityLogger()

    recorded = logger.log(
        "tool.call",
        source="agent",
        tool="browser_open",
        request_id="request-1",
        run_id="run-1",
        task_id="task-1",
    )

    assert recorded.event == "tool.call"
    assert recorded.source == "agent"
    assert recorded.request_id == "request-1"
    assert recorded.run_id == "run-1"
    assert recorded.task_id == "task-1"
    assert recorded.metadata["tool"] == "browser_open"


def test_legacy_event_maps_known_execution_fields() -> None:
    logger = InMemoryObservabilityLogger()

    recorded = logger.log(
        "action.completed",
        source="router",
        request_id="request-1",
        run_id="run-1",
        task_id="task-1",
        action_kind=ActionKind.DESKTOP,
        action_name="desktop_click_point",
        status="completed",
        duration_ms=42,
        decision="allow",
        attempt=1,
        max_attempts=2,
        error=None,
        backend="fake",
    )

    assert recorded.action_kind == ActionKind.DESKTOP.value
    assert recorded.action_name == "desktop_click_point"
    assert recorded.status == "completed"
    assert recorded.duration_ms == 42
    assert recorded.decision == "allow"
    assert recorded.attempt == 1
    assert recorded.max_attempts == 2
    assert recorded.error is None
    assert recorded.metadata["backend"] == "fake"


def test_legacy_event_merges_metadata() -> None:
    logger = InMemoryObservabilityLogger()

    recorded = logger.log(
        "action.failed",
        source="adapter",
        metadata={
            "backend": "fake",
            "retryable": True,
        },
        error="backend failure",
        extra="value",
    )

    assert recorded.metadata == {
        "backend": "fake",
        "retryable": True,
        "extra": "value",
    }
    assert recorded.error == "backend failure"


def test_logging_observability_event_with_extra_data_is_rejected() -> None:
    logger = InMemoryObservabilityLogger()

    event = ObservabilityEvent(
        event="action.started",
        source="pipeline",
    )

    with pytest.raises(
        ObservabilityLoggerError,
        match="Additional data",
    ):
        logger.log(
            event,
            status="started",
        )


def test_invalid_event_type_is_rejected() -> None:
    logger = InMemoryObservabilityLogger()

    with pytest.raises(
        ObservabilityLoggerError,
        match="event must be",
    ):
        logger.log(123)  # type: ignore[arg-type]


def test_empty_string_event_is_rejected() -> None:
    logger = InMemoryObservabilityLogger()

    with pytest.raises(
        ObservabilityLoggerError,
        match="event cannot be empty",
    ):
        logger.log("   ")


def test_invalid_metadata_is_rejected() -> None:
    logger = InMemoryObservabilityLogger()

    with pytest.raises(
        ObservabilityLoggerError,
        match="metadata must be a mapping",
    ):
        logger.log(
            "action.started",
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_events_returns_snapshot() -> None:
    logger = InMemoryObservabilityLogger()

    first = logger.log(
        "action.started",
        source="pipeline",
    )

    snapshot = logger.events()
    logger.log(
        "action.completed",
        source="pipeline",
    )

    assert snapshot == (
        first,
    )
    assert len(logger.events()) == 2


def test_clear_removes_all_events() -> None:
    logger = InMemoryObservabilityLogger()

    logger.log(
        "action.started",
        source="pipeline",
    )
    logger.log(
        "action.completed",
        source="pipeline",
    )

    logger.clear()

    assert logger.events() == ()
    assert len(logger) == 0


def test_event_with_explicit_timestamp_is_preserved() -> None:
    logger = InMemoryObservabilityLogger()

    timestamp = datetime(
        2026,
        9,
        29,
        12,
        0,
        tzinfo=timezone.utc,
    )

    recorded = logger.log(
        "action.completed",
        source="pipeline",
        timestamp=timestamp,
    )

    assert recorded.timestamp == timestamp


def test_concurrent_logging_is_thread_safe() -> None:
    logger = InMemoryObservabilityLogger()

    def write_events(prefix: str) -> None:
        for index in range(50):
            logger.log(
                "action.started",
                source="test",
                request_id=f"{prefix}-{index}",
            )

    threads = [
        Thread(
            target=write_events,
            args=(f"thread-{index}",),
        )
        for index in range(4)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(logger) == 200
    assert len(
        {
            event.request_id
            for event in logger.events()
        }
    ) == 200
