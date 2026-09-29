from __future__ import annotations

from datetime import datetime, timezone

import pytest

from osa.observability import (
    InMemoryObservabilityLogger,
    ObservabilityEvent,
    ObservabilityEventQuery,
    ObservabilityEventQueryAPI,
    ObservabilityQueryError,
)


def _event(
    *,
    event: str,
    source: str,
    status: str,
    request_id: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    action_kind: str | None = None,
    action_name: str | None = None,
    decision: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ObservabilityEvent:
    return ObservabilityEvent(
        event=event,
        source=source,
        status=status,
        request_id=request_id,
        run_id=run_id,
        task_id=task_id,
        action_kind=action_kind,
        action_name=action_name,
        decision=decision,
        metadata=metadata or {},
        timestamp=datetime.now(timezone.utc),
    )


def _logger_with_events() -> InMemoryObservabilityLogger:
    logger = InMemoryObservabilityLogger()

    logger.log(
        _event(
            event="execution.started",
            source="browser",
            status="started",
            request_id="req-1",
            run_id="run-1",
            task_id="task-1",
            action_kind="browser",
            action_name="browser_open",
            metadata={
                "arguments_present": True,
                "attempt": 1,
            },
        )
    )

    logger.log(
        _event(
            event="execution.completed",
            source="browser",
            status="completed",
            request_id="req-1",
            run_id="run-1",
            task_id="task-1",
            action_kind="browser",
            action_name="browser_open",
            metadata={
                "output_present": True,
            },
        )
    )

    logger.log(
        _event(
            event="execution.failed",
            source="desktop",
            status="failed",
            request_id="req-2",
            run_id="run-1",
            task_id="task-2",
            action_kind="desktop",
            action_name="desktop_click",
        )
    )

    logger.log(
        _event(
            event="agent.chat.completed",
            source="agent",
            status="completed",
            metadata={
                "response_present": True,
            },
        )
    )

    return logger


def test_query_by_single_field() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    result = api.query(
        event="execution.completed"
    )

    assert len(result) == 1
    assert result[0].event == "execution.completed"


def test_query_by_multiple_fields() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    result = api.query(
        source="browser",
        run_id="run-1",
        action_name="browser_open",
    )

    assert len(result) == 2
    assert all(
        event.source == "browser"
        for event in result
    )
    assert all(
        event.run_id == "run-1"
        for event in result
    )


def test_tuple_filter_matches_any_value() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    result = api.query(
        status=("started", "failed")
    )

    assert len(result) == 2
    assert {
        event.status
        for event in result
    } == {
        "started",
        "failed",
    }


def test_query_object_is_supported() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    query = ObservabilityEventQuery(
        action_kind="browser",
        metadata={
            "arguments_present": True,
        },
    )

    result = api.query(query)

    assert len(result) == 1
    assert result[0].action_name == "browser_open"


def test_metadata_matching_uses_exact_values() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    result = api.query(
        metadata={
            "response_present": True,
        }
    )

    assert len(result) == 1
    assert result[0].event == "agent.chat.completed"


def test_limit_is_respected() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    result = api.query(
        source="browser",
        limit=1,
    )

    assert len(result) == 1
    assert result[0].event == "execution.started"


def test_count_returns_matching_event_count() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    assert api.count(
        run_id="run-1"
    ) == 3


def test_latest_returns_reverse_source_order() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    result = api.latest(
        2,
        run_id="run-1",
    )

    assert [event.event for event in result] == [
        "execution.failed",
        "execution.completed",
    ]


def test_latest_requires_positive_limit() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    with pytest.raises(
        ObservabilityQueryError,
        match="limit must be at least 1",
    ):
        api.latest(0)


def test_query_cannot_mix_query_object_and_filters() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    query = ObservabilityEventQuery(
        source="browser"
    )

    with pytest.raises(
        ObservabilityQueryError,
        match="cannot be combined",
    ):
        api.query(
            query,
            status="completed",
        )


def test_query_object_rejects_invalid_limit() -> None:
    with pytest.raises(
        ObservabilityQueryError,
        match="limit must be at least 1",
    ):
        ObservabilityEventQuery(
            limit=0
        )


def test_query_object_rejects_empty_string_filter() -> None:
    with pytest.raises(
        ObservabilityQueryError,
        match="cannot be empty",
    ):
        ObservabilityEventQuery(
            source="   "
        )


def test_query_object_rejects_non_string_tuple_item() -> None:
    with pytest.raises(
        ObservabilityQueryError,
        match="tuple items must be strings",
    ):
        ObservabilityEventQuery(
            event=("execution.started", 123)
        )


def test_query_skips_invalid_source_items() -> None:
    class _Source:
        def events(self):
            return [
                "invalid",
                _event(
                    event="valid",
                    source="test",
                    status="completed",
                ),
            ]

    api = ObservabilityEventQueryAPI(_Source())

    result = api.query()

    assert len(result) == 1
    assert result[0].event == "valid"


def test_source_property_is_preserved() -> None:
    logger = _logger_with_events()
    api = ObservabilityEventQueryAPI(logger)

    assert api.source is logger


def test_none_source_is_rejected() -> None:
    with pytest.raises(
        ObservabilityQueryError,
        match="source is required",
    ):
        ObservabilityEventQueryAPI(None)
