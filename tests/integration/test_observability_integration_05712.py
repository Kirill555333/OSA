from __future__ import annotations

from osa.actions import ActionKind, ActionRequest, ActionResult
from osa.actions.router import ActionRouter
from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableActionExecution,
    ObservableActionRouter,
    ObservabilityEvent,
    ObservabilityEventQuery,
    ObservabilityEventQueryAPI,
    ObservabilityRedactor,
    RedactingObservabilityLogger,
)


class _Handler:
    def __init__(
        self,
        *,
        result_output: str = "ok",
    ) -> None:
        self.result_output = result_output
        self.calls = 0

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        self.calls += 1

        return ActionResult.succeeded(
            request.request_id,
            output=self.result_output,
            data={
                "action": request.name,
            },
        )


def _request(
    *,
    name: str = "browser_open",
    arguments: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.BROWSER,
        name=name,
        arguments=(
            {"url": "https://example.com"}
            if arguments is None
            else arguments
        ),
        metadata=metadata or {},
    )


def test_router_and_execution_events_share_one_logger() -> None:
    logger = InMemoryObservabilityLogger()

    backend_handler = _Handler()

    execution = ObservableActionExecution.browser(
        backend_handler,
        logger=logger,
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=logger,
    )

    request = _request(
        name="browser_open",
        metadata={
            "run_id": "run-integration",
            "task_id": "task-integration",
        },
    )

    result = observed_router.dispatch(request)

    assert result.success is True
    assert backend_handler.calls == 1

    events = logger.events()

    assert [event.event for event in events] == [
        "router.started",
        "execution.started",
        "execution.completed",
        "router.completed",
    ]

    assert all(
        event.request_id == request.request_id
        for event in events
    )

    assert all(
        event.run_id == "run-integration"
        for event in events
    )

    assert all(
        event.task_id == "task-integration"
        for event in events
    )


def test_query_api_can_trace_one_request_across_layers() -> None:
    logger = InMemoryObservabilityLogger()

    handler = _Handler()

    execution = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=logger,
    )

    request = _request()

    observed_router.dispatch(request)

    api = ObservabilityEventQueryAPI(logger)

    traced = api.query(
        ObservabilityEventQuery(
            request_id=request.request_id,
        )
    )

    assert [event.event for event in traced] == [
        "router.started",
        "execution.started",
        "execution.completed",
        "router.completed",
    ]


def test_query_can_filter_by_run_and_task() -> None:
    logger = InMemoryObservabilityLogger()

    handler = _Handler()

    execution = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=logger,
    )

    first = _request(
        metadata={
            "run_id": "run-a",
            "task_id": "task-a",
        }
    )

    second = _request(
        metadata={
            "run_id": "run-b",
            "task_id": "task-b",
        }
    )

    observed_router.dispatch(first)
    observed_router.dispatch(second)

    api = ObservabilityEventQueryAPI(logger)

    first_trace = api.query(
        run_id="run-a",
        task_id="task-a",
    )

    second_trace = api.query(
        run_id="run-b",
        task_id="task-b",
    )

    assert len(first_trace) == 4
    assert len(second_trace) == 4

    assert all(
        event.run_id == "run-a"
        for event in first_trace
    )

    assert all(
        event.run_id == "run-b"
        for event in second_trace
    )


def test_redaction_survives_full_execution_chain() -> None:
    raw_logger = InMemoryObservabilityLogger()

    safe_logger = RedactingObservabilityLogger(
        raw_logger,
        redactor=ObservabilityRedactor(),
    )

    secret = "VERY_SECRET_TOKEN_VALUE"

    handler = _Handler(
        result_output=secret,
    )

    execution = ObservableActionExecution.browser(
        handler,
        logger=safe_logger,
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=safe_logger,
    )

    request = _request(
        name="browser_type",
        arguments={
            "text": secret,
        },
    )

    result = observed_router.dispatch(request)

    assert result.success is True
    assert result.output == secret

    stored_events = raw_logger.events()

    assert len(stored_events) == 4

    for event in stored_events:
        serialized = str(event.to_dict())
        assert secret not in serialized


def test_query_api_reads_redacted_events() -> None:
    raw_logger = InMemoryObservabilityLogger()

    safe_logger = RedactingObservabilityLogger(
        raw_logger,
        redactor=ObservabilityRedactor(),
    )

    secret = "SECRET_QUERY_VALUE"

    safe_logger.log(
        ObservabilityEvent(
            event="execution.failed",
            source="browser",
            status="failed",
            error="backend failure",
            metadata={
                "token": secret,
                "nested": {
                    "password": secret,
                },
            },
        )
    )

    api = ObservabilityEventQueryAPI(
        raw_logger
    )

    result = api.query(
        event="execution.failed"
    )

    assert len(result) == 1

    stored = result[0]

    assert stored.metadata["token"] == "[REDACTED]"
    assert stored.metadata["nested"]["password"] == (
        "[REDACTED]"
    )


def test_query_latest_preserves_cross_layer_order() -> None:
    logger = InMemoryObservabilityLogger()

    handler = _Handler()

    execution = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=logger,
    )

    observed_router.dispatch(
        _request()
    )

    api = ObservabilityEventQueryAPI(logger)

    latest = api.latest(
        2,
        request_id=(
            logger.events()[0].request_id
        ),
    )

    assert [event.event for event in latest] == [
        "router.completed",
        "execution.completed",
    ]


def test_observability_failures_do_not_change_execution_result() -> None:
    class _BrokenLogger:
        def log(
            self,
            event,
            **data,
        ) -> None:
            raise RuntimeError(
                "observability failure"
            )

    handler = _Handler(
        result_output="real-result"
    )

    execution = ObservableActionExecution.browser(
        handler,
        logger=_BrokenLogger(),
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=_BrokenLogger(),
    )

    result = observed_router.dispatch(
        _request()
    )

    assert result.success is True
    assert result.output == "real-result"
    assert handler.calls == 1


def test_request_arguments_are_never_copied_into_observability() -> None:
    logger = InMemoryObservabilityLogger()

    handler = _Handler()

    execution = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    router = ActionRouter(
        {
            ActionKind.BROWSER: execution,
        }
    )

    observed_router = ObservableActionRouter(
        router,
        logger=logger,
    )

    secret = "NEVER_LOG_THIS"

    request = _request(
        name="browser_type",
        arguments={
            "text": secret,
            "nested": {
                "value": secret,
            },
        },
    )

    observed_router.dispatch(request)

    for event in logger.events():
        assert secret not in str(
            event.to_dict()
        )


def test_observability_events_are_immutable_after_storage() -> None:
    logger = InMemoryObservabilityLogger()

    event = ObservabilityEvent(
        event="test.event",
        source="test",
        status="completed",
        metadata={
            "value": "original",
        },
    )

    logger.log(event)

    stored = logger.events()[0]

    assert stored is event
    assert stored.metadata["value"] == "original"

    try:
        stored.metadata["value"] = "changed"
    except TypeError:
        pass
    else:
        raise AssertionError(
            "stored metadata must be immutable"
        )
