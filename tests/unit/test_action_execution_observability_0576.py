from __future__ import annotations

import pytest

from osa.actions import ActionKind, ActionRequest, ActionResult
from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableActionExecution,
    ObservabilityContext,
    ObservabilityEvent,
)


class _Handler:
    def __init__(
        self,
        *,
        result: ActionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[ActionRequest] = []

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls.append(request)

        if self.error is not None:
            raise self.error

        if self.result is not None:
            return self.result

        return ActionResult.succeeded(
            request.request_id,
            output="ok",
            data={"value": 1},
        )


def _request(
    kind: ActionKind,
    *,
    name: str = "action",
    metadata: dict[str, object] | None = None,
    arguments: dict[str, object] | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=kind,
        name=name,
        arguments=(
            {"value": "test"}
            if arguments is None
            else arguments
        ),
        metadata=metadata or {},
    )


@pytest.mark.parametrize(
    ("kind", "constructor"),
    [
        (ActionKind.BROWSER, ObservableActionExecution.browser),
        (ActionKind.DESKTOP, ObservableActionExecution.desktop),
        (ActionKind.TOOL, ObservableActionExecution.tool),
    ],
)
def test_execution_events_cover_browser_desktop_and_tool(
    kind: ActionKind,
    constructor,
) -> None:
    logger = InMemoryObservabilityLogger()
    handler = _Handler()
    observed = constructor(
        handler,
        logger=logger,
    )

    request = _request(
        kind,
        name=f"{kind.value}_test",
    )

    result = observed.execute(request)

    assert result.success is True
    assert handler.calls == [request]

    events = logger.events()

    assert [event.event for event in events] == [
        "execution.started",
        "execution.completed",
    ]
    assert all(
        event.source == kind.value
        for event in events
    )
    assert all(
        event.request_id == request.request_id
        for event in events
    )
    assert all(
        event.action_kind == kind.value
        for event in events
    )


def test_failed_result_emits_execution_failed() -> None:
    logger = InMemoryObservabilityLogger()
    request = _request(
        ActionKind.BROWSER,
        name="browser_click",
    )

    handler = _Handler(
        result=ActionResult.failed(
            request.request_id,
            "click failed",
        )
    )

    observed = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    result = observed.execute(request)

    assert result.success is False
    assert result.error == "click failed"

    events = logger.events()

    assert [event.event for event in events] == [
        "execution.started",
        "execution.failed",
    ]
    assert events[-1].error == "click failed"
    assert events[-1].status == "failed"


def test_handler_exception_is_logged_and_reraised() -> None:
    logger = InMemoryObservabilityLogger()
    handler = _Handler(
        error=RuntimeError("backend exploded"),
    )

    observed = ObservableActionExecution.desktop(
        handler,
        logger=logger,
    )

    request = _request(
        ActionKind.DESKTOP,
        name="desktop_click_element",
    )

    with pytest.raises(RuntimeError, match="backend exploded"):
        observed.execute(request)

    events = logger.events()

    assert [event.event for event in events] == [
        "execution.started",
        "execution.failed",
    ]
    assert events[-1].error == "backend exploded"


def test_invalid_result_is_normalized_to_failed_action_result() -> None:
    class _InvalidHandler:
        def execute(self, request: ActionRequest):
            return "invalid-result"

    logger = InMemoryObservabilityLogger()
    observed = ObservableActionExecution.tool(
        _InvalidHandler(),
        logger=logger,
    )

    request = _request(
        ActionKind.TOOL,
        name="tool_test",
    )

    result = observed.execute(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == "wrapped handler returned invalid result"

    events = logger.events()

    assert [event.event for event in events] == [
        "execution.started",
        "execution.failed",
    ]


def test_mismatched_result_is_normalized_to_failed_action_result() -> None:
    logger = InMemoryObservabilityLogger()
    request = _request(
        ActionKind.BROWSER,
        name="browser_open",
    )

    wrong_request = ActionRequest(
        kind=ActionKind.BROWSER,
        name="browser_open",
        arguments={"url": "https://other.example"},
    )

    handler = _Handler(
        result=ActionResult.succeeded(
            wrong_request.request_id,
            output="wrong",
        )
    )

    observed = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    result = observed.execute(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == (
        "wrapped handler returned mismatched request_id"
    )

    events = logger.events()

    assert [event.event for event in events] == [
        "execution.started",
        "execution.failed",
    ]


def test_context_is_derived_from_request_metadata() -> None:
    logger = InMemoryObservabilityLogger()
    handler = _Handler()

    observed = ObservableActionExecution.desktop(
        handler,
        logger=logger,
    )

    request = _request(
        ActionKind.DESKTOP,
        metadata={
            "run_id": "run-123",
            "task_id": "task-456",
        },
    )

    observed.execute(request)

    events = logger.events()

    assert all(
        event.run_id == "run-123"
        for event in events
    )
    assert all(
        event.task_id == "task-456"
        for event in events
    )


def test_explicit_context_provider_is_used() -> None:
    logger = InMemoryObservabilityLogger()
    handler = _Handler()
    context = ObservabilityContext(
        run_id="run-explicit",
        task_id="task-explicit",
    )

    observed = ObservableActionExecution.tool(
        handler,
        logger=logger,
        context_provider=lambda request: context,
    )

    request = _request(
        ActionKind.TOOL,
        metadata={
            "run_id": "run-from-request",
            "task_id": "task-from-request",
        },
    )

    observed.execute(request)

    events = logger.events()

    assert all(
        event.run_id == "run-explicit"
        for event in events
    )
    assert all(
        event.task_id == "task-explicit"
        for event in events
    )


def test_raw_arguments_are_never_logged() -> None:
    logger = InMemoryObservabilityLogger()
    handler = _Handler()

    secret = "TOP_SECRET_EXECUTION_VALUE"

    observed = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    request = _request(
        ActionKind.BROWSER,
        name="browser_type",
        arguments={
            "text": secret,
        },
    )

    observed.execute(request)

    for event in logger.events():
        serialized = str(event.to_dict())
        assert secret not in serialized

    started_event = next(
        event
        for event in logger.events()
        if event.event == "execution.started"
    )

    assert started_event.metadata["arguments_present"] is True


def test_passed_logger_identity_is_preserved_when_empty() -> None:
    logger = InMemoryObservabilityLogger()
    handler = _Handler()

    observed = ObservableActionExecution.browser(
        handler,
        logger=logger,
    )

    assert observed.logger is logger


def test_logging_failure_does_not_break_execution() -> None:
    class _BrokenLogger:
        def log(self, event, **data) -> None:
            raise RuntimeError("logger broken")

    handler = _Handler()

    observed = ObservableActionExecution.desktop(
        handler,
        logger=_BrokenLogger(),
    )

    request = _request(
        ActionKind.DESKTOP,
        name="desktop_type",
    )

    result = observed.execute(request)

    assert result.success is True


def test_invalid_source_is_rejected() -> None:
    handler = _Handler()

    with pytest.raises(
        Exception,
        match="source must be one of",
    ):
        ObservableActionExecution(
            handler,
            source="unknown",
        )


def test_source_factory_sets_expected_source() -> None:
    handler = _Handler()

    browser = ObservableActionExecution.browser(handler)
    desktop = ObservableActionExecution.desktop(handler)
    tool = ObservableActionExecution.tool(handler)

    assert browser.source == "browser"
    assert desktop.source == "desktop"
    assert tool.source == "tool"


def test_source_defaults_are_not_silently_guessed() -> None:
    handler = _Handler()

    with pytest.raises(Exception, match="source is required"):
        ObservableActionExecution(handler)
