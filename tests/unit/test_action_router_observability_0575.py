from __future__ import annotations

from dataclasses import dataclass

from osa.actions import ActionKind, ActionRequest, ActionResult
from osa.actions.router import ActionRouter
from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableActionRouter,
    ObservabilityContext,
    ObservabilityEvent,
)


@dataclass
class _Handler:
    kind: ActionKind
    success: bool = True
    error: str | None = None

    def execute(self, request: ActionRequest) -> ActionResult:
        if self.success:
            return ActionResult.succeeded(
                request.request_id,
                output="ok",
                data={"handled": request.name},
            )

        return ActionResult.failed(
            request.request_id,
            self.error or "backend failure",
        )


class _ExplodingRouter:
    def dispatch(self, request: ActionRequest) -> ActionResult:
        raise RuntimeError("router exploded")

    @property
    def supported_kinds(self) -> tuple[str, ...]:
        return ()

    def handler_for(self, kind: ActionKind):
        return None

    def register(self, handler) -> None:
        raise RuntimeError("not supported")

    def unregister(self, kind: ActionKind):
        return None


def _request(
    *,
    kind: ActionKind = ActionKind.BROWSER,
    name: str = "browser_open",
    metadata: dict[str, object] | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=kind,
        name=name,
        arguments={"url": "https://example.com"},
        metadata=metadata or {},
    )


def _events(
    logger: InMemoryObservabilityLogger,
) -> list[ObservabilityEvent]:
    return logger.events()


def test_success_emits_started_and_completed_events() -> None:
    logger = InMemoryObservabilityLogger()
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )
    observed = ObservableActionRouter(router, logger)

    request = _request()
    result = observed.dispatch(request)

    assert result.success is True
    assert result.request_id == request.request_id

    events = _events(logger)
    assert [event.event for event in events] == [
        "router.started",
        "router.completed",
    ]
    assert all(
        event.request_id == request.request_id
        for event in events
    )
    assert all(
        event.action_kind == ActionKind.BROWSER.value
        for event in events
    )
    assert all(
        event.action_name == "browser_open"
        for event in events
    )
    assert events[1].status == "completed"
    assert events[1].duration_ms is not None


def test_failed_result_emits_started_and_failed_events() -> None:
    logger = InMemoryObservabilityLogger()
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(
                ActionKind.BROWSER,
                success=False,
                error="backend failed",
            ),
        }
    )
    observed = ObservableActionRouter(router, logger)

    result = observed.dispatch(_request())

    assert result.success is False
    assert result.error == "backend failed"

    events = _events(logger)
    assert [event.event for event in events] == [
        "router.started",
        "router.failed",
    ]
    assert events[-1].error == "backend failed"


def test_router_exception_is_logged_and_reraised() -> None:
    logger = InMemoryObservabilityLogger()
    observed = ObservableActionRouter(
        _ExplodingRouter(),
        logger,
    )

    try:
        observed.dispatch(_request())
    except RuntimeError as exc:
        assert str(exc) == "router exploded"
    else:
        raise AssertionError("expected RuntimeError")

    events = _events(logger)
    assert [event.event for event in events] == [
        "router.started",
        "router.failed",
    ]
    assert events[-1].error == "router exploded"


def test_request_context_is_derived_from_metadata() -> None:
    logger = InMemoryObservabilityLogger()
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )
    observed = ObservableActionRouter(router, logger)

    observed.dispatch(
        _request(
            metadata={
                "run_id": "run-123",
                "task_id": "task-456",
            }
        )
    )

    events = _events(logger)
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
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )
    context = ObservabilityContext(
        run_id="run-explicit",
        task_id="task-explicit",
    )

    observed = ObservableActionRouter(
        router,
        logger,
        context_provider=lambda request: context,
    )

    observed.dispatch(_request())

    events = _events(logger)
    assert all(
        event.run_id == "run-explicit"
        for event in events
    )
    assert all(
        event.task_id == "task-explicit"
        for event in events
    )


def test_observability_never_logs_raw_arguments() -> None:
    logger = InMemoryObservabilityLogger()
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )
    observed = ObservableActionRouter(router, logger)

    secret = "TOP_SECRET_ARGUMENT"
    request = ActionRequest(
        kind=ActionKind.BROWSER,
        name="browser_type",
        arguments={"text": secret},
    )

    observed.dispatch(request)

    events = _events(logger)

    for event in events:
        serialized = str(event.to_dict())
        assert secret not in serialized

    started_event = next(
        event
        for event in events
        if event.event == "router.started"
    )

    assert started_event.metadata["arguments_present"] is True


def test_passed_logger_identity_is_preserved_even_when_empty() -> None:
    logger = InMemoryObservabilityLogger()
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )

    observed = ObservableActionRouter(router, logger)

    assert observed.logger is logger


def test_router_delegation_api_is_preserved() -> None:
    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )
    observed = ObservableActionRouter(router)

    assert observed.router is router
    assert observed.supported_kinds == router.supported_kinds
    assert observed.handler_for(ActionKind.BROWSER) is not None


def test_logging_failure_does_not_break_routing() -> None:
    class _BrokenLogger:
        def log(self, event, **data) -> None:
            raise RuntimeError("logger broken")

    router = ActionRouter(
        {
            ActionKind.BROWSER: _Handler(ActionKind.BROWSER),
        }
    )
    observed = ObservableActionRouter(
        router,
        _BrokenLogger(),
    )

    result = observed.dispatch(_request())

    assert result.success is True
