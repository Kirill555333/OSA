from __future__ import annotations

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableActionSafetyPipeline,
    ObservabilityContext,
)


class FakePipeline:
    def __init__(
        self,
        result: ActionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.requests: list[ActionRequest] = []

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        self.requests.append(
            request
        )

        if self.error is not None:
            raise self.error

        if self.result is None:
            return ActionResult.succeeded(
                request.request_id,
                output="ok",
            )

        return self.result


def make_request() -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="integration_tool",
        arguments={
            "value": "test",
        },
        metadata={
            "run_id": "run-1",
            "task_id": "task-1",
        },
    )


def test_success_emits_started_and_completed() -> None:
    request = make_request()

    pipeline = FakePipeline(
        ActionResult.succeeded(
            request.request_id,
            output="ok",
        )
    )
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    result = observed.execute(
        request
    )

    assert result.success is True
    assert pipeline.requests == [
        request,
    ]

    events = logger.events()

    assert [event.event for event in events] == [
        "action.started",
        "action.completed",
    ]

    assert events[0].request_id == request.request_id
    assert events[0].run_id == "run-1"
    assert events[0].task_id == "task-1"
    assert events[0].action_kind == ActionKind.TOOL.value
    assert events[0].action_name == "integration_tool"

    assert events[1].request_id == request.request_id
    assert events[1].status == "completed"
    assert events[1].duration_ms is not None


def test_failed_result_emits_started_and_failed() -> None:
    request = make_request()

    pipeline = FakePipeline(
        ActionResult.failed(
            request.request_id,
            error="backend failure",
        )
    )
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    result = observed.execute(
        request
    )

    assert result.success is False
    assert result.error == "backend failure"

    events = logger.events()

    assert [event.event for event in events] == [
        "action.started",
        "action.failed",
    ]

    assert events[1].error == "backend failure"
    assert events[1].duration_ms is not None


def test_pipeline_exception_emits_failed_and_re_raises() -> None:
    request = make_request()

    pipeline = FakePipeline(
        error=RuntimeError(
            "boom"
        )
    )
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    try:
        observed.execute(
            request
        )
        raise AssertionError(
            "expected RuntimeError"
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"

    events = logger.events()

    assert [event.event for event in events] == [
        "action.started",
        "action.failed",
    ]

    assert events[1].error == "boom"


def test_invalid_pipeline_result_is_normalized_to_failure() -> None:
    request = make_request()

    class InvalidPipeline:
        def execute(
            self,
            request: ActionRequest,
        ) -> object:
            return object()

    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        InvalidPipeline(),
        logger=logger,
    )

    result = observed.execute(
        request
    )

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == "invalid_action_result"

    events = logger.events()

    assert [event.event for event in events] == [
        "action.started",
        "action.failed",
    ]

    assert events[1].error == "invalid_action_result"


def test_request_id_mismatch_is_normalized_to_failure() -> None:
    request = make_request()

    pipeline = FakePipeline(
        ActionResult.succeeded(
            "different-request",
            output="wrong",
        )
    )
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    result = observed.execute(
        request
    )

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == (
        "action_result_request_id_mismatch"
    )

    events = logger.events()

    assert events[-1].event == "action.failed"
    assert events[-1].error == (
        "action_result_request_id_mismatch"
    )


def test_explicit_context_provider_is_used() -> None:
    request = make_request()

    pipeline = FakePipeline()
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
        context_provider=lambda _: ObservabilityContext(
            run_id="explicit-run",
            task_id="explicit-task",
            request_id=request.request_id,
        ),
    )

    observed.execute(
        request
    )

    events = logger.events()

    assert events[0].run_id == "explicit-run"
    assert events[0].task_id == "explicit-task"


def test_logging_failure_does_not_break_execution() -> None:
    request = make_request()

    pipeline = FakePipeline()

    class BrokenLogger:
        def log(self, event: object) -> None:
            raise RuntimeError(
                "logger failure"
            )

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=BrokenLogger(),  # type: ignore[arg-type]
    )

    result = observed.execute(
        request
    )

    assert result.success is True
    assert pipeline.requests == [
        request,
    ]


def test_wrapper_exposes_original_pipeline() -> None:
    pipeline = FakePipeline()
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    assert observed.pipeline is pipeline
    assert observed.logger is logger


def test_empty_arguments_are_not_recorded_as_raw_arguments() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="integration_tool",
        arguments={},
    )

    pipeline = FakePipeline()
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    observed.execute(
        request
    )

    event = logger.events()[0]

    assert event.metadata == {
        "arguments_present": False,
    }


def test_arguments_presence_does_not_leak_argument_values() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="integration_tool",
        arguments={
            "secret": "do-not-record",
        },
    )

    pipeline = FakePipeline()
    logger = InMemoryObservabilityLogger()

    observed = ObservableActionSafetyPipeline(
        pipeline,
        logger=logger,
    )

    observed.execute(
        request
    )

    event = logger.events()[0]

    assert event.metadata == {
        "arguments_present": True,
    }
    assert "do-not-record" not in repr(
        event.to_dict()
    )
