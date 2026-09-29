from __future__ import annotations

import pytest

from osa.observability.logger import (
    InMemoryObservabilityLogger,
)
from osa.observability.recovery_controller import (
    ObservableRecoveryController,
    ObservableRecoveryControllerError,
)

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryFailureKind,
    RecoveryRequest,
    RecoveryResult,
)


class FakeController:
    def __init__(
        self,
        result=None,
        *,
        exception: Exception | None = None,
    ) -> None:
        self.result = result
        self.exception = exception
        self.requests: list[RecoveryRequest] = []

    def execute(
        self,
        request: RecoveryRequest,
    ):
        self.requests.append(request)

        if self.exception is not None:
            raise self.exception

        return self.result


class RaisingLogger:
    def __init__(self) -> None:
        self.calls = 0

    def log(
        self,
        event,
        **data,
    ) -> None:
        self.calls += 1
        raise RuntimeError(
            "logger unavailable"
        )


def _request() -> RecoveryRequest:
    return RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
        attempt=1,
        max_attempts=3,
        metadata={
            "token": "super-secret",
        },
    )


def _successful_result() -> RecoveryResult:
    return RecoveryResult.succeeded(
        result={
            "ok": True,
        },
    )


def _failed_result(
    *,
    exhausted: bool,
) -> RecoveryResult:
    attempts = (
        RecoveryAttempt(
            attempt=1,
            max_attempts=2,
            success=False,
            failure_kind=RecoveryFailureKind.RETRYABLE,
            error="temporary failure",
            output_present=False,
            metadata={
                "secret": "do-not-log",
            },
        ),
        RecoveryAttempt(
            attempt=2,
            max_attempts=2,
            success=False,
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
            error="backend unavailable",
            output_present=False,
            metadata={
                "secret": "do-not-log",
            },
        ),
    )

    return RecoveryResult.failed(
        attempts=attempts,
        failure_kind=(
            RecoveryFailureKind.BACKEND_ERROR
        ),
        error="backend unavailable",
        exhausted=exhausted,
        result=None,
    )


def test_success_emits_started_and_completed():
    logger = InMemoryObservabilityLogger()

    controller = FakeController(
        _successful_result()
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    result = observable.execute(
        _request()
    )

    assert result.success is True
    assert len(logger.events()) == 2

    names = tuple(
        event.event
        for event in logger.events()
    )

    assert names == (
        "recovery.started",
        "recovery.completed",
    )

    completed = logger.events()[-1]

    assert completed.status == "completed"
    assert completed.attempt == 1
    assert completed.max_attempts == 3
    assert completed.metadata[
        "attempt_count"
    ] == 0


def test_retry_attempts_are_observable_without_raw_request_metadata():
    logger = InMemoryObservabilityLogger()

    controller = FakeController(
        _failed_result(
            exhausted=False
        )
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    result = observable.execute(
        _request()
    )

    assert result.success is False

    attempt_events = tuple(
        event
        for event in logger.events()
        if event.event == "recovery.attempt"
    )

    assert len(attempt_events) == 2

    assert attempt_events[0].decision == "retry"
    assert attempt_events[1].decision == "stop"

    for event in attempt_events:
        serialized = repr(event)

        assert "super-secret" not in serialized
        assert "do-not-log" not in serialized

        assert event.metadata[
            "attempt_metadata_present"
        ] is True


def test_exhausted_recovery_emits_exhausted_event():
    logger = InMemoryObservabilityLogger()

    controller = FakeController(
        _failed_result(
            exhausted=True
        )
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    result = observable.execute(
        _request()
    )

    assert result.success is False
    assert result.exhausted is True

    names = tuple(
        event.event
        for event in logger.events()
    )

    assert names == (
        "recovery.started",
        "recovery.attempt",
        "recovery.attempt",
        "recovery.exhausted",
    )

    exhausted = logger.events()[-1]

    assert exhausted.status == "exhausted"
    assert exhausted.metadata[
        "attempt_count"
    ] == 2


def test_non_exhausted_failure_emits_failed_event():
    logger = InMemoryObservabilityLogger()

    controller = FakeController(
        _failed_result(
            exhausted=False
        )
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    result = observable.execute(
        _request()
    )

    assert result.success is False
    assert result.exhausted is False

    assert logger.events()[-1].event == (
        "recovery.failed"
    )
    assert logger.events()[-1].status == "failed"


def test_controller_exception_is_re_raised_and_observed():
    logger = InMemoryObservabilityLogger()

    controller = FakeController(
        exception=RuntimeError(
            "controller failure"
        )
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    with pytest.raises(
        RuntimeError,
        match="controller failure",
    ):
        observable.execute(
            _request()
        )

    assert logger.events()[-1].event == (
        "recovery.failed"
    )
    assert logger.events()[-1].metadata[
        "controller_exception"
    ] is True


def test_logger_failure_never_changes_recovery_result():
    logger = RaisingLogger()

    controller = FakeController(
        _successful_result()
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    result = observable.execute(
        _request()
    )

    assert result.success is True
    assert logger.calls > 0


def test_invalid_request_is_rejected_before_controller_execution():
    logger = InMemoryObservabilityLogger()
    controller = FakeController(
        _successful_result()
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    with pytest.raises(
        TypeError,
        match="RecoveryRequest",
    ):
        observable.execute(
            object()
        )

    assert controller.requests == []
    assert logger.events() == ()


def test_invalid_controller_result_fails_closed():
    logger = InMemoryObservabilityLogger()

    controller = FakeController(
        object()
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    with pytest.raises(
        ObservableRecoveryControllerError,
        match="invalid result",
    ):
        observable.execute(
            _request()
        )

    assert logger.events()[-1].event == (
        "recovery.failed"
    )
    assert logger.events()[-1].metadata[
        "invalid_result"
    ] is True


def test_action_context_is_preserved_in_observability():
    logger = InMemoryObservabilityLogger()

    request = RecoveryRequest(
        run_id="run-42",
        task_id="task-42",
        request_id="req-42",
        action_kind="tool",
        action_name="demo_tool",
        attempt=1,
        max_attempts=2,
        metadata={
            "password": "hidden",
        },
    )

    controller = FakeController(
        RecoveryResult.succeeded(
            result="ok"
        )
    )

    observable = ObservableRecoveryController(
        controller,
        logger=logger,
    )

    observable.execute(
        request
    )

    event = logger.events()[0]

    assert event.run_id == "run-42"
    assert event.task_id == "task-42"
    assert event.request_id == "req-42"
    assert event.action_kind == "tool"
    assert event.action_name == "demo_tool"
    assert event.metadata[
        "metadata_present"
    ] is True

    assert "hidden" not in repr(event)
