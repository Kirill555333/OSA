from __future__ import annotations

import pytest

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryFailureKind,
    RecoveryRequest,
)
from osa.recovery_controller import (
    RecoveryController,
    RecoveryControllerError,
)
from osa.recovery_policy import (
    DefaultRecoveryPolicy,
    RecoveryPolicyConfig,
)


def _request(
    *,
    max_attempts: int = 3,
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id="req-1",
        attempt=1,
        max_attempts=max_attempts,
    )


def _attempt_builder(
    request: RecoveryRequest,
    result: dict[str, object],
) -> RecoveryAttempt:
    if result["ok"]:
        return RecoveryAttempt(
            attempt=request.attempt,
            max_attempts=request.max_attempts,
            success=True,
            output_present=True,
        )

    return RecoveryAttempt(
        attempt=request.attempt,
        max_attempts=request.max_attempts,
        success=False,
        failure_kind=RecoveryFailureKind.BACKEND_ERROR,
        error=str(
            result.get(
                "error",
                "action_failed",
            )
        ),
    )


def _retry_policy() -> DefaultRecoveryPolicy:
    return DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.BACKEND_ERROR,
                }
            )
        )
    )


def test_success_does_not_retry() -> None:
    calls: list[int] = []

    def executor(request: RecoveryRequest):
        calls.append(request.attempt)

        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        _request()
    )

    assert result.success is True
    assert result.attempt_count == 1
    assert calls == [1]


def test_retry_reexecutes_same_executor() -> None:
    calls: list[int] = []

    def executor(request: RecoveryRequest):
        calls.append(request.attempt)

        if request.attempt == 1:
            return {
                "ok": False,
                "error": "temporary",
            }

        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        _request()
    )

    assert result.success is True
    assert result.attempt_count == 2
    assert calls == [1, 2]


def test_retry_preserves_request_identifiers() -> None:
    seen: list[tuple[str, str | None, str, int]] = []

    request = RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
        max_attempts=2,
    )

    def executor(current: RecoveryRequest):
        seen.append(
            (
                current.run_id or "",
                current.task_id,
                current.request_id or "",
                current.attempt,
            )
        )

        if current.attempt == 1:
            return {
                "ok": False,
                "error": "temporary",
            }

        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        request
    )

    assert result.success is True
    assert seen == [
        ("run-1", "task-1", "req-1", 1),
        ("run-1", "task-1", "req-1", 2),
    ]


def test_retry_delay_uses_injected_sleeper() -> None:
    delays: list[float] = []

    def executor(request: RecoveryRequest):
        if request.attempt == 1:
            return {
                "ok": False,
                "error": "temporary",
            }

        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.BACKEND_ERROR,
                    }
                ),
                default_delay_seconds=0.75,
            )
        ),
        sleeper=delays.append,
    )

    result = controller.execute(
        _request()
    )

    assert result.success is True
    assert delays == [0.75]


def test_policy_denial_preserves_real_failure_kind() -> None:
    def executor(request: RecoveryRequest):
        return {
            "ok": False,
            "error": "temporary",
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
    )

    result = controller.execute(
        _request()
    )

    assert result.success is False
    assert result.attempt_count == 1
    assert result.exhausted is False
    assert result.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )


def test_max_attempts_preserves_real_failure_kind() -> None:
    calls: list[int] = []

    def executor(request: RecoveryRequest):
        calls.append(request.attempt)

        return {
            "ok": False,
            "error": "still failing",
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        _request(max_attempts=2)
    )

    assert result.success is False
    assert result.exhausted is True
    assert result.attempt_count == 2
    assert calls == [1, 2]
    assert result.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert result.error == "still failing"


def test_executor_exception_can_be_retried() -> None:
    calls: list[int] = []

    def executor(request: RecoveryRequest):
        calls.append(request.attempt)

        if len(calls) == 1:
            raise RuntimeError(
                "temporary backend outage"
            )

        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        _request(max_attempts=2)
    )

    assert result.success is True
    assert result.attempt_count == 2
    assert calls == [1, 2]
    assert result.attempts[0].failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )


def test_executor_exception_stops_when_policy_denies_retry() -> None:
    calls: list[int] = []

    def executor(request: RecoveryRequest):
        calls.append(request.attempt)
        raise RuntimeError(
            "backend exploded"
        )

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
    )

    result = controller.execute(
        _request()
    )

    assert result.success is False
    assert result.attempt_count == 1
    assert result.exhausted is False
    assert result.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert result.error == "backend exploded"
    assert calls == [1]


def test_attempt_builder_failure_is_fail_closed() -> None:
    def executor(request: RecoveryRequest):
        return {
            "ok": True,
        }

    def broken_builder(
        request: RecoveryRequest,
        result,
    ):
        raise RuntimeError(
            "classifier exploded"
        )

    controller = RecoveryController(
        executor,
        attempt_builder=broken_builder,
    )

    result = controller.execute(
        _request()
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.INVALID_RESULT
    )


def test_invalid_attempt_builder_return_is_fail_closed() -> None:
    def executor(request: RecoveryRequest):
        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=lambda request, result: (
            "invalid"
        ),
    )

    result = controller.execute(
        _request()
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.INVALID_RESULT
    )


def test_policy_exception_is_fail_closed() -> None:
    class BrokenPolicy:
        def decide(
            self,
            request,
            attempt,
        ):
            raise RuntimeError(
                "policy exploded"
            )

    def executor(request: RecoveryRequest):
        return {
            "ok": False,
            "error": "temporary",
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=BrokenPolicy(),
    )

    result = controller.execute(
        _request()
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.NON_RETRYABLE
    )
    assert "policy exploded" in (
        result.error or ""
    )


def test_invalid_request_is_rejected() -> None:
    def executor(request):
        return {}

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
    )

    with pytest.raises(
        RecoveryControllerError,
        match="request must be",
    ):
        controller.execute(
            "invalid"
        )


def test_final_result_is_preserved() -> None:
    final_value = {
        "ok": True,
        "payload": "real-value",
    }

    def executor(request: RecoveryRequest):
        return final_value

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
    )

    result = controller.execute(
        _request()
    )

    assert result.result is final_value


def test_retry_preserves_final_result() -> None:
    first_value = {
        "ok": False,
        "payload": "first",
    }

    final_value = {
        "ok": True,
        "payload": "second",
    }

    def executor(request: RecoveryRequest):
        if request.attempt == 1:
            return first_value

        return final_value

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        _request()
    )

    assert result.success is True
    assert result.result is final_value


def test_success_never_retries() -> None:
    calls = 0

    def executor(request: RecoveryRequest):
        nonlocal calls
        calls += 1

        return {
            "ok": True,
        }

    controller = RecoveryController(
        executor,
        attempt_builder=_attempt_builder,
        policy=_retry_policy(),
    )

    result = controller.execute(
        _request(max_attempts=5)
    )

    assert result.success is True
    assert result.attempt_count == 1
    assert calls == 1
