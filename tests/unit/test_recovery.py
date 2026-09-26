import pytest

from osa.models import (
    ModelConnectionError,
    ModelError,
    ModelResponseError,
)
from osa.recovery import (
    ErrorRecovery,
    RecoveryConfig,
)


def test_recovery_config_rejects_negative_retries() -> None:
    with pytest.raises(ValueError):
        RecoveryConfig(
            max_model_retries=-1,
        )


def test_recovery_config_rejects_negative_delay() -> None:
    with pytest.raises(ValueError):
        RecoveryConfig(
            model_retry_delay_seconds=-1,
        )


def test_model_connection_failure_is_retried() -> None:
    recovery = ErrorRecovery(
        RecoveryConfig(
            max_model_retries=1,
            model_retry_delay_seconds=0,
        )
    )

    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise ModelConnectionError(
                "temporary connection failure"
            )

        return "success"

    result, count = recovery.run_model(
        operation
    )

    assert result == "success"
    assert count == 2
    assert attempts == 2


def test_model_connection_failure_stops_after_retry_limit() -> None:
    recovery = ErrorRecovery(
        RecoveryConfig(
            max_model_retries=1,
            model_retry_delay_seconds=0,
        )
    )

    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ModelConnectionError(
            "connection unavailable"
        )

    with pytest.raises(
        ModelConnectionError
    ):
        recovery.run_model(
            operation
        )

    assert attempts == 2


def test_non_connection_model_error_is_not_retried() -> None:
    recovery = ErrorRecovery(
        RecoveryConfig(
            max_model_retries=3,
            model_retry_delay_seconds=0,
        )
    )

    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ModelResponseError(
            "invalid model response"
        )

    with pytest.raises(
        ModelResponseError
    ):
        recovery.run_model(
            operation
        )

    assert attempts == 1


def test_tool_exception_message_is_model_readable() -> None:
    message = ErrorRecovery.tool_exception_message(
        "write_file",
        RuntimeError("disk failure"),
    )

    assert "write_file" in message
    assert "disk failure" in message
