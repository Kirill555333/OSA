from __future__ import annotations

import pytest

from osa.observability import (
    InMemoryObservabilityLogger,
    ObservabilityEvent,
    ObservabilityRedactionError,
    ObservabilityRedactor,
    RedactingObservabilityLogger,
)


def _event(
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> ObservabilityEvent:
    return ObservabilityEvent(
        event="test.event",
        source="test",
        status="failed",
        error=error,
        metadata=metadata,
    )


def test_sensitive_metadata_keys_are_redacted() -> None:
    source_logger = InMemoryObservabilityLogger()
    logger = RedactingObservabilityLogger(
        source_logger
    )

    event = _event(
        metadata={
            "password": "super-secret",
            "token": "token-value",
            "safe_value": "visible",
        }
    )

    logger.log(event)

    stored = source_logger.events()[0]

    assert stored.metadata["password"] == "[REDACTED]"
    assert stored.metadata["token"] == "[REDACTED]"
    assert stored.metadata["safe_value"] == "visible"


def test_nested_sensitive_values_are_redacted() -> None:
    source_logger = InMemoryObservabilityLogger()
    logger = RedactingObservabilityLogger(
        source_logger
    )

    event = _event(
        metadata={
            "request": {
                "authorization": "Bearer SECRET",
                "headers": [
                    {
                        "api_key": "KEY_SECRET",
                    }
                ],
            }
        }
    )

    logger.log(event)

    stored = source_logger.events()[0]

    assert stored.metadata["request"]["authorization"] == (
        "[REDACTED]"
    )
    assert stored.metadata["request"]["headers"][0][
        "api_key"
    ] == "[REDACTED]"


def test_error_is_truncated_but_not_secret_scanned() -> None:
    source_logger = InMemoryObservabilityLogger()
    redactor = ObservabilityRedactor(
        max_string_length=10
    )
    logger = RedactingObservabilityLogger(
        source_logger,
        redactor=redactor,
    )

    logger.log(
        _event(
            metadata={},
            error="12345678901234567890",
        )
    )

    stored = source_logger.events()[0]

    assert stored.error == "1234567890…[TRUNCATED]"


def test_custom_sensitive_key_and_replacement_work() -> None:
    source_logger = InMemoryObservabilityLogger()

    redactor = ObservabilityRedactor(
        sensitive_keys={"private"},
        replacement="<hidden>",
    )

    logger = RedactingObservabilityLogger(
        source_logger,
        redactor=redactor,
    )

    logger.log(
        _event(
            metadata={
                "private": "secret",
                "normal": "value",
            }
        )
    )

    stored = source_logger.events()[0]

    assert stored.metadata["private"] == "<hidden>"
    assert stored.metadata["normal"] == "value"


def test_original_event_is_not_mutated() -> None:
    metadata = {
        "token": "secret",
        "nested": {
            "password": "another-secret",
        },
    }

    event = _event(
        metadata=metadata
    )

    redactor = ObservabilityRedactor()
    sanitized = redactor.redact_event(event)

    assert event.metadata["token"] == "secret"
    assert event.metadata["nested"]["password"] == (
        "another-secret"
    )

    assert sanitized.metadata["token"] == "[REDACTED]"
    assert sanitized.metadata["nested"]["password"] == (
        "[REDACTED]"
    )


def test_original_nested_collections_are_not_mutated() -> None:
    original = {
        "items": [
            {
                "token": "secret",
            },
            {
                "safe": "value",
            },
        ]
    }

    redactor = ObservabilityRedactor()
    result = redactor.redact_mapping(original)

    assert original["items"][0]["token"] == "secret"
    assert result["items"][0]["token"] == "[REDACTED]"
    assert result["items"][1]["safe"] == "value"


def test_string_logging_interface_is_redacted() -> None:
    source_logger = InMemoryObservabilityLogger()

    logger = RedactingObservabilityLogger(
        source_logger
    )

    logger.log(
        "legacy.event",
        password="secret-password",
        token="secret-token",
        visible="hello",
    )

    stored = source_logger.events()[0]

    assert stored.metadata["password"] == "[REDACTED]"
    assert stored.metadata["token"] == "[REDACTED]"
    assert stored.metadata["visible"] == "hello"


def test_logger_and_redactor_properties_are_preserved() -> None:
    source_logger = InMemoryObservabilityLogger()
    redactor = ObservabilityRedactor()

    logger = RedactingObservabilityLogger(
        source_logger,
        redactor=redactor,
    )

    assert logger.logger is source_logger
    assert logger.redactor is redactor


def test_invalid_sensitive_keys_are_rejected() -> None:
    with pytest.raises(
        ObservabilityRedactionError,
        match="must contain strings",
    ):
        ObservabilityRedactor(
            sensitive_keys={"token", 123}
        )


def test_invalid_string_length_is_rejected() -> None:
    with pytest.raises(
        ObservabilityRedactionError,
        match="at least 1",
    ):
        ObservabilityRedactor(
            max_string_length=0
        )


def test_invalid_event_is_rejected() -> None:
    redactor = ObservabilityRedactor()

    with pytest.raises(
        ObservabilityRedactionError,
        match="event must be",
    ):
        redactor.redact_event("not-an-event")


def test_none_logger_is_rejected() -> None:
    with pytest.raises(
        ObservabilityRedactionError,
        match="logger is required",
    ):
        RedactingObservabilityLogger(None)
