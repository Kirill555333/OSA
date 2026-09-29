from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from osa.recovery_context import (
    EMPTY_RECOVERY_CONTEXT,
    RecoveryContext,
    RecoveryContextError,
)
from osa.recovery_contracts import (
    RecoveryRequest,
)


def _request() -> RecoveryRequest:
    return RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
        action_kind="tool",
        action_name="demo_tool",
        attempt=1,
        max_attempts=3,
        metadata={
            "token": "secret",
            "scope": "demo",
        },
    )


def test_context_is_created_from_recovery_request():
    context = RecoveryContext.from_request(
        _request()
    )

    assert context.run_id == "run-1"
    assert context.task_id == "task-1"
    assert context.request_id == "req-1"
    assert context.action_kind == "tool"
    assert context.action_name == "demo_tool"
    assert context.attempt == 1
    assert context.max_attempts == 3

    assert context.metadata["scope"] == "demo"
    assert context.is_autonomous is True
    assert context.is_action is True
    assert context.correlation_id == "req-1"
    assert context.exhausted is False


def test_context_does_not_expose_raw_action_arguments():
    context = RecoveryContext.from_request(
        _request()
    )

    assert not hasattr(
        context,
        "arguments",
    )

    assert not hasattr(
        context,
        "output",
    )

    assert "secret" not in repr(
        context
    )


def test_context_metadata_is_immutable():
    context = RecoveryContext(
        request_id="req-1",
        metadata={
            "scope": "demo",
        },
    )

    with pytest.raises(
        TypeError,
    ):
        context.metadata["scope"] = "changed"


def test_context_itself_is_immutable():
    context = RecoveryContext(
        request_id="req-1",
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        context.attempt = 2


def test_next_attempt_returns_new_context_and_preserves_correlation():
    context = RecoveryContext(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
        action_kind="browser",
        action_name="browser_click",
        attempt=1,
        max_attempts=3,
        metadata={
            "scope": "demo",
        },
    )

    next_context = context.next_attempt()

    assert context.attempt == 1
    assert next_context.attempt == 2

    assert next_context.max_attempts == 3

    assert next_context.run_id == context.run_id
    assert next_context.task_id == context.task_id
    assert next_context.request_id == context.request_id
    assert next_context.action_kind == context.action_kind
    assert next_context.action_name == context.action_name
    assert dict(next_context.metadata) == dict(
        context.metadata
    )


def test_explicit_next_attempt_is_supported():
    context = RecoveryContext(
        request_id="req-1",
        attempt=1,
        max_attempts=4,
    )

    next_context = context.next_attempt(
        attempt=3
    )

    assert next_context.attempt == 3
    assert context.attempt == 1


def test_next_attempt_rejects_non_increasing_attempt():
    context = RecoveryContext(
        request_id="req-1",
        attempt=2,
        max_attempts=4,
    )

    with pytest.raises(
        RecoveryContextError,
        match="greater than",
    ):
        context.next_attempt(
            attempt=2
        )

    with pytest.raises(
        RecoveryContextError,
        match="greater than",
    ):
        context.next_attempt(
            attempt=1
        )


def test_next_attempt_rejects_attempt_beyond_budget():
    context = RecoveryContext(
        request_id="req-1",
        attempt=2,
        max_attempts=2,
    )

    assert context.exhausted is True

    with pytest.raises(
        RecoveryContextError,
        match="cannot exceed",
    ):
        context.next_attempt()


def test_with_metadata_returns_new_context():
    context = RecoveryContext(
        request_id="req-1",
        metadata={
            "scope": "demo",
        },
    )

    updated = context.with_metadata(
        attempt_reason="temporary_failure"
    )

    assert dict(
        context.metadata
    ) == {
        "scope": "demo",
    }

    assert dict(
        updated.metadata
    ) == {
        "scope": "demo",
        "attempt_reason": "temporary_failure",
    }


def test_without_metadata_returns_new_context():
    context = RecoveryContext(
        request_id="req-1",
        metadata={
            "scope": "demo",
            "temporary": True,
        },
    )

    cleaned = context.without_metadata(
        "temporary"
    )

    assert dict(
        cleaned.metadata
    ) == {
        "scope": "demo",
    }

    assert dict(
        context.metadata
    ) == {
        "scope": "demo",
        "temporary": True,
    }


def test_context_requires_at_least_one_identifier():
    with pytest.raises(
        RecoveryContextError,
        match="identifier",
    ):
        RecoveryContext()


def test_context_validates_attempt_bounds():
    with pytest.raises(
        RecoveryContextError,
        match="at least 1",
    ):
        RecoveryContext(
            request_id="req-1",
            attempt=0,
            max_attempts=1,
        )

    with pytest.raises(
        RecoveryContextError,
        match="at least 1",
    ):
        RecoveryContext(
            request_id="req-1",
            attempt=1,
            max_attempts=0,
        )

    with pytest.raises(
        RecoveryContextError,
        match="cannot exceed",
    ):
        RecoveryContext(
            request_id="req-1",
            attempt=2,
            max_attempts=1,
        )


def test_context_rejects_invalid_identifiers():
    with pytest.raises(
        RecoveryContextError,
        match="run_id",
    ):
        RecoveryContext(
            run_id="   ",
        )

    with pytest.raises(
        RecoveryContextError,
        match="task_id",
    ):
        RecoveryContext(
            task_id=123,
        )

    with pytest.raises(
        RecoveryContextError,
        match="request_id",
    ):
        RecoveryContext(
            request_id="   ",
        )


def test_context_rejects_invalid_action_name():
    with pytest.raises(
        RecoveryContextError,
        match="action_name",
    ):
        RecoveryContext(
            request_id="req-1",
            action_name="   ",
        )

    with pytest.raises(
        RecoveryContextError,
        match="action_name",
    ):
        RecoveryContext(
            request_id="req-1",
            action_name=123,
        )


def test_context_rejects_invalid_metadata():
    with pytest.raises(
        RecoveryContextError,
        match="metadata",
    ):
        RecoveryContext(
            request_id="req-1",
            metadata=[],
        )


def test_from_request_rejects_invalid_input():
    with pytest.raises(
        TypeError,
        match="RecoveryRequest",
    ):
        RecoveryContext.from_request(
            object()
        )


def test_empty_context_is_a_valid_immutable_sentinel():
    assert EMPTY_RECOVERY_CONTEXT.request_id == (
        "empty-context"
    )
    assert EMPTY_RECOVERY_CONTEXT.attempt == 1
    assert EMPTY_RECOVERY_CONTEXT.max_attempts == 1
    assert EMPTY_RECOVERY_CONTEXT.exhausted is True


def test_correlation_id_prefers_request_then_task_then_run():
    request_context = RecoveryContext(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
    )

    task_context = RecoveryContext(
        run_id="run-1",
        task_id="task-1",
    )

    run_context = RecoveryContext(
        run_id="run-1",
    )

    assert request_context.correlation_id == "req-1"
    assert task_context.correlation_id == "task-1"
    assert run_context.correlation_id == "run-1"
