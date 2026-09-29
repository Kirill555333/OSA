"""0.8.9.8 ExecutionContext hardening and fail-closed tests."""

from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.core.execution_context import (
    ExecutionContext,
    ExecutionContextError,
)


def _request(
    *,
    request_id: str = "hardening-089",
    metadata: dict | None = None,
    arguments: dict | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={} if arguments is None else arguments,
        request_id=request_id,
        metadata={} if metadata is None else metadata,
    )


def test_explicit_context_wins_without_mutating_request() -> None:
    request = _request(
        metadata={
            "run_id": "metadata-run",
            "task_id": "metadata-task",
            "round": 4,
            "source": "agent",
            "custom": "kept",
        }
    )

    context = ExecutionContext.from_action_request(
        request,
        run_id="explicit-run",
        task_id="explicit-task",
    )

    assert context.run_id == "explicit-run"
    assert context.task_id == "explicit-task"
    assert context.round_number == 4
    assert context.source == "agent"
    assert context.metadata == {
        "custom": "kept",
    }

    assert request.metadata == {
        "run_id": "metadata-run",
        "task_id": "metadata-task",
        "round": 4,
        "source": "agent",
        "custom": "kept",
    }


def test_voice_task_id_is_consumed_into_task_id() -> None:
    request = _request(
        metadata={
            "source": "voice",
            "voice_task_id": "voice-task-089",
            "custom": "value",
        }
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.task_id == "voice-task-089"
    assert context.source == "voice"
    assert context.metadata == {
        "custom": "value",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "run_id": "",
        },
        {
            "run_id": "   ",
        },
        {
            "task_id": "",
        },
        {
            "task_id": "   ",
        },
        {
            "voice_task_id": "",
        },
        {
            "voice_task_id": "   ",
        },
        {
            "source": "",
        },
        {
            "source": "   ",
        },
    ],
)
def test_invalid_identifier_metadata_fails_closed(
    metadata: dict,
) -> None:
    with pytest.raises(
        ExecutionContextError,
    ):
        ExecutionContext.from_action_request(
            _request(
                metadata=metadata
            )
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "round": True,
        },
        {
            "round": False,
        },
        {
            "round": 0,
        },
        {
            "round": -1,
        },
        {
            "round": 1.5,
        },
        {
            "round": "1",
        },
    ],
)
def test_invalid_round_metadata_fails_closed(
    metadata: dict,
) -> None:
    with pytest.raises(
        ExecutionContextError,
        match="round_number",
    ):
        ExecutionContext.from_action_request(
            _request(
                metadata=metadata
            )
        )


def test_unknown_metadata_survives_as_residual_metadata() -> None:
    context = ExecutionContext.from_action_request(
        _request(
            metadata={
                "custom": "value",
                "trace": 42,
            }
        )
    )

    assert context.metadata == {
        "custom": "value",
        "trace": 42,
    }


def test_action_arguments_are_never_in_context() -> None:
    request = _request(
        arguments={
            "secret": "must-never-appear",
            "value": 42,
        },
        metadata={
            "source": "agent",
        },
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.metadata == {}
    assert "secret" not in str(context)
    assert "must-never-appear" not in str(context)
    assert "value" not in context.metadata


def test_request_identity_always_remains_authoritative() -> None:
    request = _request(
        request_id="authoritative-request",
        metadata={
            "request_id": "spoofed-request",
            "source": "agent",
        },
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.request_id == (
        "authoritative-request"
    )

    assert context.metadata == {
        "request_id": "spoofed-request",
    }


def test_explicit_none_does_not_erase_request_metadata() -> None:
    request = _request(
        metadata={
            "run_id": "metadata-run",
            "task_id": "metadata-task",
        }
    )

    context = ExecutionContext.from_action_request(
        request,
        run_id=None,
        task_id=None,
    )

    assert context.run_id == "metadata-run"
    assert context.task_id == "metadata-task"
