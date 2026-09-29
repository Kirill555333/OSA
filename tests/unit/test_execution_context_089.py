"""0.8.9/0.8.10 execution-context contract tests."""

from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.core.execution_context import (
    ExecutionContext,
    ExecutionContextError,
)


def test_execution_context_requires_request_id() -> None:
    with pytest.raises(
        ExecutionContextError,
        match="request_id is required",
    ):
        ExecutionContext(
            request_id=None,
        )


@pytest.mark.parametrize(
    "field",
    [
        "request_id",
        "run_id",
        "task_id",
        "source",
    ],
)
def test_execution_context_rejects_empty_identifiers(
    field: str,
) -> None:
    values = {
        "request_id": "request-1",
        "run_id": "run-1",
        "task_id": "task-1",
        "source": "agent",
    }

    values[field] = "   "

    with pytest.raises(
        ExecutionContextError,
        match=field,
    ):
        ExecutionContext(
            **values,
        )


def test_execution_context_normalizes_identifiers() -> None:
    context = ExecutionContext(
        request_id="  request-1  ",
        run_id="  run-1  ",
        task_id="  task-1  ",
        round_number=3,
        source="  voice  ",
    )

    assert context.request_id == "request-1"
    assert context.run_id == "run-1"
    assert context.task_id == "task-1"
    assert context.round_number == 3
    assert context.round == 3
    assert context.source == "voice"
    assert context.autonomous is True


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        -1,
        1.5,
        "1",
    ],
)
def test_execution_context_rejects_invalid_round(
    value,
) -> None:
    with pytest.raises(
        ExecutionContextError,
        match="round_number",
    ):
        ExecutionContext(
            request_id="request-1",
            round_number=value,
        )


def test_execution_context_allows_missing_optional_scope() -> None:
    context = ExecutionContext(
        request_id="request-1",
    )

    assert context.run_id is None
    assert context.task_id is None
    assert context.round_number is None
    assert context.source is None
    assert context.autonomous is False
    assert context.round is None


def test_execution_context_copies_and_freezes_metadata() -> None:
    source = {
        "execution_path": "agent.execute_tool",
        "value": 42,
    }

    context = ExecutionContext(
        request_id="request-1",
        metadata=source,
    )

    source["value"] = 99
    source["new"] = "should-not-appear"

    assert context.metadata["value"] == 42
    assert "new" not in context.metadata

    with pytest.raises(TypeError):
        context.metadata["value"] = 100


def test_execution_context_is_immutable() -> None:
    context = ExecutionContext(
        request_id="request-1",
    )

    with pytest.raises(
        AttributeError,
    ):
        context.request_id = "request-2"


def test_execution_context_extracts_action_request_metadata() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={
            "value": 42,
        },
        request_id="request-from-action",
        metadata={
            "source": "voice",
            "run_id": "run-from-request",
            "task_id": "task-from-request",
            "round": 4,
            "execution_path": "voice.adapter",
        },
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.request_id == "request-from-action"
    assert context.run_id == "run-from-request"
    assert context.task_id == "task-from-request"
    assert context.round_number == 4
    assert context.source == "voice"

    assert context.metadata == {
        "execution_path": "voice.adapter",
    }


def test_explicit_scope_overrides_request_metadata() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
        request_id="request-explicit",
        metadata={
            "run_id": "metadata-run",
            "task_id": "metadata-task",
            "round": 2,
            "source": "metadata-source",
        },
    )

    context = ExecutionContext.from_action_request(
        request,
        run_id="explicit-run",
        task_id="explicit-task",
        source="explicit-source",
    )

    assert context.request_id == "request-explicit"
    assert context.run_id == "explicit-run"
    assert context.task_id == "explicit-task"
    assert context.round_number == 2
    assert context.source == "explicit-source"


def test_execution_context_does_not_mutate_action_request_metadata() -> None:
    metadata = {
        "source": "agent",
        "round": 1,
        "custom": "value",
    }

    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
        request_id="request-immutable",
        metadata=metadata,
    )

    ExecutionContext.from_action_request(
        request
    )

    assert request.metadata == {
        "source": "agent",
        "round": 1,
        "custom": "value",
    }


def test_autonomous_context_is_canonical() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={
            "value": 42,
        },
        request_id="autonomous-context-1",
    )

    context = ExecutionContext.from_autonomous_action_request(
        request,
        run_id="run-089",
        task_id="task-089",
    )

    assert context.request_id == "autonomous-context-1"
    assert context.run_id == "run-089"
    assert context.task_id == "task-089"
    assert context.source == "autonomous"
    assert context.round_number is None
    assert context.autonomous is True
    assert context.metadata == {}


def test_autonomous_context_does_not_mutate_request() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={
            "value": 42,
        },
        request_id="autonomous-immutable-1",
        metadata={
            "custom": "value",
        },
    )

    context = ExecutionContext.from_autonomous_action_request(
        request,
        run_id="run-immutable",
        task_id="task-immutable",
    )

    assert context.metadata == {
        "custom": "value",
    }

    assert request.request_id == "autonomous-immutable-1"
    assert request.metadata == {
        "custom": "value",
    }


def test_autonomous_context_does_not_create_round() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
        request_id="autonomous-round-1",
        metadata={
            "round": 1,
        },
    )

    with pytest.raises(
        ExecutionContextError,
        match="round_number",
    ):
        ExecutionContext.from_autonomous_action_request(
            request,
            run_id="run-round",
            task_id="task-round",
        )


@pytest.mark.parametrize(
    "run_id, task_id",
    [
        ("", "task-1"),
        ("   ", "task-1"),
        ("run-1", ""),
        ("run-1", "   "),
    ],
)
def test_autonomous_context_requires_valid_scope(
    run_id: str,
    task_id: str,
) -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
        request_id="autonomous-validation-1",
    )

    with pytest.raises(
        ExecutionContextError,
    ):
        ExecutionContext.from_autonomous_action_request(
            request,
            run_id=run_id,
            task_id=task_id,
        )


def test_autonomous_context_keeps_action_arguments_out_of_context() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={
            "secret": "must-not-be-context",
            "value": 42,
        },
        request_id="autonomous-privacy-1",
    )

    context = ExecutionContext.from_autonomous_action_request(
        request,
        run_id="run-privacy",
        task_id="task-privacy",
    )

    assert context.metadata == {}
    assert "secret" not in context.metadata
    assert "value" not in context.metadata
    assert "must-not-be-context" not in str(context)
