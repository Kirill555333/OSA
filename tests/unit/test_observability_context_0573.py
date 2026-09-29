from __future__ import annotations

import pytest

from osa.observability import (
    EMPTY_OBSERVABILITY_CONTEXT,
    ObservabilityContext,
    ObservabilityContextError,
)


def test_empty_context_has_no_identifiers() -> None:
    context = ObservabilityContext()

    assert context.run_id is None
    assert context.task_id is None
    assert context.request_id is None
    assert context.empty is True
    assert context.autonomous is False
    assert context.action is False


def test_global_empty_context_is_available() -> None:
    assert EMPTY_OBSERVABILITY_CONTEXT.empty is True
    assert EMPTY_OBSERVABILITY_CONTEXT.to_dict() == {
        "run_id": None,
        "task_id": None,
        "request_id": None,
    }


def test_identifiers_are_normalized() -> None:
    context = ObservabilityContext(
        run_id="  run-1  ",
        task_id="  task-1  ",
        request_id="  request-1  ",
    )

    assert context.run_id == "run-1"
    assert context.task_id == "task-1"
    assert context.request_id == "request-1"
    assert context.autonomous is True
    assert context.action is True


def test_partial_context_is_valid() -> None:
    context = ObservabilityContext(
        run_id="run-1",
    )

    assert context.run_id == "run-1"
    assert context.task_id is None
    assert context.request_id is None
    assert context.autonomous is True
    assert context.action is False


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("run_id", ""),
        ("run_id", "   "),
        ("task_id", ""),
        ("task_id", "   "),
        ("request_id", ""),
        ("request_id", "   "),
    ],
)
def test_empty_identifiers_are_rejected(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(
        ObservabilityContextError
    ):
        ObservabilityContext(
            **{
                field_name: value,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("run_id", 123),
        ("task_id", 123),
        ("request_id", 123),
    ],
)
def test_non_string_identifiers_are_rejected(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(
        ObservabilityContextError
    ):
        ObservabilityContext(
            **{
                field_name: value,
            }
        )


def test_with_run_preserves_other_identifiers() -> None:
    original = ObservabilityContext(
        task_id="task-1",
        request_id="request-1",
    )

    updated = original.with_run(
        "run-1"
    )

    assert original.run_id is None
    assert original.task_id == "task-1"
    assert original.request_id == "request-1"

    assert updated.run_id == "run-1"
    assert updated.task_id == "task-1"
    assert updated.request_id == "request-1"


def test_with_task_preserves_run_and_request() -> None:
    original = ObservabilityContext(
        run_id="run-1",
        request_id="request-1",
    )

    updated = original.with_task(
        "task-1"
    )

    assert updated.run_id == "run-1"
    assert updated.task_id == "task-1"
    assert updated.request_id == "request-1"


def test_with_request_preserves_run_and_task() -> None:
    original = ObservabilityContext(
        run_id="run-1",
        task_id="task-1",
    )

    updated = original.with_request(
        "request-1"
    )

    assert updated.run_id == "run-1"
    assert updated.task_id == "task-1"
    assert updated.request_id == "request-1"


def test_clear_request_only_removes_request() -> None:
    context = ObservabilityContext(
        run_id="run-1",
        task_id="task-1",
        request_id="request-1",
    )

    updated = context.clear_request()

    assert updated.run_id == "run-1"
    assert updated.task_id == "task-1"
    assert updated.request_id is None


def test_clear_task_removes_task_and_request() -> None:
    context = ObservabilityContext(
        run_id="run-1",
        task_id="task-1",
        request_id="request-1",
    )

    updated = context.clear_task()

    assert updated.run_id == "run-1"
    assert updated.task_id is None
    assert updated.request_id is None


def test_clear_run_removes_all_identifiers() -> None:
    context = ObservabilityContext(
        run_id="run-1",
        task_id="task-1",
        request_id="request-1",
    )

    updated = context.clear_run()

    assert updated.empty is True
    assert updated.to_dict() == {
        "run_id": None,
        "task_id": None,
        "request_id": None,
    }


def test_to_dict_preserves_identifier_values() -> None:
    context = ObservabilityContext(
        run_id="run-1",
        task_id="task-1",
        request_id="request-1",
    )

    assert context.to_dict() == {
        "run_id": "run-1",
        "task_id": "task-1",
        "request_id": "request-1",
    }


def test_context_is_immutable() -> None:
    context = ObservabilityContext(
        run_id="run-1",
    )

    with pytest.raises(
        AttributeError
    ):
        context.run_id = "run-2"  # type: ignore[misc]
