from __future__ import annotations

import pytest

from osa.actions import (
    ActionContractError,
    ActionKind,
    ActionRequest,
    ActionResult,
)


def test_action_request_normalizes_and_freezes_arguments() -> None:
    arguments = {
        "url": "https://example.com",
        "timeout": 5,
    }

    request = ActionRequest(
        kind="browser",
        name="browser_open",
        arguments=arguments,
    )

    assert request.kind is ActionKind.BROWSER
    assert request.backend == "browser"
    assert request.name == "browser_open"
    assert request.arguments["url"] == "https://example.com"
    assert request.request_id
    assert isinstance(request.metadata, dict) is False


def test_action_request_rejects_empty_name() -> None:
    with pytest.raises(
        ActionContractError,
        match="name cannot be empty",
    ):
        ActionRequest(
            kind=ActionKind.TOOL,
            name="   ",
        )


def test_action_request_rejects_invalid_kind() -> None:
    with pytest.raises(
        ActionContractError,
        match="kind must be a valid ActionKind",
    ):
        ActionRequest(
            kind="unknown",
            name="test",
        )


def test_action_request_rejects_non_string_argument_keys() -> None:
    with pytest.raises(
        ActionContractError,
        match="arguments keys must be strings",
    ):
        ActionRequest(
            kind=ActionKind.DESKTOP,
            name="click",
            arguments={1: "invalid"},
        )


def test_action_result_success_factory() -> None:
    result = ActionResult.succeeded(
        request_id="request-1",
        output="done",
        data={"tab_id": "tab-1"},
    )

    assert result.success is True
    assert result.output == "done"
    assert result.error is None
    assert result.data["tab_id"] == "tab-1"


def test_action_result_failure_factory() -> None:
    result = ActionResult.failed(
        request_id="request-2",
        error="backend failure",
    )

    assert result.success is False
    assert result.error == "backend failure"


def test_action_result_rejects_success_with_error() -> None:
    with pytest.raises(
        ActionContractError,
        match="successful action results cannot contain an error",
    ):
        ActionResult(
            request_id="request-3",
            success=True,
            output="done",
            error="unexpected",
        )


def test_action_result_rejects_failure_without_error() -> None:
    with pytest.raises(
        ActionContractError,
        match="failed action results must contain an error",
    ):
        ActionResult(
            request_id="request-4",
            success=False,
        )


def test_action_result_rejects_empty_request_id() -> None:
    with pytest.raises(
        ActionContractError,
        match="request_id cannot be empty",
    ):
        ActionResult(
            request_id="   ",
            success=True,
        )
