from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osa.actions import (
    BROWSER_ACTION_NAMES,
    ActionKind,
    ActionRequest,
    BrowserActionAdapter,
    BrowserActionAdapterError,
)


@dataclass(frozen=True)
class FakeToolResult:
    success: bool
    output: str = ""
    error: str | None = None


class FakeBrowserTool:
    def __init__(
        self,
        name: str,
        result: FakeToolResult | None = None,
    ) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self.result = result or FakeToolResult(
            success=True,
            output=f"executed:{name}",
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> FakeToolResult:
        self.calls.append(arguments)
        return self.result


class RaisingBrowserTool:
    name = "browser_click"

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> FakeToolResult:
        raise RuntimeError("simulated browser failure")


class InvalidResultBrowserTool:
    name = "browser_read"

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> object:
        return object()


def make_request(
    name: str = "browser_open",
    arguments: dict[str, Any] | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.BROWSER,
        name=name,
        arguments=arguments or {},
    )


def test_adapter_supports_all_existing_browser_actions() -> None:
    adapter = BrowserActionAdapter(
        FakeBrowserTool(name)
        for name in BROWSER_ACTION_NAMES
    )

    assert adapter.supported_actions == BROWSER_ACTION_NAMES


def test_adapter_executes_registered_browser_tool() -> None:
    tool = FakeBrowserTool("browser_open")

    adapter = BrowserActionAdapter([tool])

    request = make_request(
        arguments={
            "url": "https://example.com",
        }
    )

    result = adapter.execute(request)

    assert result.success is True
    assert result.request_id == request.request_id
    assert result.output == "executed:browser_open"
    assert tool.calls == [
        {
            "url": "https://example.com",
        }
    ]


def test_adapter_preserves_tool_failure() -> None:
    tool = FakeBrowserTool(
        "browser_click",
        FakeToolResult(
            success=False,
            error="element not found",
        ),
    )

    adapter = BrowserActionAdapter([tool])

    result = adapter.execute(
        make_request("browser_click")
    )

    assert result.success is False
    assert result.error == "element not found"
    assert result.request_id
    assert result.metadata["adapter"] == "browser"


def test_adapter_converts_backend_exception_to_failed_action() -> None:
    adapter = BrowserActionAdapter(
        [RaisingBrowserTool()]
    )

    request = make_request("browser_click")
    result = adapter.execute(request)

    assert result.success is False
    assert result.request_id == request.request_id
    assert result.error == (
        "Browser action failed: simulated browser failure"
    )
    assert result.metadata["adapter_error"] == "tool_exception"
    assert result.metadata["exception_type"] == "RuntimeError"


def test_adapter_rejects_wrong_action_kind() -> None:
    adapter = BrowserActionAdapter(
        [FakeBrowserTool("browser_open")]
    )

    request = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="browser_open",
    )

    result = adapter.execute(request)

    assert result.success is False
    assert result.error == (
        "BrowserActionAdapter received 'desktop' action."
    )
    assert result.metadata["adapter_error"] == "wrong_action_kind"


def test_adapter_reports_unregistered_action() -> None:
    adapter = BrowserActionAdapter([])

    request = make_request("browser_observe")
    result = adapter.execute(request)

    assert result.success is False
    assert result.error == (
        "Unsupported browser action 'browser_observe'."
    )
    assert result.metadata["adapter_error"] == "action_not_registered"


def test_adapter_rejects_unsupported_tool_name() -> None:
    try:
        BrowserActionAdapter(
            [FakeBrowserTool("unknown_browser_action")]
        )
    except BrowserActionAdapterError as exc:
        assert str(exc) == (
            "Unsupported browser action 'unknown_browser_action'."
        )
    else:
        raise AssertionError(
            "BrowserActionAdapterError was not raised."
        )


def test_adapter_rejects_tool_without_name() -> None:
    class NamelessTool:
        def execute(
            self,
            arguments: dict[str, Any],
        ) -> FakeToolResult:
            return FakeToolResult(True, "done")

    try:
        BrowserActionAdapter([NamelessTool()])
    except BrowserActionAdapterError as exc:
        assert str(exc) == (
            "Browser action tool must expose a non-empty name."
        )
    else:
        raise AssertionError(
            "BrowserActionAdapterError was not raised."
        )


def test_adapter_rejects_tool_without_execute() -> None:
    class InvalidTool:
        name = "browser_open"

    try:
        BrowserActionAdapter([InvalidTool()])
    except BrowserActionAdapterError as exc:
        assert str(exc) == (
            "Browser action 'browser_open' must expose execute()."
        )
    else:
        raise AssertionError(
            "BrowserActionAdapterError was not raised."
        )


def test_adapter_handles_invalid_tool_result() -> None:
    adapter = BrowserActionAdapter(
        [InvalidResultBrowserTool()]
    )

    request = make_request("browser_read")
    result = adapter.execute(request)

    assert result.success is False
    assert result.error == (
        "Browser action returned an invalid tool result."
    )
    assert result.metadata["adapter_error"] == (
        "invalid_tool_result"
       )


def test_adapter_handles_non_string_tool_output() -> None:
    class BadOutputTool:
        name = "browser_read"

        def execute(
            self,
            arguments: dict[str, Any],
        ) -> Any:
            return FakeToolResult(
                success=True,
                output=123,  # type: ignore[arg-type]
            )

    adapter = BrowserActionAdapter(
        [BadOutputTool()]
    )

    result = adapter.execute(
        make_request("browser_read")
    )

    assert result.success is False
    assert result.error == (
        "Browser action returned non-string output."
    )
    assert result.metadata["adapter_error"] == (
        "invalid_tool_output"
    )


def test_adapter_handles_failed_tool_without_error_message() -> None:
    tool = FakeBrowserTool(
        "browser_wait",
        FakeToolResult(
            success=False,
            error=None,
        ),
    )

    adapter = BrowserActionAdapter([tool])

    result = adapter.execute(
        make_request("browser_wait")
    )

    assert result.success is False
    assert result.error == (
        "Browser action failed without an error message."
    )
    assert result.metadata["adapter_error"] == (
        "missing_tool_error"
    )


def test_adapter_tool_lookup_and_unregister() -> None:
    tool = FakeBrowserTool("browser_open")
    adapter = BrowserActionAdapter([tool])

    assert adapter.tool_for("browser_open") is tool

    adapter.unregister("browser_open")

    assert adapter.tool_for("browser_open") is None
    assert adapter.supported_actions == ()


def test_adapter_dispatch_is_execute_alias() -> None:
    tool = FakeBrowserTool("browser_observe")
    adapter = BrowserActionAdapter([tool])

    request = make_request("browser_observe")

    direct = adapter.execute(request)
    via_dispatch = adapter.dispatch(request)

    assert direct.success is True
    assert via_dispatch.success is True
    assert direct.request_id == via_dispatch.request_id
