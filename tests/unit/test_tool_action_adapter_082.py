from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.actions.tool import ToolRegistryActionAdapter
from osa.tools import ToolRegistry, ToolResult


class DemoTool:
    name = "demo"
    description = "Demo tool"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output=f"value={arguments['value']}",
            metadata={"source": "demo"},
        )


class FailingTool:
    name = "failing"
    description = "Failing tool"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    def execute(self, arguments):
        return ToolResult(
            success=False,
            error="demo failure",
            metadata={"reason": "test"},
        )


@pytest.fixture()
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(DemoTool())
    registry.register(FailingTool())
    return registry


def test_tool_registry_adapter_returns_successful_action_result(
    registry: ToolRegistry,
) -> None:
    adapter = ToolRegistryActionAdapter(registry)
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={"value": 42},
        request_id="call_demo",
    )

    result = adapter.execute(request)

    assert result.request_id == "call_demo"
    assert result.success is True
    assert result.output == "value=42"
    assert result.data["tool_result"]["metadata"] == {
        "source": "demo",
    }
    assert result.metadata["action_kind"] == "tool"
    assert result.metadata["action_name"] == "demo"


def test_tool_registry_adapter_preserves_failed_tool_result(
    registry: ToolRegistry,
) -> None:
    adapter = ToolRegistryActionAdapter(registry)
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="failing",
        arguments={},
        request_id="call_failed",
    )

    result = adapter.execute(request)

    assert result.request_id == "call_failed"
    assert result.success is False
    assert result.error == "demo failure"
    assert result.data["tool_result"]["error"] == "demo failure"
    assert result.metadata["tool_failed"] is True


def test_tool_registry_adapter_reports_unknown_tool(
    registry: ToolRegistry,
) -> None:
    adapter = ToolRegistryActionAdapter(registry)
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="missing",
        arguments={},
        request_id="call_missing",
    )

    result = adapter.execute(request)

    assert result.success is False
    assert "missing" in (result.error or "")
    assert result.metadata["tool_error"] == "tool_not_registered"


def test_tool_registry_adapter_rejects_non_tool_request(
    registry: ToolRegistry,
) -> None:
    adapter = ToolRegistryActionAdapter(registry)
    request = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_launch_application",
        arguments={"command": "Google Chrome"},
        request_id="call_desktop",
    )

    with pytest.raises(
        RuntimeError,
        match="only supports TOOL",
    ):
        adapter.execute(request)
