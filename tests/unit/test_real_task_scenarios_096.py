"""End-to-end real task scenarios testing OSA combat readiness (0.9.6)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from osa.actions import (
    ActionConfirmationHandler,
    ActionDecision,
    ActionKind,
    ActionRequest,
    ActionRouter,
    ActionSafetyPipeline,
    DesktopActionSafetyPolicy,
    DesktopSafetyConfig,
)
from osa.actions.desktop import DesktopActionAdapter
from osa.core import Agent
from osa.core.agent_voice_action import TaskActionResolverVoiceAdapter
from osa.core.execution_context import ExecutionContext
from osa.desktop.automation import create_fake_desktop_automation
from osa.models import ChatMessage, ModelInterface, ModelRequest, ModelResponse, ToolCall, ToolDefinition
from osa.planning.planner import Plan, Planner, PlanStep
from osa.tasks.action import TaskAction, TaskActionResolver
from osa.tasks.manager import Task
from osa.tools import (
    FileExistsTool,
    FindFilesTool,
    ListDirectoryTool,
    PatchFileTool,
    ReadFileTool,
    SafeFilesystem,
    SafeShell,
    TerminalExecuteTool,
    ToolRegistry,
    WriteFileTool,
)


class MockConfirmationHandler(ActionConfirmationHandler):
    """Predictable confirmation handler."""

    def __init__(self, should_confirm: bool = True) -> None:
        self.should_confirm = should_confirm
        self.confirmed_requests: list[ActionRequest] = []

    def confirm(self, request: ActionRequest, reason: str) -> bool:
        self.confirmed_requests.append(request)
        return self.should_confirm


def test_scenario_workspace_code_and_shell_execution(tmp_path: Path) -> None:
    """Full lifecycle: create project structure, execute terminal test, patch code."""
    fs = SafeFilesystem(tmp_path)
    shell = SafeShell(tmp_path)
    registry = ToolRegistry()

    write_tool = WriteFileTool(fs)
    read_tool = ReadFileTool(fs)
    patch_tool = PatchFileTool(fs)
    find_tool = FindFilesTool(fs)
    exec_tool = TerminalExecuteTool(shell)

    registry.register(write_tool)
    registry.register(read_tool)
    registry.register(patch_tool)
    registry.register(find_tool)
    registry.register(exec_tool)

    # Step 1: Write initial Python script
    res_w = write_tool.execute({
        "path": "app.py",
        "content": "def run():\n    return 'v1.0'\n\nprint(run())\n",
    })
    assert res_w.success is True

    # Step 2: Execute python script via terminal
    res_sh = exec_tool.execute({"command": "python3 app.py"})
    assert res_sh.success is True
    assert "v1.0" in res_sh.output

    # Step 3: Patch the code to v2.0
    res_p = patch_tool.execute({
        "path": "app.py",
        "target": "v1.0",
        "replacement": "v2.0",
    })
    assert res_p.success is True

    # Step 4: Verify patch via terminal execution
    res_sh2 = exec_tool.execute({"command": "python3 app.py"})
    assert res_sh2.success is True
    assert "v2.0" in res_sh2.output

    # Step 5: Find python files
    res_f = find_tool.execute({"pattern": "*.py"})
    assert res_f.success is True
    assert "app.py" in res_f.output


def test_scenario_desktop_inspection_and_safety_pipeline() -> None:
    """Desktop window inspection and screenshot under DesktopActionSafetyPolicy."""
    fake_desktop = create_fake_desktop_automation()
    fake_desktop.add_window(title="Safari - Python Docs")
    fake_desktop.add_window(title="Terminal - OSA")

    adapter = DesktopActionAdapter(fake_desktop)
    router = ActionRouter()
    router.register(ActionKind.DESKTOP, adapter)

    policy = DesktopActionSafetyPolicy()
    handler = MockConfirmationHandler(should_confirm=True)

    pipeline = ActionSafetyPipeline(
        safety_policy=policy,
        confirmation_handler=handler,
        router=router,
    )

    # 1. Inspect windows (read-only, allowed without confirmation)
    req_win = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_windows",
        arguments={},
        request_id="sc-win-1",
    )
    res_win = pipeline.dispatch(req_win)
    assert res_win.success is True
    assert "Python Docs" in res_win.output
    assert len(handler.confirmed_requests) == 0  # No confirmation needed for read-only

    # 2. Take desktop screenshot (read-only, allowed)
    req_shot = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_screenshot",
        arguments={},
        request_id="sc-shot-1",
    )
    res_shot = pipeline.dispatch(req_shot)
    assert res_shot.success is True
    assert len(handler.confirmed_requests) == 0


def test_scenario_dangerous_action_confirmation_interception() -> None:
    """Destructive desktop action triggers confirmation prompt and halts if denied."""
    fake_desktop = create_fake_desktop_automation()
    app = fake_desktop.launch_application("Calculator")

    adapter = DesktopActionAdapter(fake_desktop)
    router = ActionRouter()
    router.register(ActionKind.DESKTOP, adapter)

    policy = DesktopActionSafetyPolicy()

    # User denies confirmation
    denying_handler = MockConfirmationHandler(should_confirm=False)
    pipeline = ActionSafetyPipeline(
        safety_policy=policy,
        confirmation_handler=denying_handler,
        router=router,
    )

    req_close = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_application",
        arguments={"name": app.application_id},
        request_id="sc-close-1",
    )
    res_close = pipeline.dispatch(req_close)

    assert res_close.success is False
    assert "Action confirmation was not granted." in (res_close.error or "")
    assert len(denying_handler.confirmed_requests) == 1


def test_scenario_voice_to_action_resolution() -> None:
    """Resolve a voice command into a validated ActionRequest and ExecutionContext."""
    mock_model = MagicMock(spec=ModelInterface)
    tool_call = ToolCall(
        id="voice_call_1",
        name="submit_action",
        arguments={"tool_name": "list_directory", "arguments": {"path": "."}},
    )
    mock_model.generate.return_value = ModelResponse(
        content="",
        model_name="mock-qwen",
        tool_calls=(tool_call,),
    )

    registry = ToolRegistry()
    registry.register(ListDirectoryTool(SafeFilesystem(Path.cwd())))

    resolver = TaskActionResolver(model=mock_model, tool_registry=registry)
    voice_adapter = TaskActionResolverVoiceAdapter(resolver)

    action_request = voice_adapter.resolve("List files in current directory")

    assert action_request.kind is ActionKind.TOOL
    assert action_request.name == "list_directory"
    assert action_request.metadata["source"] == "voice"
    assert "voice_task_id" in action_request.metadata

    # Canonical execution context projection
    ctx = ExecutionContext.from_action_request(action_request)
    assert ctx.source == "voice"
    assert ctx.task_id == action_request.metadata["voice_task_id"]
    assert ctx.round_number is None
