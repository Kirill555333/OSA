"""Unit tests for InteractiveShell and confirmation prompts (0.9.4)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock
import pytest

from osa.core import Agent
from osa.core.modes import AgentMode
from osa.models import ModelInterface, ModelResponse, ModelStreamEvent
from osa.shell import (
    InteractiveShell,
    create_interactive_confirmation_handler,
    terminal_confirmation_prompt,
)


def _mock_agent(reply: str = "Assistant response") -> Agent:
    mock_model = MagicMock(spec=ModelInterface)
    mock_model.model_name = "test-qwen"
    mock_model.health_check.return_value = True
    mock_model.generate.return_value = ModelResponse(content=reply, model_name="test-qwen")
    mock_model.generate_stream_events.return_value = iter([ModelStreamEvent(content=reply)])
    return Agent(model=mock_model, system_prompt="System")


def test_terminal_confirmation_prompt_yes() -> None:
    inp = io.StringIO("y\n")
    out = io.StringIO()

    result = terminal_confirmation_prompt("Delete temporary file?", input_stream=inp, output_stream=out)
    assert result is True
    assert "[ACTION CONFIRMATION REQUIRED]" in out.getvalue()
    assert "Delete temporary file?" in out.getvalue()


def test_terminal_confirmation_prompt_no() -> None:
    inp = io.StringIO("n\n")
    out = io.StringIO()

    result = terminal_confirmation_prompt("Format hard drive?", input_stream=inp, output_stream=out)
    assert result is False


def test_terminal_confirmation_prompt_eof_or_cancel() -> None:
    inp = io.StringIO("")  # EOF
    out = io.StringIO()

    result = terminal_confirmation_prompt("Restart system?", input_stream=inp, output_stream=out)
    assert result is False


def test_shell_banner_and_status_command() -> None:
    agent = _mock_agent()
    inp = io.StringIO()
    out = io.StringIO()

    shell = InteractiveShell(agent, input_stream=inp, output_stream=out)
    shell.print_banner()

    banner_text = out.getvalue()
    assert "OSA — PERSONAL AI AGENT" in banner_text
    assert "test-qwen" in banner_text

    # Test /status command
    out.truncate(0)
    out.seek(0)
    shell.handle_command("/status")

    status_text = out.getvalue()
    assert "OSA System Status" in status_text
    assert "Context Budget" in status_text


def test_shell_mode_command() -> None:
    agent = _mock_agent()
    inp = io.StringIO()
    out = io.StringIO()

    shell = InteractiveShell(agent, input_stream=inp, output_stream=out)

    shell.handle_command("/mode task")
    assert agent.mode == AgentMode.TASK

    shell.handle_command("/mode invalid_mode")
    assert "Invalid mode" in out.getvalue()


def test_shell_new_command_resets_session() -> None:
    agent = _mock_agent()
    agent.context.add(MagicMock(role="user", content="Hello"))

    shell = InteractiveShell(agent, input_stream=io.StringIO(), output_stream=io.StringIO())
    shell.handle_command("/new")

    # Context should be reset to system prompt only
    assert len(agent.context.messages()) == 1


def test_shell_tools_command() -> None:
    agent = _mock_agent()
    out = io.StringIO()
    shell = InteractiveShell(agent, input_stream=io.StringIO(), output_stream=out)

    shell.handle_command("/tools")
    assert "Registered Tools" in out.getvalue()


def test_shell_run_loop_with_chat_and_exit() -> None:
    agent = _mock_agent(reply="I am online and ready.")
    inp = io.StringIO("Hello OSA!\nexit\n")
    out = io.StringIO()

    shell = InteractiveShell(agent, input_stream=inp, output_stream=out)
    shell.run()

    output = out.getvalue()
    assert "You >" in output
    assert "OSA >" in output
    assert "I am online and ready." in output
    assert "OSA session terminated." in output
