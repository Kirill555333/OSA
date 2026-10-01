"""Unit tests for voice integration in InteractiveShell (0.9.5)."""

from __future__ import annotations

from dataclasses import dataclass
import io
from unittest.mock import MagicMock
import pytest

from osa.core import Agent
from osa.models import ModelInterface, ModelResponse
from osa.shell import InteractiveShell


@dataclass
class FakeVoiceSessionResult:
    transcribed_text: str


def _mock_agent(reply: str = "Assistant reply") -> Agent:
    mock_model = MagicMock(spec=ModelInterface)
    mock_model.model_name = "test-qwen"
    mock_model.health_check.return_value = True
    mock_model.generate.return_value = ModelResponse(content=reply, model_name="test-qwen")
    return Agent(model=mock_model, system_prompt="System")


def test_voice_command_standby_notice() -> None:
    agent = _mock_agent()
    out = io.StringIO()
    # Shell without voice_runtime
    shell = InteractiveShell(agent, voice_runtime=None, input_stream=io.StringIO(), output_stream=out)

    shell.handle_command("/voice")
    output = out.getvalue()
    assert "[Voice Runtime Standby]" in output
    assert "Live microphone/speaker runtime is not configured" in output


def test_voice_command_active_run_and_exit() -> None:
    agent = _mock_agent(reply="I am listening and speaking.")
    out = io.StringIO()

    mock_voice = MagicMock()
    mock_voice.run_once.side_effect = [
        FakeVoiceSessionResult(transcribed_text="What is the weather?"),
        FakeVoiceSessionResult(transcribed_text="выход"),
    ]

    shell = InteractiveShell(
        agent,
        voice_runtime=mock_voice,
        input_stream=io.StringIO(),
        output_stream=out,
    )

    shell.handle_command("/voice")

    output = out.getvalue()
    assert "OSA LIVE VOICE MODE ACTIVE" in output
    assert "You (Voice) > What is the weather?" in output
    assert "OSA (Voice) > I am listening and speaking." in output
    assert "Voice session ended." in output

    mock_voice.start.assert_called_once()
    mock_voice.speak.assert_called_once_with("I am listening and speaking.")
    mock_voice.stop.assert_called_once()


def test_voice_status_indicator_in_cmd_status() -> None:
    agent = _mock_agent()

    # Without voice runtime
    out1 = io.StringIO()
    shell1 = InteractiveShell(agent, voice_runtime=None, output_stream=out1)
    shell1.handle_command("/status")
    assert "Standby" in out1.getvalue()

    # With voice runtime
    out2 = io.StringIO()
    shell2 = InteractiveShell(agent, voice_runtime=MagicMock(), output_stream=out2)
    shell2.handle_command("/status")
    assert "Connected" in out2.getvalue()
