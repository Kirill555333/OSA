"""Unit tests for OutputTruncator, ActionLoopDetector, and OSAVoiceEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osa.tools.guardrails import ActionLoopDetector, OutputTruncator, apply_tool_guardrail
from osa.tools.registry import ToolResult
from osa.voice.engine import OSAVoiceEngine


def test_output_truncator_short_text() -> None:
    truncator = OutputTruncator(max_characters=100)
    text = "Short output"
    assert truncator.truncate(text) == text


def test_output_truncator_long_text() -> None:
    truncator = OutputTruncator(max_characters=50, head_lines=2, tail_lines=2)
    lines = [f"line {i}" for i in range(20)]
    long_text = "\n".join(lines)
    truncated = truncator.truncate(long_text)
    assert len(truncated) < len(long_text)
    assert "omitted to fit context budget" in truncated
    assert "line 0" in truncated
    assert "line 19" in truncated


def test_apply_tool_guardrail_truncates_large_result() -> None:
    truncator = OutputTruncator(max_characters=40, head_lines=1, tail_lines=1)
    huge_result = ToolResult(
        success=True,
        output="A" * 200,
    )
    safe_result = apply_tool_guardrail(huge_result, truncator)
    assert safe_result.success
    assert len(safe_result.output) < 200


def test_action_loop_detector() -> None:
    detector = ActionLoopDetector(max_repeated_calls=3)
    assert not detector.record_and_check("open_url", {"url": "https://google.com"})
    assert not detector.record_and_check("open_url", {"url": "https://google.com"})
    # Third consecutive identical call triggers loop detection
    assert detector.record_and_check("open_url", {"url": "https://google.com"})

    detector.reset()
    assert not detector.record_and_check("open_url", {"url": "https://google.com"})


def test_voice_engine_macos_say() -> None:
    engine = OSAVoiceEngine()
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        with patch.object(engine, "_has_say", True):
            success = engine.speak("Привет")
            assert success
            mock_popen.assert_called_once()
