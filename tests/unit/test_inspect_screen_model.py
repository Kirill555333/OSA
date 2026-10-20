"""Tests for InspectScreenTool via ModelInterface (4.3)."""

from __future__ import annotations

from unittest.mock import MagicMock

from osa.models import ModelResponse
from osa.tools.desktop import InspectScreenTool


def _make_tool(tmp_path, monkeypatch, *, model=None):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    monkeypatch.setattr("osa.tools.desktop.sys.platform", "linux")

    tool = InspectScreenTool(model=model)
    fake_automation = MagicMock()
    fake_automation.screenshot.return_value = b"\x89PNG\r\n\x1a\nfake"
    tool._automation = fake_automation
    return tool


def test_inspect_screen_uses_model_when_provided(tmp_path, monkeypatch):
    model = MagicMock()
    model.generate.return_value = ModelResponse(
        content="На экране открыт калькулятор",
        model_name="mock",
    )
    tool = _make_tool(tmp_path, monkeypatch, model=model)

    result = tool.execute({"query": "что на экране"})

    assert result.success is True
    assert "калькулятор" in result.output
    model.generate.assert_called_once()

    call_args = model.generate.call_args
    request = call_args.args[0]
    assert "[image_path:" in request.messages[0].content
    assert "что на экране" in request.messages[0].content


def test_inspect_screen_model_empty_content_falls_back_to_message(tmp_path, monkeypatch):
    model = MagicMock()
    model.generate.return_value = ModelResponse(content="   ", model_name="mock")
    tool = _make_tool(tmp_path, monkeypatch, model=model)

    result = tool.execute({"query": "что видно"})

    assert result.success is True
    assert "Не удалось распознать экран" in result.output


def test_inspect_screen_model_exception_returns_failure(tmp_path, monkeypatch):
    model = MagicMock()
    model.generate.side_effect = RuntimeError("model offline")
    tool = _make_tool(tmp_path, monkeypatch, model=model)

    result = tool.execute({"query": "что на экране"})

    assert result.success is False
    assert "model offline" in (result.error or "")


def test_inspect_screen_without_model_uses_urllib(tmp_path, monkeypatch):
    tool = _make_tool(tmp_path, monkeypatch, model=None)

    fake_response_body = b'{"choices":[{"message":{"content":"urllib reply"}}]}'

    class FakeResponse:
        def read(self) -> bytes:
            return fake_response_body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    called = {}

    def fake_urlopen(req, timeout=None):
        called["req"] = req
        return FakeResponse()

    monkeypatch.setattr("osa.tools.desktop.urllib.request.urlopen", fake_urlopen)

    result = tool.execute({"query": "что видно"})

    assert result.success is True
    assert result.output == "urllib reply"
    assert "req" in called


def test_inspect_screen_metadata_contains_screenshot_path(tmp_path, monkeypatch):
    model = MagicMock()
    model.generate.return_value = ModelResponse(content="ok", model_name="mock")
    tool = _make_tool(tmp_path, monkeypatch, model=model)

    result = tool.execute({"query": "x"})

    assert result.success is True
    assert "screenshot" in result.metadata
    assert result.metadata["screenshot"].endswith("latest.png")
