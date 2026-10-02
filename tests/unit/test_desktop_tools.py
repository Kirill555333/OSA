"""Targeted unit tests for desktop control tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osa.desktop.automation import DesktopPoint
from osa.tools import (
    ClickMouseTool,
    CloseApplicationTool,
    HotkeyTool,
    LaunchApplicationTool,
    OpenUrlTool,
    PressKeyTool,
    ScrollMouseTool,
    SetSystemVolumeTool,
    TakeScreenshotTool,
    ToolRegistry,
    TypeTextTool,
)


def test_take_screenshot_tool() -> None:
    tool = TakeScreenshotTool()
    fake_auto = MagicMock()
    fake_auto.screenshot.return_value = b"\x89PNG\r\n\x1a\nfake"
    tool._automation = fake_auto

    result = tool.execute({})
    assert result.success
    assert "data/screenshots" in result.output
    fake_auto.screenshot.assert_called_once()


def test_click_mouse_tool() -> None:
    tool = ClickMouseTool()
    fake_auto = MagicMock()
    tool._automation = fake_auto

    result = tool.execute({"x": 150, "y": 250})
    assert result.success
    fake_auto.click_point.assert_called_with(DesktopPoint(x=150, y=250))


def test_scroll_mouse_tool() -> None:
    tool = ScrollMouseTool()
    with patch("ctypes.cdll.LoadLibrary") as mock_lib:
        fake_cg = MagicMock()
        fake_cg.CGEventCreateScrollWheelEvent.return_value = 12345
        mock_lib.return_value = fake_cg
        result = tool.execute({"delta": 3})
        assert result.success


def test_hotkey_tool() -> None:
    tool = HotkeyTool()
    fake_auto = MagicMock()
    tool._automation = fake_auto

    result = tool.execute({"keys": ["command", "f"]})
    assert result.success
    fake_auto.hotkey.assert_called_with(["command", "f"])


def test_desktop_registry_full() -> None:
    registry = ToolRegistry()
    registry.register(LaunchApplicationTool())
    registry.register(CloseApplicationTool())
    registry.register(TakeScreenshotTool())
    registry.register(ClickMouseTool())
    registry.register(ScrollMouseTool())
    registry.register(HotkeyTool())
    registry.register(TypeTextTool())
    registry.register(PressKeyTool())
    registry.register(SetSystemVolumeTool())
    registry.register(OpenUrlTool())

    assert registry.get("take_screenshot") is not None
    assert registry.get("click_mouse") is not None
    assert registry.get("scroll_mouse") is not None
    assert registry.get("hotkey") is not None
