"""Unit tests for MacOSDesktopAutomation and desktop factory (0.9.1)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from osa.desktop import (
    DesktopPoint,
    MacOSDesktopAutomation,
    MacOSDesktopAutomationConfig,
    create_desktop_automation,
)
from osa.desktop.automation import (
    DesktopAutomationActionError,
    DesktopAutomationConnectionError,
)


def test_factory_creates_platform_backend() -> None:
    backend = create_desktop_automation()
    assert backend is not None

    if sys.platform == "darwin":
        assert isinstance(backend, MacOSDesktopAutomation)
    elif sys.platform == "win32":
        from osa.desktop.windows import WindowsDesktopAutomation

        assert isinstance(backend, WindowsDesktopAutomation)


def test_macos_desktop_requires_darwin() -> None:
    with patch("sys.platform", "linux"):
        mac_backend = MacOSDesktopAutomation()
        with pytest.raises(DesktopAutomationConnectionError, match="available only on macOS"):
            mac_backend.screenshot()


def test_macos_desktop_health_check() -> None:
    backend = MacOSDesktopAutomation()
    if sys.platform == "darwin":
        assert backend.health_check() is True
    else:
        with patch("sys.platform", "win32"):
            assert backend.health_check() is False


def test_macos_desktop_applications_parsing() -> None:
    backend = MacOSDesktopAutomation()

    mock_output = "Finder, Safari, Terminal, Code"
    with patch.object(backend, "_run_osascript", return_value=mock_output):
        apps = backend.applications()
        assert len(apps) == 4
        app_names = [a.name for a in apps]
        assert "Finder" in app_names
        assert "Safari" in app_names
        assert apps[0].running is True


def test_macos_desktop_launch_and_close_application() -> None:
    backend = MacOSDesktopAutomation()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        app = backend.launch_application("Calculator")
        assert app.name == "Calculator"
        assert app.running is True
        mock_run.assert_called_once()
        assert "/usr/bin/open" in mock_run.call_args[0][0]

    with patch.object(backend, "_run_osascript", return_value="") as mock_osa:
        backend.close_application("mac-app-calculator")
        mock_osa.assert_called_once()
        assert 'tell application "calculator" to quit' in mock_osa.call_args[0][0]


def test_macos_desktop_windows_parsing() -> None:
    backend = MacOSDesktopAutomation()

    mock_script_out = "Safari:::OSA Project Docs\nTerminal:::zsh - /Users/kirill/OSA\n"
    with patch.object(backend, "_run_osascript", return_value=mock_script_out):
        windows = backend.windows()
        assert len(windows) == 2
        assert windows[0].title == "OSA Project Docs"
        assert windows[0].application_id == "Safari"
        assert windows[1].title == "zsh - /Users/kirill/OSA"


def test_macos_desktop_screenshot_flow(tmp_path: Path) -> None:
    backend = MacOSDesktopAutomation()

    # Emulate screencapture writing a dummy PNG
    fake_png = b"\x89PNG\r\n\x1a\nfake-image-bytes"

    def fake_screencapture(*args: object, **kwargs: object) -> MagicMock:
        # args[0] is ['/usr/sbin/screencapture', '-x', '-t', 'png', path]
        target_path = Path(args[0][4])  # type: ignore[index]
        target_path.write_bytes(fake_png)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_screencapture):
        with patch.object(backend, "_require_macos", return_value=None):
            data = backend.screenshot()
            assert data == fake_png


def test_macos_desktop_type_press_hotkey() -> None:
    backend = MacOSDesktopAutomation()

    with patch.object(backend, "_run_osascript", return_value="") as mock_osa:
        backend.type_text("Hello World!")
        assert 'keystroke "Hello World!"' in mock_osa.call_args[0][0]

    with patch.object(backend, "_run_osascript", return_value="") as mock_osa:
        backend.press("return")
        assert "key code 36" in mock_osa.call_args[0][0]

    with patch.object(backend, "_run_osascript", return_value="") as mock_osa:
        backend.hotkey("command", "c")
        assert 'keystroke "c" using {command down}' in mock_osa.call_args[0][0]
