"""Cross-platform regression tests for Mac development and Windows target (0.9.7)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from osa.actions import (
    ActionKind,
    ActionRequest,
    ActionRouter,
    ActionSafetyPipeline,
    AllowAllActionPolicy,
)
from osa.actions.desktop import DesktopActionAdapter
from osa.desktop import (
    DesktopAutomationConnectionError,
    DesktopAutomationInterface,
    FakeDesktopAutomation,
    MacOSDesktopAutomation,
    WindowsDesktopAutomation,
    create_desktop_automation,
    create_fake_desktop_automation,
)
from osa.tools import SafeFilesystem, SafeShell


def test_factory_resolves_platform_backend() -> None:
    """Factory must return platform-appropriate backend based on sys.platform."""
    # 1. Darwin (macOS)
    with patch("sys.platform", "darwin"):
        backend = create_desktop_automation()
        assert isinstance(backend, MacOSDesktopAutomation)

    # 2. Win32 (Windows)
    with patch("sys.platform", "win32"):
        backend = create_desktop_automation()
        assert isinstance(backend, WindowsDesktopAutomation)

    # 3. Other (Linux / Headless CI)
    with patch("sys.platform", "linux"):
        backend = create_desktop_automation()
        assert isinstance(backend, FakeDesktopAutomation)


def test_windows_backend_guards_against_non_windows_execution() -> None:
    """WindowsDesktopAutomation must reject execution on non-Windows platforms cleanly."""
    if sys.platform != "win32":
        backend = WindowsDesktopAutomation()
        with pytest.raises(DesktopAutomationConnectionError, match="available only on Windows"):
            backend.screenshot()


def test_macos_backend_guards_against_non_macos_execution() -> None:
    """MacOSDesktopAutomation must reject execution on non-macOS platforms cleanly."""
    with patch("sys.platform", "win32"):
        backend = MacOSDesktopAutomation()
        with pytest.raises(DesktopAutomationConnectionError, match="available only on macOS"):
            backend.screenshot()


def test_safe_filesystem_path_resolution_and_traversal_guards(tmp_path: Path) -> None:
    """SafeFilesystem must enforce workspace boundaries across platforms."""
    fs = SafeFilesystem(tmp_path)

    # Valid subpaths
    sub_dir = tmp_path / "src" / "osa"
    sub_dir.mkdir(parents=True)
    file_path = sub_dir / "agent.py"
    file_path.write_text("print('OSA')", encoding="utf-8")

    resolved = fs.resolve_path("src/osa/agent.py")
    assert resolved == file_path

    # Traversal attempts must fail closed
    with pytest.raises(Exception, match="outside the OSA workspace"):
        fs.resolve_path("../../etc/passwd")


def test_safe_shell_cross_platform_exit_and_echo(tmp_path: Path) -> None:
    """SafeShell runs basic commands across supported operating systems."""
    shell = SafeShell(tmp_path)

    # Echo
    res_echo = shell.execute("echo cross-platform-test")
    assert res_echo.returncode == 0
    assert "cross-platform-test" in res_echo.stdout

    # Non-zero exit code
    res_exit = shell.execute("exit 3")
    assert res_exit.returncode == 3


def test_desktop_action_adapter_parity_across_backends() -> None:
    """DesktopActionAdapter must expose identical action contracts regardless of backend."""
    fake_backend = create_fake_desktop_automation()
    adapter = DesktopActionAdapter(fake_backend)

    assert "desktop_screenshot" in adapter.supported_actions
    assert "desktop_applications" in adapter.supported_actions
    assert "desktop_windows" in adapter.supported_actions
    assert "desktop_health_check" in adapter.supported_actions
    assert "desktop_hotkey" in adapter.supported_actions

    req = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_health_check",
        arguments={},
        request_id="cp-hc-1",
    )
    result = adapter.execute(req)
    assert result.success is True
    assert result.data["healthy"] is True


@pytest.mark.skipif(sys.platform != "darwin", reason="Live macOS automation test")
def test_macos_live_health_check() -> None:
    """Live macOS health check verifying availability of screencapture and osascript."""
    backend = MacOSDesktopAutomation()
    assert backend.health_check() is True


@pytest.mark.skipif(sys.platform != "win32", reason="Live Windows automation test")
def test_windows_live_health_check() -> None:
    """Live Windows health check for target Ryzen/Radeon PC (skipped on Mac)."""
    backend = WindowsDesktopAutomation()
    assert backend.health_check() is True
