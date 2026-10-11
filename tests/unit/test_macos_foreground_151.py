"""Tests for the macOS foreground application query (OSA 1.5.1.4)."""

from __future__ import annotations

from osa.desktop.automation import DesktopAutomationActionError
from osa.desktop.macos import MacOSDesktopAutomation


def _backend(answer) -> tuple[MacOSDesktopAutomation, list[str]]:
    backend = MacOSDesktopAutomation()
    scripts: list[str] = []

    def fake_osascript(script: str) -> str:
        scripts.append(script)

        if isinstance(answer, Exception):
            raise answer

        return answer

    backend._run_osascript = fake_osascript

    return backend, scripts


def test_foreground_application_returns_the_frontmost_process_name() -> None:
    backend, scripts = _backend("Terminal\n")

    assert backend.foreground_application() == "Terminal"
    assert "frontmost" in scripts[0]
    assert "System Events" in scripts[0]


def test_foreground_application_trims_whitespace() -> None:
    backend, _ = _backend("  Safari  ")

    assert backend.foreground_application() == "Safari"


def test_foreground_application_is_none_for_an_empty_answer() -> None:
    backend, _ = _backend("")

    assert backend.foreground_application() is None


def test_foreground_application_is_none_when_the_script_fails() -> None:
    backend, _ = _backend(DesktopAutomationActionError("not authorized"))

    assert backend.foreground_application() is None
