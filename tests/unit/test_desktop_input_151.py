"""Tests for honest results and safe handling in desktop input tools (OSA 1.5.1.4)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from osa.tools import desktop as desktop_module
from osa.tools.desktop import (
    HotkeyTool,
    PressKeyTool,
    ScrollMouseTool,
    SetSystemVolumeTool,
    TakeScreenshotTool,
)


class _Backend:
    """Records what the tools ask the desktop layer to do."""

    def __init__(self) -> None:
        self.hotkeys: list[tuple[str, ...]] = []
        self.pressed: list[str] = []

    def hotkey(self, *keys: str) -> None:
        self.hotkeys.append(keys)

    def press(self, key: str) -> None:
        self.pressed.append(key)

    def screenshot(self) -> bytes:
        return b"\x89PNG-test-bytes"


def _with_backend(build):
    """Create a tool while the desktop factory returns a recording backend."""
    backend = _Backend()
    original = desktop_module.create_desktop_automation
    desktop_module.create_desktop_automation = lambda: backend

    try:
        return build(), backend
    finally:
        desktop_module.create_desktop_automation = original


def _on_platform(platform: str, **replacements):
    """Context-like helper: run a function with a fake platform and fake modules."""

    def runner(function):
        original_sys = desktop_module.sys
        saved = {name: getattr(desktop_module, name) for name in replacements}
        desktop_module.sys = SimpleNamespace(platform=platform)

        for name, value in replacements.items():
            setattr(desktop_module, name, value)

        try:
            return function()
        finally:
            desktop_module.sys = original_sys

            for name, value in saved.items():
                setattr(desktop_module, name, value)

    return runner


# --- hotkey ------------------------------------------------------------------------


def test_hotkey_passes_each_key_separately_to_the_desktop_layer() -> None:
    tool, backend = _with_backend(HotkeyTool)

    result = tool.execute({"keys": ["command", "c"]})

    assert result.success is True
    assert backend.hotkeys == [("command", "c")]
    assert "command+c" in result.output


def test_hotkey_converts_keys_to_text_and_accepts_tuples() -> None:
    tool, backend = _with_backend(HotkeyTool)

    tool.execute({"keys": ("ctrl", 5)})

    assert backend.hotkeys == [("ctrl", "5")]


def test_hotkey_rejects_missing_or_malformed_keys() -> None:
    tool, backend = _with_backend(HotkeyTool)

    for arguments in ({}, {"keys": []}, {"keys": "command+c"}, {"keys": None}):
        assert tool.execute(arguments).success is False

    assert backend.hotkeys == []


def test_press_key_still_works() -> None:
    tool, backend = _with_backend(PressKeyTool)

    assert tool.execute({"key": " Enter "}).success is True
    assert backend.pressed == ["enter"]


# --- screenshot file names ---------------------------------------------------------


def _screenshot(filename: str | None) -> tuple[Path, Path]:
    tool, _ = _with_backend(TakeScreenshotTool)
    previous_home = os.environ.get("OSA_HOME")

    with tempfile.TemporaryDirectory() as folder:
        os.environ["OSA_HOME"] = folder

        try:
            arguments = {} if filename is None else {"filename": filename}
            result = tool.execute(arguments)

            assert result.success is True, result.error
            written = Path(result.metadata["path"])
            root = Path(folder).resolve()
            inside = (root / "data" / "screenshots").resolve()

            return written.resolve().relative_to(root), inside.relative_to(root)
        finally:
            if previous_home is None:
                os.environ.pop("OSA_HOME", None)
            else:
                os.environ["OSA_HOME"] = previous_home


def test_screenshot_defaults_to_latest_png() -> None:
    written, folder = _screenshot(None)

    assert written == folder / "latest.png"


def test_screenshot_adds_the_png_extension() -> None:
    written, folder = _screenshot("shot")

    assert written == folder / "shot.png"


def test_screenshot_name_cannot_escape_the_screenshots_folder() -> None:
    for hostile, expected in (
        ("../../evil", "evil.png"),
        ("../../../.ssh/authorized_keys", "authorized_keys.png"),
        ("/etc/passwd", "passwd.png"),
        ("..\\..\\windows\\x.png", "x.png"),
        ("a/b/c.png", "c.png"),
    ):
        written, folder = _screenshot(hostile)

        assert written == folder / expected, hostile


def test_screenshot_with_only_dots_falls_back_to_the_default_name() -> None:
    for name in ("..", ".", "   ", "../.."):
        written, folder = _screenshot(name)

        assert written.parent == folder


# --- scrolling ---------------------------------------------------------------------


def test_scroll_is_not_reported_as_done_where_it_is_not_implemented() -> None:
    for platform in ("win32", "linux"):
        result = _on_platform(platform)(lambda: ScrollMouseTool().execute({"delta": 3}))

        assert result.success is False, platform
        assert "не поддерживается" in (result.error or "")


# --- volume ------------------------------------------------------------------------


def test_volume_is_not_reported_as_set_on_windows() -> None:
    calls: list[list[str]] = []
    fake = SimpleNamespace(run=lambda command, **kwargs: calls.append(command))

    result = _on_platform("win32", subprocess=fake)(
        lambda: SetSystemVolumeTool().execute({"volume": 40})
    )

    assert result.success is False
    assert "Windows" in (result.error or "")
    assert calls == []


def test_volume_on_macos_runs_osascript_and_reports_success() -> None:
    calls: list[tuple[list[str], dict]] = []
    fake = SimpleNamespace(run=lambda command, **kwargs: calls.append((command, kwargs)))

    result = _on_platform("darwin", subprocess=fake)(
        lambda: SetSystemVolumeTool().execute({"volume": 40})
    )

    assert result.success is True
    assert "40%" in result.output
    assert calls[0][0] == ["osascript", "-e", "set volume output volume 40"]
    assert calls[0][1]["check"] is True


def test_volume_on_linux_checks_the_mixer_result() -> None:
    calls: list[tuple[list[str], dict]] = []
    fake = SimpleNamespace(run=lambda command, **kwargs: calls.append((command, kwargs)))

    result = _on_platform("linux", subprocess=fake)(
        lambda: SetSystemVolumeTool().execute({"volume": 70})
    )

    assert result.success is True
    assert calls[0][0][0] == "amixer"
    assert calls[0][1]["check"] is True


def test_volume_failure_on_linux_is_reported() -> None:
    def failing(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    result = _on_platform("linux", subprocess=SimpleNamespace(run=failing))(
        lambda: SetSystemVolumeTool().execute({"volume": 70})
    )

    assert result.success is False
    assert "volume" in (result.error or "").lower()
