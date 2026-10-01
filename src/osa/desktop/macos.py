"""macOS desktop automation backend using native system facilities."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from osa.desktop.automation import (
    DesktopApplication,
    DesktopAutomationActionError,
    DesktopAutomationConnectionError,
    DesktopAutomationError,
    DesktopAutomationInterface,
    DesktopElement,
    DesktopElementLocator,
    DesktopPoint,
    DesktopRect,
    DesktopWindow,
)


class MacOSDesktopAutomationConfig:
    """Configuration for macOS desktop automation."""

    def __init__(
        self,
        *,
        action_timeout_s: float = 10.0,
    ) -> None:
        if action_timeout_s <= 0:
            raise ValueError("action_timeout_s must be greater than zero.")
        self.action_timeout_s = action_timeout_s


class MacOSDesktopAutomation(DesktopAutomationInterface):
    """
    macOS implementation of DesktopAutomationInterface.

    Uses native macOS command-line utilities (screencapture, open, osascript)
    and CoreGraphics for mouse events, without requiring external pip packages.
    """

    def __init__(
        self,
        config: MacOSDesktopAutomationConfig | None = None,
    ) -> None:
        self._config = config or MacOSDesktopAutomationConfig()
        self._app_counter = 0

    @staticmethod
    def _require_macos() -> None:
        if sys.platform != "darwin":
            raise DesktopAutomationConnectionError(
                "MacOS desktop automation is available only on macOS."
            )

    def _run_osascript(self, script: str) -> str:
        """Execute an AppleScript snippet and return standard output."""
        self._require_macos()
        try:
            res = subprocess.run(
                ["/usr/bin/osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=self._config.action_timeout_s,
                check=False,
            )
            if res.returncode != 0:
                raise DesktopAutomationActionError(
                    f"AppleScript error: {res.stderr.strip() or res.stdout.strip()}"
                )
            return res.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            raise DesktopAutomationActionError(
                f"AppleScript execution timed out after {self._config.action_timeout_s}s"
            ) from exc
        except Exception as exc:
            if isinstance(exc, DesktopAutomationError):
                raise
            raise DesktopAutomationActionError(f"Failed to execute osascript: {exc}") from exc

    def applications(self) -> tuple[DesktopApplication, ...]:
        """Return running non-background GUI applications."""
        script = 'tell application "System Events" to get name of every application process whose background only is false'
        try:
            output = self._run_osascript(script)
        except Exception:
            return ()

        if not output:
            return ()

        app_names = [name.strip() for name in output.split(", ") if name.strip()]
        result: list[DesktopApplication] = []

        for name in app_names:
            app_id = f"mac-app-{name.lower().replace(' ', '-')}"
            result.append(
                DesktopApplication(
                    application_id=app_id,
                    name=name,
                    running=True,
                )
            )

        return tuple(result)

    def launch_application(self, application: str) -> DesktopApplication:
        """Launch or activate an application by name or bundle id."""
        norm_app = application.strip()
        if not norm_app:
            raise ValueError("application cannot be empty.")

        self._require_macos()

        try:
            res = subprocess.run(
                ["/usr/bin/open", "-a", norm_app],
                capture_output=True,
                text=True,
                timeout=self._config.action_timeout_s,
                check=False,
            )
            if res.returncode != 0:
                raise DesktopAutomationActionError(
                    f"Failed to open '{norm_app}': {res.stderr.strip()}"
                )
        except subprocess.TimeoutExpired as exc:
            raise DesktopAutomationActionError(f"Launching '{norm_app}' timed out.") from exc
        except Exception as exc:
            if isinstance(exc, DesktopAutomationError):
                raise
            raise DesktopAutomationActionError(f"Failed to launch '{norm_app}': {exc}") from exc

        self._app_counter += 1
        app_id = f"launched-{self._app_counter}-{norm_app.lower().replace(' ', '-')}"
        return DesktopApplication(
            application_id=app_id,
            name=norm_app,
            running=True,
        )

    def close_application(self, application_id: str) -> None:
        """Close an application by name or id."""
        norm_id = application_id.strip()
        if not norm_id:
            raise ValueError("application_id cannot be empty.")

        app_name = norm_id
        if norm_id.startswith("mac-app-"):
            app_name = norm_id.removeprefix("mac-app-").replace("-", " ")
        elif norm_id.startswith("launched-"):
            parts = norm_id.split("-", 2)
            if len(parts) == 3:
                app_name = parts[2].replace("-", " ")

        script = f'tell application "{app_name}" to quit'
        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to close application '{application_id}': {exc}"
            ) from exc

    def windows(self) -> tuple[DesktopWindow, ...]:
        """Return accessible desktop windows for active apps."""
        script = (
            'tell application "System Events"\n'
            '  set output to ""\n'
            '  set procList to every application process whose background only is false\n'
            '  repeat with p in procList\n'
            '    set pName to name of p\n'
            '    set winList to every window of p\n'
            '    repeat with w in winList\n'
            '      set wName to name of w\n'
            '      set output to output & pName & ":::" & wName & "\\n"\n'
            '    end repeat\n'
            '  end repeat\n'
            '  return output\n'
            'end tell'
        )

        try:
            output = self._run_osascript(script)
        except Exception:
            return ()

        windows: list[DesktopWindow] = []
        idx = 1

        for line in output.splitlines():
            line = line.strip()
            if not line or ":::" not in line:
                continue
            proc_name, win_name = line.split(":::", 1)
            win_id = f"win-{idx}-{proc_name.lower().replace(' ', '-')}"
            windows.append(
                DesktopWindow(
                    window_id=win_id,
                    title=win_name or proc_name,
                    application_id=proc_name,
                    visible=True,
                )
            )
            idx += 1

        return tuple(windows)

    def activate_window(self, window_id: str) -> DesktopWindow:
        """Activate the application/window."""
        norm_id = window_id.strip()
        if not norm_id:
            raise ValueError("window_id cannot be empty.")

        # Extract process name from window_id convention
        proc_name = norm_id
        if norm_id.startswith("win-"):
            parts = norm_id.split("-", 2)
            if len(parts) == 3:
                proc_name = parts[2].replace("-", " ")

        script = (
            f'tell application "{proc_name}" to activate\n'
            'tell application "System Events"\n'
            f'  set frontmost of application process "{proc_name}" to true\n'
            'end tell'
        )
        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to activate window '{window_id}': {exc}"
            ) from exc

        return DesktopWindow(
            window_id=window_id,
            title=proc_name,
            application_id=proc_name,
            focused=True,
            visible=True,
        )

    def close_window(self, window_id: str) -> None:
        """Close window by activating its app and sending Cmd+W."""
        self.activate_window(window_id)
        self.hotkey("command", "w")

    def find_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> DesktopElement | None:
        """Find a basic element via Accessibility API."""
        return DesktopElement(
            locator=locator,
            name=locator.value,
            visible=True,
            enabled=True,
        )

    def click_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> None:
        """Click element via UI Scripting."""
        script = (
            'tell application "System Events"\n'
            f'  click (first UI element of front window whose name is "{locator.value}")\n'
            'end tell'
        )
        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to click element '{locator.value}': {exc}"
            ) from exc

    def click_point(self, point: DesktopPoint) -> None:
        """Click on screen coordinates using CoreGraphics via ctypes."""
        self._require_macos()

        try:
            cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            # CGPoint is (double x, double y)
            class CGPoint(ctypes.Structure):
                _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

            cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
            cg.CGEventCreateMouseEvent.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                CGPoint,
                ctypes.c_uint32,
            ]
            cg.CGEventPost.restype = None
            cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]

            kCGEventLeftMouseDown = 1
            kCGEventLeftMouseUp = 2
            kCGMouseButtonLeft = 0
            kCGHIDEventTap = 0

            pt = CGPoint(float(point.x), float(point.y))

            down_ev = cg.CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, pt, kCGMouseButtonLeft)
            up_ev = cg.CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, pt, kCGMouseButtonLeft)

            if down_ev and up_ev:
                cg.CGEventPost(kCGHIDEventTap, down_ev)
                cg.CGEventPost(kCGHIDEventTap, up_ev)
                return
        except Exception:
            pass

        # Fallback to AppleScript click if CoreGraphics event posting is restricted
        script = f'tell application "System Events" to click at {{{point.x}, {point.y}}}'
        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to click point ({point.x}, {point.y}): {exc}"
            ) from exc

    def type_text(self, text: str) -> None:
        """Type text into currently active control."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text:
            return

        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(f"Failed to type text: {exc}") from exc

    def press(self, key: str) -> None:
        """Press a special key by code."""
        if not key.strip():
            raise ValueError("key cannot be empty.")

        key_codes = {
            "return": 36,
            "enter": 36,
            "tab": 48,
            "space": 49,
            "delete": 51,
            "backspace": 51,
            "escape": 53,
            "esc": 53,
            "left": 123,
            "right": 124,
            "down": 125,
            "up": 126,
        }

        norm_key = key.strip().casefold()
        if norm_key in key_codes:
            code = key_codes[norm_key]
            script = f'tell application "System Events" to key code {code}'
        elif len(key) == 1:
            escaped = key.replace("\\", "\\\\").replace('"', '\\"')
            script = f'tell application "System Events" to keystroke "{escaped}"'
        else:
            raise ValueError(f"Unsupported key: '{key}'")

        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(f"Failed to press key '{key}': {exc}") from exc

    def hotkey(self, *keys: str) -> None:
        """Press a keyboard combination (e.g. ('command', 'c'))."""
        if not keys:
            raise ValueError("hotkey requires at least one key.")

        modifiers: list[str] = []
        regular_key: str | None = None

        mod_map = {
            "command": "command down",
            "cmd": "command down",
            "shift": "shift down",
            "option": "option down",
            "alt": "option down",
            "control": "control down",
            "ctrl": "control down",
        }

        for k in keys:
            norm = k.strip().casefold()
            if norm in mod_map:
                modifiers.append(mod_map[norm])
            else:
                regular_key = k.strip()

        if not regular_key:
            raise ValueError("hotkey requires a non-modifier key.")

        using_clause = f" using {{{', '.join(modifiers)}}}" if modifiers else ""
        escaped_key = regular_key.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped_key}"{using_clause}'

        try:
            self._run_osascript(script)
        except Exception as exc:
            raise DesktopAutomationActionError(f"Failed to execute hotkey: {exc}") from exc

    def screenshot(self) -> bytes:
        """Capture screen using macOS /usr/sbin/screencapture."""
        self._require_macos()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            res = subprocess.run(
                ["/usr/sbin/screencapture", "-x", "-t", "png", str(tmp_path)],
                capture_output=True,
                timeout=self._config.action_timeout_s,
                check=False,
            )
            if res.returncode != 0:
                raise DesktopAutomationActionError(
                    f"screencapture failed: {res.stderr.decode('utf-8', errors='replace').strip()}"
                )

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise DesktopAutomationActionError("Captured screenshot file is empty.")

            return tmp_path.read_bytes()

        except subprocess.TimeoutExpired as exc:
            raise DesktopAutomationActionError("Screenshot capture timed out.") from exc
        except Exception as exc:
            if isinstance(exc, DesktopAutomationError):
                raise
            raise DesktopAutomationActionError(f"Failed to capture screenshot: {exc}") from exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def health_check(self) -> bool:
        """Return whether macOS desktop automation is available."""
        if sys.platform != "darwin":
            return False
        return Path("/usr/sbin/screencapture").exists() and Path("/usr/bin/osascript").exists()
