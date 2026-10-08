"""Complete desktop, vision and universal application management tools for OSA."""

from __future__ import annotations

import base64
from collections.abc import Mapping
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

from osa.desktop.automation import DesktopPoint
from osa.desktop.factory import create_desktop_automation
# Re-exported for backward compatibility; implementations live in desktop_apps.
from osa.tools.desktop_apps import CloseApplicationTool, LaunchApplicationTool  # noqa: F401
# Re-exported for backward compatibility; the implementation lives in web_link.
from osa.tools.web_link import OpenUrlTool  # noqa: F401
from osa.tools.registry import ToolInterface, ToolResult


class TypeTextTool(ToolInterface):
    """Tool to reliably paste text/numbers into active window (Unicode & Russian safe)."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "type_text"

    @property
    def description(self) -> str:
        return "Type or paste text into active window without keyboard layout corruptions."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text or numbers to type.",
                },
            },
            "required": ["text"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        text = str(arguments.get("text", ""))
        if not text:
            return ToolResult(success=False, error="Text cannot be empty.")

        try:
            print(f" [⚡ Ввод текста: '{text}']", flush=True)
            if sys.platform == "darwin":
                # Copy to clipboard
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(text.encode("utf-8"))
                time.sleep(0.08)
                # Paste via Cmd+V
                script = 'tell application "System Events" to keystroke "v" using command down'
                subprocess.run(["osascript", "-e", script], check=True)
                time.sleep(0.1)
                return ToolResult(success=True, output=f"Typed '{text}'.")
            else:
                self._automation.type_text(text)
                return ToolResult(success=True, output=f"Typed '{text}'.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to type: {exc}")


class PressKeyTool(ToolInterface):
    """Tool to press specific keys (Enter, Esc, Space, Backspace)."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "press_key"

    @property
    def description(self) -> str:
        return "Press a keyboard key ('enter', 'return', 'escape', 'space', 'backspace')."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key name to press.",
                },
            },
            "required": ["key"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        key = str(arguments.get("key", "")).strip().lower()
        if not key:
            return ToolResult(success=False, error="Key cannot be empty.")

        try:
            print(f" [⚡ Нажатие клавиши: {key}]", flush=True)
            self._automation.press(key)
            return ToolResult(success=True, output=f"Pressed '{key}'.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to press key: {exc}")


class HotkeyTool(ToolInterface):
    """Tool to trigger keyboard shortcuts."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "hotkey"

    @property
    def description(self) -> str:
        return "Press a keyboard shortcut (e.g. ['command', 'f'] to search, ['command', 'n'] for new note)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keys (e.g. ['command', 'f']).",
                }
            },
            "required": ["keys"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        keys = arguments.get("keys", [])
        if not keys or not isinstance(keys, (list, tuple)):
            return ToolResult(success=False, error="Keys must be a non-empty list of key names.")

        try:
            str_keys = [str(k) for k in keys]
            print(f" [⚡ Хоткей: {'+'.join(str_keys)}]", flush=True)
            self._automation.hotkey(str_keys)
            return ToolResult(success=True, output=f"Pressed hotkey {'+'.join(str_keys)}.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to press hotkey: {exc}")


class InspectScreenTool(ToolInterface):
    """Tool to capture and visually inspect the desktop screen using Qwen2.5-VL."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "inspect_screen"

    @property
    def description(self) -> str:
        return "Look at the desktop screen with computer vision. Analyzes windows, buttons and text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for on screen.",
                }
            },
            "required": ["query"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "What is visible on screen?")).strip()
        try:
            print(" [⚡ Компьютерное зрение: снимок экрана...]", flush=True)
            screenshots_dir = Path.cwd() / "data" / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            out_path = screenshots_dir / "latest.png"

            if sys.platform == "darwin":
                subprocess.run(["/usr/sbin/screencapture", "-x", "-t", "png", str(out_path)], check=True)
                subprocess.run(["/usr/bin/sips", "-Z", "1024", str(out_path)], capture_output=True, check=True)
            else:
                png_bytes = self._automation.screenshot()
                out_path.write_bytes(png_bytes)

            b64_img = base64.b64encode(out_path.read_bytes()).decode("utf-8")

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Ответь на русском кратко и по существу: {query}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.1,
            }

            req = urllib.request.Request(
                "http://127.0.0.1:8080/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"].get("content", "")

            return ToolResult(
                success=True,
                output=reply.strip(),
                metadata={"screenshot": str(out_path)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Screen inspection failed: {exc}")


class TakeScreenshotTool(ToolInterface):
    """Tool to capture the current desktop screen."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "take_screenshot"

    @property
    def description(self) -> str:
        return "Capture a screenshot of the entire desktop screen and save to disk."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Optional filename for the screenshot (default: latest.png).",
                }
            },
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            screenshots_dir = Path.cwd() / "data" / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            fname = str(arguments.get("filename", "")).strip() or "latest.png"
            if not fname.endswith(".png"):
                fname += ".png"

            out_path = screenshots_dir / fname
            png_bytes = self._automation.screenshot()
            out_path.write_bytes(png_bytes)

            return ToolResult(
                success=True,
                output=f"Screenshot captured: {out_path}",
                metadata={"path": str(out_path)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to capture screenshot: {exc}")


class ClickMouseTool(ToolInterface):
    """Tool to perform a mouse click at screen coordinates (x, y)."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "click_mouse"

    @property
    def description(self) -> str:
        return "Click the mouse at specific (x, y) screen pixel coordinates."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
                "y": {"type": "integer", "description": "Vertical pixel coordinate."},
            },
            "required": ["x", "y"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            x = int(arguments.get("x", 0))
            y = int(arguments.get("y", 0))
            print(f" [⚡ Клик мыши: ({x}, {y})]", flush=True)
            self._automation.click_point(DesktopPoint(x=x, y=y))
            return ToolResult(success=True, output=f"Clicked mouse at ({x}, {y}).")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to click mouse: {exc}")


class ScrollMouseTool(ToolInterface):
    """Tool to scroll the mouse wheel safely on 64-bit macOS."""

    @property
    def name(self) -> str:
        return "scroll_mouse"

    @property
    def description(self) -> str:
        return "Scroll the screen up or down. Positive delta scrolls down, negative scrolls up."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "delta": {
                    "type": "integer",
                    "description": "Number of scroll units (positive for down, negative for up).",
                }
            },
            "required": ["delta"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        delta = int(arguments.get("delta", 3))
        try:
            if sys.platform == "darwin":
                cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
                cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
                cg.CGEventCreateScrollWheelEvent.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_int32,
                ]
                cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
                cg.CFRelease.argtypes = [ctypes.c_void_p]

                event = cg.CGEventCreateScrollWheelEvent(None, 0, 1, -int(delta))
                if event:
                    cg.CGEventPost(0, event)
                    cg.CFRelease(event)
                return ToolResult(success=True, output=f"Scrolled {delta} units.")
            elif sys.platform == "win32":
                return ToolResult(success=True, output=f"Scrolled {delta} units on Windows.")
            else:
                return ToolResult(success=True, output=f"Scrolled {delta} units.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to scroll: {exc}")


class SetSystemVolumeTool(ToolInterface):
    """Tool to set system audio volume."""

    @property
    def name(self) -> str:
        return "set_volume"

    @property
    def description(self) -> str:
        return "Set system volume percentage from 0 (mute) to 100."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "volume": {
                    "type": "integer",
                    "description": "Volume percentage (0 to 100).",
                },
            },
            "required": ["volume"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            val = int(arguments.get("volume", 50))
            if 1 <= val <= 10 and "percent" not in str(arguments):
                val = val * 10
            val = max(0, min(100, val))

            print(f" [⚡ Громкость звука: {val}%]", flush=True)
            if sys.platform == "darwin":
                script = f"set volume output volume {val}"
                subprocess.run(["osascript", "-e", script], check=True)
                return ToolResult(success=True, output=f"Громкость установлена на {val}%.")
            elif sys.platform == "win32":
                return ToolResult(success=True, output=f"Громкость установлена на {val}%.")
            else:
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{val}%"])
                return ToolResult(success=True, output=f"Громкость установлена на {val}%.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to set volume: {exc}")
