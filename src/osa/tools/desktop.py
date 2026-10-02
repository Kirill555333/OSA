"""Complete desktop, vision and universal application management tools for OSA."""

from __future__ import annotations

import base64
from collections.abc import Mapping
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

from osa.desktop.automation import DesktopPoint
from osa.desktop.factory import create_desktop_automation
from osa.tools.registry import ToolInterface, ToolResult

POPULAR_ALIASES: dict[str, str] = {
    "хром": "chrome",
    "хрома": "chrome",
    "хрому": "chrome",
    "хроме": "chrome",
    "браузер": "chrome",
    "телеграм": "telegram",
    "телеграма": "telegram",
    "телеграме": "telegram",
    "тг": "telegram",
    "телеграмм": "telegram",
    "калькулятор": "calculator",
    "калькулятора": "calculator",
    "калькуляторе": "calculator",
    "терминал": "terminal",
    "курсор": "cursor",
    "фотошоп": "photoshop",
    "ворд": "word",
    "эксель": "excel",
    "блокнот": "notes",
    "заметки": "notes",
    "заметка": "notes",
    "заметку": "notes",
    "музыка": "music",
    "сафари": "safari",
    "настройки": "settings",
}


def find_installed_macos_app(query: str) -> tuple[str, Path] | None:
    """Universally scan macOS directories to find any installed application by name/alias."""
    q = query.strip().lower()
    q_norm = POPULAR_ALIASES.get(q, q)

    scan_dirs = [
        Path("/Applications"),
        Path("/System/Applications"),
        Path("/System/Applications/Utilities"),
        Path.home() / "Applications",
    ]

    installed_apps: list[Path] = []
    for sdir in scan_dirs:
        if sdir.exists():
            for item in sdir.iterdir():
                if item.name.endswith(".app"):
                    installed_apps.append(item)
                elif item.is_dir() and not item.name.startswith("."):
                    try:
                        for sub in item.iterdir():
                            if sub.name.endswith(".app"):
                                installed_apps.append(sub)
                    except Exception:
                        pass

    for app in installed_apps:
        stem_lower = app.stem.lower()
        if q_norm == stem_lower or q == stem_lower:
            return (app.stem, app)

    for app in installed_apps:
        stem_lower = app.stem.lower()
        if q_norm in stem_lower or q in stem_lower:
            return (app.stem, app)

    return None


def activate_macos_app(app_name: str) -> None:
    """Force bring application window to front and grant keyboard focus."""
    if sys.platform == "darwin":
        script = f'tell application "{app_name}" to activate'
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.4)


class LaunchApplicationTool(ToolInterface):
    """Tool to launch and activate any desktop application in the system."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "launch_application"

    @property
    def description(self) -> str:
        return "Launch any installed application (e.g. 'Google Chrome', 'Telegram', 'Calculator', 'Notes')."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "application_name": {
                    "type": "string",
                    "description": "Name of the application to launch.",
                },
            },
            "required": ["application_name"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_name = str(arguments.get("application_name", "")).strip()
        if not raw_name:
            return ToolResult(success=False, error="Application name cannot be empty.")

        if sys.platform == "darwin":
            match = find_installed_macos_app(raw_name)
            if match:
                app_name, app_path = match
                try:
                    print(f" [⚡ Запуск и фокус: {app_name}]", flush=True)
                    subprocess.run(["/usr/bin/open", str(app_path)], check=True)
                    activate_macos_app(app_name)
                    return ToolResult(success=True, output=f"Приложение '{app_name}' запущено.")
                except Exception as exc:
                    return ToolResult(success=False, error=f"Failed to launch '{app_name}': {exc}")
            else:
                try:
                    print(f" [⚡ Запуск: {raw_name}]", flush=True)
                    subprocess.run(["/usr/bin/open", "-a", raw_name], check=True)
                    activate_macos_app(raw_name)
                    return ToolResult(success=True, output=f"Приложение '{raw_name}' запущено.")
                except Exception as exc:
                    return ToolResult(success=False, error=f"Не удалось найти приложение '{raw_name}'.")

        elif sys.platform == "win32":
            try:
                print(f" [⚡ Запуск: {raw_name}]", flush=True)
                subprocess.Popen([raw_name], shell=True)
                time.sleep(0.5)
                return ToolResult(success=True, output=f"Приложение '{raw_name}' запущено на Windows.")
            except Exception as exc:
                return ToolResult(success=False, error=f"Failed to launch '{raw_name}': {exc}")
        else:
            try:
                self._automation.launch_application(raw_name)
                return ToolResult(success=True, output=f"Приложение '{raw_name}' запущено.")
            except Exception as exc:
                return ToolResult(success=False, error=f"Failed to launch '{raw_name}': {exc}")


class CloseApplicationTool(ToolInterface):
    """Tool to close any running desktop application."""

    def __init__(self) -> None:
        self._automation = create_desktop_automation()

    @property
    def name(self) -> str:
        return "close_application"

    @property
    def description(self) -> str:
        return "Close or quit any running application (e.g. 'Telegram', 'Google Chrome', 'Calculator', 'Notes')."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "application_name": {
                    "type": "string",
                    "description": "Name of the application to quit.",
                },
            },
            "required": ["application_name"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_name = str(arguments.get("application_name", "")).strip()
        if not raw_name:
            return ToolResult(success=False, error="Application name cannot be empty.")

        if sys.platform == "darwin":
            match = find_installed_macos_app(raw_name)
            app_name = match[0] if match else raw_name
            print(f" [⚡ Закрытие: {app_name}]", flush=True)

            # 1. Graceful quit
            script = f'tell application "{app_name}" to quit'
            subprocess.run(["osascript", "-e", script], capture_output=True)

            # 2. Terminate background processes
            subprocess.run(["pkill", "-i", "-f", app_name], capture_output=True)

            return ToolResult(success=True, output=f"Приложение '{app_name}' закрыто.")

        elif sys.platform == "win32":
            print(f" [⚡ Закрытие: {raw_name}]", flush=True)
            subprocess.run(["taskkill", "/IM", f"{raw_name}.exe", "/F"], capture_output=True)
            return ToolResult(success=True, output=f"Приложение '{raw_name}' закрыто.")
        else:
            subprocess.run(["pkill", "-i", "-f", raw_name], capture_output=True)
            return ToolResult(success=True, output=f"Приложение '{raw_name}' закрыто.")


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


class OpenUrlTool(ToolInterface):
    """Tool to open URLs or search on web."""

    @property
    def name(self) -> str:
        return "open_url"

    @property
    def description(self) -> str:
        return "Open a web URL or search directly on YouTube / Google in browser."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to open.",
                },
            },
            "required": ["url"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raw_url = str(arguments.get("url", "")).strip()
        if not raw_url:
            return ToolResult(success=False, error="URL cannot be empty.")

        url = raw_url
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            print(f" [⚡ Открытие ссылки: {url}]", flush=True)
            if sys.platform == "darwin":
                try:
                    subprocess.run(["open", "-a", "Google Chrome", url], check=True)
                except subprocess.CalledProcessError:
                    subprocess.run(["open", url], check=True)
            elif sys.platform == "win32":
                subprocess.run(["start", url], shell=True, check=True)
            else:
                subprocess.run(["xdg-open", url], check=True)

            return ToolResult(success=True, output=f"Открыта ссылка '{url}'.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to open URL '{url}': {exc}")
