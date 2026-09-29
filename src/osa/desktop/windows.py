"""Windows desktop automation backend using Microsoft UI Automation."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class WindowsDesktopAutomationConfig:
    """Configuration for the Windows desktop backend."""

    backend: str = "uia"
    application_start_timeout_ms: int = 30_000
    action_timeout_ms: int = 10_000
    screenshot_all_screens: bool = True

    def __post_init__(self) -> None:
        if self.backend != "uia":
            raise ValueError(
                "Windows desktop backend currently requires 'uia'."
            )

        if self.application_start_timeout_ms <= 0:
            raise ValueError(
                "application_start_timeout_ms must be greater than zero."
            )

        if self.action_timeout_ms <= 0:
            raise ValueError(
                "action_timeout_ms must be greater than zero."
            )


class WindowsDesktopAutomation(DesktopAutomationInterface):
    """
    Windows implementation of DesktopAutomationInterface.

    Uses pywinauto's Microsoft UI Automation backend and keeps all Windows/
    pywinauto-specific details inside this module.
    """

    def __init__(
        self,
        *,
        config: WindowsDesktopAutomationConfig | None = None,
    ) -> None:
        self._config = (
            config
            or WindowsDesktopAutomationConfig()
        )

        self._desktop: Any | None = None

        self._managed_applications: dict[
            str,
            tuple[Any, str],
        ] = {}

        self._managed_application_ids_by_pid: dict[
            int,
            str,
        ] = {}

        self._windows: dict[
            str,
            Any,
        ] = {}

        self._application_counter = 0

    @property
    def config(self) -> WindowsDesktopAutomationConfig:
        """Return backend configuration."""
        return self._config

    @staticmethod
    def _require_windows() -> None:
        if sys.platform != "win32":
            raise DesktopAutomationConnectionError(
                "Windows desktop automation is available only on Windows."
            )

    def _load_pywin32_backend(self) -> Any:
        self._require_windows()

        if self._desktop is not None:
            return self._desktop

        try:
            from pywinauto import Desktop
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "pywinauto is not installed. "
                "Run 'python -m pip install pywinauto==0.6.9 pillow' "
                "on Windows."
            ) from exc

        try:
            self._desktop = Desktop(
                backend=self._config.backend,
                allow_magic_lookup=False,
            )
        except Exception as exc:
            raise DesktopAutomationConnectionError(
                f"Failed to initialize Windows UI Automation: {exc}"
            ) from exc

        return self._desktop

    def _refresh_windows(self) -> None:
        desktop = self._load_pywin32_backend()

        try:
            windows = desktop.windows()
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to enumerate Windows desktop windows: {exc}"
            ) from exc

        refreshed: dict[str, Any] = {}

        for window in windows:
            try:
                if not window.is_visible():
                    continue

                window_id = self._window_id(window)
                refreshed[window_id] = window

            except Exception:
                continue

        self._windows = refreshed

    @staticmethod
    def _window_id(window: Any) -> str:
        """Build a session-stable identifier from UIA identity."""
        element_info = window.element_info

        try:
            handle = int(element_info.handle)

            if handle:
                return f"hwnd:{handle}"
        except Exception:
            pass

        try:
            process_id = int(
                element_info.process_id
            )
        except Exception:
            process_id = 0

        try:
            runtime_id = element_info.runtime_id
        except Exception:
            runtime_id = None

        return (
            f"uia:{process_id}:"
            f"{runtime_id!r}"
        )

    @staticmethod
    def _rect(
        rectangle: Any,
    ) -> DesktopRect:
        return DesktopRect(
            x=int(rectangle.left),
            y=int(rectangle.top),
            width=max(
                0,
                int(rectangle.right)
                - int(rectangle.left),
            ),
            height=max(
                0,
                int(rectangle.bottom)
                - int(rectangle.top),
            ),
        )

    def _application_id_for_pid(
        self,
        process_id: int,
    ) -> str:
        return self._managed_application_ids_by_pid.get(
            process_id,
            f"pid:{process_id}",
        )

    def _window_to_model(
        self,
        window_id: str,
        window: Any,
    ) -> DesktopWindow:
        try:
            title = str(
                window.window_text()
            )

            process_id = int(
                window.element_info.process_id
            )

            bounds = self._rect(
                window.rectangle()
            )

            visible = bool(
                window.is_visible()
            )

            focused = bool(
                window.is_active()
            )

            minimized = bool(
                window.is_minimized()
            )

        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to read Windows window state: {exc}"
            ) from exc

        return DesktopWindow(
            window_id=window_id,
            title=title,
            application_id=self._application_id_for_pid(
                process_id
            ),
            bounds=bounds,
            visible=visible,
            focused=focused,
            minimized=minimized,
        )

    def applications(
        self,
    ) -> tuple[DesktopApplication, ...]:
        self._refresh_windows()

        applications: dict[
            str,
            DesktopApplication,
        ] = {}

        # Applications explicitly started by OSA.
        for application_id, (
            application,
            application_name,
        ) in self._managed_applications.items():
            try:
                running = bool(
                    application.is_process_running()
                )
            except Exception:
                running = False

            process_id: int | None = None

            try:
                app_windows = application.windows()

                if app_windows:
                    process_id = int(
                        app_windows[0].element_info.process_id
                    )

            except Exception:
                pass

            applications[application_id] = DesktopApplication(
                application_id=application_id,
                name=application_name,
                running=running,
            )

            if process_id is not None:
                self._managed_application_ids_by_pid[
                    process_id
                ] = application_id

        # Discover visible applications from desktop windows.
        discovered: dict[
            int,
            DesktopApplication,
        ] = {}

        for window in self._windows.values():
            try:
                process_id = int(
                    window.element_info.process_id
                )

                if process_id in {
                    0,
                    4,
                }:
                    continue

                application_id = (
                    self._application_id_for_pid(
                        process_id
                    )
                )

                if application_id in applications:
                    continue

                name = str(
                    window.window_text()
                    or f"Process {process_id}"
                )

                path: str | None = None

                try:
                    from pywinauto.application import (
                        process_module,
                    )

                    path = process_module(
                        process_id
                    )
                except Exception:
                    pass

                discovered[process_id] = (
                    DesktopApplication(
                        application_id=application_id,
                        name=name,
                        path=path,
                        running=True,
                    )
                )

            except Exception:
                continue

        applications.update(
            {
                item.application_id: item
                for item in discovered.values()
            }
        )

        return tuple(
            applications.values()
        )

    def launch_application(
        self,
        application: str,
    ) -> DesktopApplication:
        if not application.strip():
            raise ValueError(
                "application cannot be empty."
            )

        self._require_windows()

        try:
            from pywinauto.application import Application
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "pywinauto is not installed."
            ) from exc

        try:
            app = Application(
                backend=self._config.backend,
                allow_magic_lookup=False,
            )

            app.start(
                application,
                timeout=(
                    self._config.application_start_timeout_ms
                    / 1000
                ),
                wait_for_idle=True,
            )

        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to launch application "
                f"'{application}': {exc}"
            ) from exc

        self._application_counter += 1

        application_id = (
            f"app-{self._application_counter}"
        )

        self._managed_applications[
            application_id
        ] = (
            app,
            application,
        )

        try:
            app_windows = app.windows()

            if app_windows:
                process_id = int(
                    app_windows[0].element_info.process_id
                )

                self._managed_application_ids_by_pid[
                    process_id
                ] = application_id

        except Exception:
            pass

        return DesktopApplication(
            application_id=application_id,
            name=application,
            path=application,
            running=True,
        )

    def close_application(
        self,
        application_id: str,
    ) -> None:
        application = self._managed_applications.get(
            application_id
        )

        try:
            if application is not None:
                app, _ = application

                for window in app.windows():
                    try:
                        window.close()
                    except Exception:
                        continue

                try:
                    app.wait_for_process_exit(
                        timeout=(
                            self._config.action_timeout_ms
                            / 1000
                        )
                    )
                except Exception:
                    pass

                self._managed_applications.pop(
                    application_id,
                    None,
                )
                return

            if application_id.startswith("pid:"):
                process_id = int(
                    application_id.removeprefix("pid:")
                )

                from pywinauto.application import Application

                app = Application(
                    backend=self._config.backend,
                    allow_magic_lookup=False,
                ).connect(
                    process=process_id
                )

                for window in app.windows():
                    try:
                        window.close()
                    except Exception:
                        continue

                return

        except ValueError as exc:
            raise DesktopAutomationActionError(
                f"Invalid application ID: {application_id}"
            ) from exc
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to close application "
                f"'{application_id}': {exc}"
            ) from exc

        raise DesktopAutomationError(
            f"Unknown application: {application_id}"
        )

    def windows(
        self,
    ) -> tuple[DesktopWindow, ...]:
        self._refresh_windows()

        return tuple(
            self._window_to_model(
                window_id,
                window,
            )
            for window_id, window in self._windows.items()
        )

    def _require_window(
        self,
        window_id: str,
    ) -> Any:
        self._refresh_windows()

        window = self._windows.get(
            window_id
        )

        if window is None:
            raise DesktopAutomationError(
                f"Unknown window: {window_id}"
            )

        return window

    def _active_window(self) -> Any:
        self._refresh_windows()

        for window in self._windows.values():
            try:
                if window.is_active():
                    return window
            except Exception:
                continue

        raise DesktopAutomationError(
            "No active desktop window."
        )

    def activate_window(
        self,
        window_id: str,
    ) -> DesktopWindow:
        window = self._require_window(
            window_id
        )

        try:
            window.set_focus()
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to activate window "
                f"'{window_id}': {exc}"
            ) from exc

        # Give UIA a fresh view of focus.
        self._refresh_windows()

        refreshed = self._windows.get(
            window_id
        )

        if refreshed is None:
            raise DesktopAutomationError(
                f"Window disappeared after activation: {window_id}"
            )

        return self._window_to_model(
            window_id,
            refreshed,
        )

    def close_window(
        self,
        window_id: str,
    ) -> None:
        window = self._require_window(
            window_id
        )

        try:
            window.close()
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to close window "
                f"'{window_id}': {exc}"
            ) from exc

        self._windows.pop(
            window_id,
            None,
        )

    @staticmethod
    def _control_type(
        value: str,
    ) -> str:
        mapping = {
            "button": "Button",
            "checkbox": "CheckBox",
            "radio": "RadioButton",
            "edit": "Edit",
            "input": "Edit",
            "text": "Text",
            "label": "Text",
            "window": "Window",
            "pane": "Pane",
            "list": "List",
            "listitem": "ListItem",
            "tab": "Tab",
            "tabitem": "TabItem",
            "menu": "Menu",
            "menuitem": "MenuItem",
            "combobox": "ComboBox",
            "tree": "Tree",
            "treeitem": "TreeItem",
            "hyperlink": "Hyperlink",
        }

        normalized = value.strip()

        return mapping.get(
            normalized.casefold(),
            normalized,
        )

    def _find_control(
        self,
        window: Any,
        locator: DesktopElementLocator,
    ) -> Any | None:
        kwargs: dict[str, Any] = {}

        if locator.kind == "name":
            kwargs["title"] = locator.value
        elif locator.kind == "role":
            kwargs["control_type"] = (
                self._control_type(locator.value)
            )
        elif locator.kind == "automation_id":
            kwargs["auto_id"] = locator.value
        else:
            raise DesktopAutomationActionError(
                f"Unsupported locator kind: {locator.kind}"
            )

        try:
            matches = window.descendants(
                **kwargs
            )

            if not matches:
                return None

            if len(matches) > 1:
                raise DesktopAutomationActionError(
                    f"Desktop locator is ambiguous and matched "
                    f"{len(matches)} elements: "
                    f"{locator.kind}={locator.value}"
                )

            return matches[0]

        except DesktopAutomationError:
            raise
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to find desktop element "
                f"'{locator.value}': {exc}"
            ) from exc

    def find_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> DesktopElement | None:
        window = (
            self._active_window()
            if window_id is None
            else self._require_window(window_id)
        )

        element = self._find_control(
            window,
            locator,
        )

        if element is None:
            return None

        try:
            rectangle = self._rect(
                element.rectangle()
            )

            try:
                role = str(
                    element.element_info.control_type
                )
            except Exception:
                role = None

            try:
                name = str(
                    element.element_info.name
                )
            except Exception:
                name = None

            return DesktopElement(
                locator=locator,
                name=name,
                role=role,
                bounds=rectangle,
                visible=bool(
                    element.is_visible()
                ),
                enabled=bool(
                    element.is_enabled()
                ),
            )

        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to inspect desktop element "
                f"'{locator.value}': {exc}"
            ) from exc

    def _require_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> Any:
        window = (
            self._active_window()
            if window_id is None
            else self._require_window(window_id)
        )

        element = self._find_control(
            window,
            locator,
        )

        if element is None:
            raise DesktopAutomationActionError(
                f"Desktop element not found: "
                f"{locator.kind}={locator.value}"
            )

        try:
            if not element.is_visible():
                raise DesktopAutomationActionError(
                    f"Desktop element is not visible: "
                    f"{locator.value}"
                )

            if not element.is_enabled():
                raise DesktopAutomationActionError(
                    f"Desktop element is not enabled: "
                    f"{locator.value}"
                )

        except DesktopAutomationError:
            raise
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to verify desktop element "
                f"'{locator.value}': {exc}"
            ) from exc

        return element

    def click_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> None:
        element = self._require_element(
            locator,
            window_id=window_id,
        )

        try:
            element.click()
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to click desktop element "
                f"'{locator.value}': {exc}"
            ) from exc

    def click_point(
        self,
        point: DesktopPoint,
    ) -> None:
        self._require_windows()

        try:
            from pywinauto import mouse

            mouse.click(
                coords=(
                    point.x,
                    point.y,
                )
            )
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "pywinauto is not installed."
            ) from exc
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to click screen point "
                f"({point.x}, {point.y}): {exc}"
            ) from exc

    @staticmethod
    def _escape_text(
        text: str,
    ) -> str:
        """
        Escape pywinauto send_keys syntax so plain user text remains text.

        pywinauto treats braces/modifiers specially, so escape them before
        passing arbitrary text to the keyboard layer.
        """
        escaped = text

        escaped = escaped.replace(
            "{",
            "{{}",
        )
        escaped = escaped.replace(
            "}",
            "{}}",
        )
        escaped = escaped.replace(
            "^",
            "{^}",
        )
        escaped = escaped.replace(
            "%",
            "{%}",
        )
        escaped = escaped.replace(
            "+",
            "{+}",
        )
        escaped = escaped.replace(
            "~",
            "{~}",
        )

        return escaped

    def type_text(
        self,
        text: str,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        if not text:
            return

        self._require_windows()

        try:
            from pywinauto import keyboard

            keyboard.send_keys(
                self._escape_text(text),
                with_spaces=True,
                with_tabs=True,
                with_newlines=True,
            )
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "pywinauto is not installed."
            ) from exc
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to type desktop text: {exc}"
            ) from exc

    @staticmethod
    def _special_key(
        key: str,
    ) -> str:
        mapping = {
            "enter": "ENTER",
            "return": "RETURN",
            "tab": "TAB",
            "escape": "ESC",
            "esc": "ESC",
            "backspace": "BACKSPACE",
            "back": "BACK",
            "delete": "DELETE",
            "del": "DELETE",
            "insert": "INSERT",
            "home": "HOME",
            "end": "END",
            "pageup": "PGUP",
            "pagedown": "PGDN",
            "up": "UP",
            "down": "DOWN",
            "left": "LEFT",
            "right": "RIGHT",
            "space": "SPACE",
            "f1": "F1",
            "f2": "F2",
            "f3": "F3",
            "f4": "F4",
            "f5": "F5",
            "f6": "F6",
            "f7": "F7",
            "f8": "F8",
            "f9": "F9",
            "f10": "F10",
            "f11": "F11",
            "f12": "F12",
        }

        normalized = key.strip().casefold()

        if normalized in mapping:
            return f"{{{mapping[normalized]}}}"

        if len(key) == 1:
            return key

        raise ValueError(
            f"Unsupported keyboard key: {key}"
        )

    def press(
        self,
        key: str,
    ) -> None:
        if not isinstance(key, str):
            raise TypeError(
                "key must be a string."
            )

        if not key.strip():
            raise ValueError(
                "key cannot be empty."
            )

        self._require_windows()

        try:
            from pywinauto import keyboard

            keyboard.send_keys(
                self._special_key(key)
            )
        except (ValueError, TypeError):
            raise
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "pywinauto is not installed."
            ) from exc
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to press key '{key}': {exc}"
            ) from exc

    def hotkey(
        self,
        *keys: str,
    ) -> None:
        if not keys:
            raise ValueError(
                "hotkey requires at least one key."
            )

        if any(
            not isinstance(key, str)
            or not key.strip()
            for key in keys
        ):
            raise ValueError(
                "hotkey keys must be non-empty strings."
            )

        self._require_windows()

        modifier_tokens = {
            "control": "VK_CONTROL",
            "ctrl": "VK_CONTROL",
            "shift": "VK_SHIFT",
            "alt": "VK_MENU",
            "menu": "VK_MENU",
            "win": "VK_LWIN",
            "windows": "VK_LWIN",
        }

        modifiers: list[str] = []
        regular: list[str] = []

        for key in keys:
            normalized = key.casefold()

            if normalized in modifier_tokens:
                modifiers.append(
                    modifier_tokens[normalized]
                )
            else:
                regular.append(
                    self._special_key(key)
                )

        if not regular:
            raise ValueError(
                "hotkey requires a non-modifier key."
            )

        try:
            from pywinauto import keyboard

            sequence = ""

            for modifier in modifiers:
                sequence += (
                    f"{{{modifier} down}}"
                )

            sequence += "".join(regular)

            for modifier in reversed(modifiers):
                sequence += (
                    f"{{{modifier} up}}"
                )

            keyboard.send_keys(sequence)

        except (ValueError, TypeError):
            raise
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "pywinauto is not installed."
            ) from exc
        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to execute desktop hotkey: {exc}"
            ) from exc

    def screenshot(self) -> bytes:
        self._require_windows()

        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise DesktopAutomationConnectionError(
                "Pillow is required for desktop screenshots. "
                "Run 'python -m pip install pillow'."
            ) from exc

        try:
            image = ImageGrab.grab(
                all_screens=(
                    self._config.screenshot_all_screens
                )
            )

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="PNG",
            )

            return buffer.getvalue()

        except Exception as exc:
            raise DesktopAutomationActionError(
                f"Failed to capture desktop screenshot: {exc}"
            ) from exc

    def health_check(self) -> bool:
        if sys.platform != "win32":
            return False

        try:
            self._load_pywin32_backend()
            return True
        except DesktopAutomationError:
            return False
