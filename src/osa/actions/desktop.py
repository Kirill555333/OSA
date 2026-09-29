"""Desktop adapter for the unified OSA action system."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.actions.router import ActionHandler
from osa.desktop.automation import (
    DesktopAutomationInterface,
    DesktopElementLocator,
    DesktopPoint,
)


class DesktopActionAdapterError(RuntimeError):
    """Raised when the desktop action adapter is misconfigured."""


DESKTOP_ACTION_NAMES = (
    "desktop_applications",
    "desktop_launch_application",
    "desktop_close_application",
    "desktop_windows",
    "desktop_activate_window",
    "desktop_close_window",
    "desktop_find_element",
    "desktop_click_element",
    "desktop_click_point",
    "desktop_type",
    "desktop_press",
    "desktop_hotkey",
    "desktop_screenshot",
    "desktop_health_check",
)


class DesktopActionAdapter(ActionHandler):
    """Adapt DesktopAutomationInterface to unified actions."""

    def __init__(
        self,
        backend: DesktopAutomationInterface,
    ) -> None:
        if not isinstance(
            backend,
            DesktopAutomationInterface,
        ):
            raise DesktopActionAdapterError(
                "backend must implement DesktopAutomationInterface."
            )

        self._backend = backend

    @property
    def backend(self) -> DesktopAutomationInterface:
        """Return the wrapped desktop backend."""
        return self._backend

    @property
    def supported_actions(self) -> tuple[str, ...]:
        """Return supported desktop action names."""
        return DESKTOP_ACTION_NAMES

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """Execute one desktop action."""
        if not isinstance(request, ActionRequest):
            raise DesktopActionAdapterError(
                "request must be an ActionRequest."
            )

        if request.kind is not ActionKind.DESKTOP:
            return ActionResult.failed(
                request.request_id,
                (
                    "DesktopActionAdapter received "
                    f"'{request.kind.value}' action."
                ),
                metadata={
                    "adapter_error": "wrong_action_kind",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        if request.name not in DESKTOP_ACTION_NAMES:
            return ActionResult.failed(
                request.request_id,
                f"Unsupported desktop action '{request.name}'.",
                metadata={
                    "adapter_error": "unsupported_action",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        try:
            result = self._dispatch(
                request.name,
                request.arguments,
            )
        except Exception as exc:
            return ActionResult.failed(
                request.request_id,
                f"Desktop action failed: {exc}",
                metadata={
                    "adapter_error": "backend_exception",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                    "exception_type": type(exc).__name__,
                },
            )

        return ActionResult.succeeded(
            request.request_id,
            output=result["output"],
            data=result["data"],
            metadata={
                "adapter": "desktop",
                "action_name": request.name,
            },
        )

    def dispatch(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """Dispatch is an alias for execute."""
        return self.execute(request)

    def _dispatch(
        self,
        action_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        if action_name == "desktop_applications":
            return self._format_value(
                self._backend.applications()
            )

        if action_name == "desktop_launch_application":
            command = self._required_string(
                arguments,
                "command",
            )
            return self._format_value(
                self._backend.launch_application(command)
            )

        if action_name == "desktop_close_application":
            name = self._required_string(
                arguments,
                "name",
            )
            self._backend.close_application(name)
            return self._empty_result(
                "Application closed."
            )

        if action_name == "desktop_windows":
            return self._format_value(
                self._backend.windows()
            )

        if action_name == "desktop_activate_window":
            title = self._required_string(
                arguments,
                "title",
            )
            return self._format_value(
                self._backend.activate_window(title)
            )

        if action_name == "desktop_close_window":
            title = self._required_string(
                arguments,
                "title",
            )
            self._backend.close_window(title)
            return self._empty_result(
                "Window closed."
            )

        if action_name == "desktop_find_element":
            locator = self._parse_locator(arguments)
            return self._format_value(
                self._backend.find_element(locator)
            )

        if action_name == "desktop_click_element":
            locator = self._parse_locator(arguments)
            self._backend.click_element(locator)
            return self._empty_result(
                "Element clicked."
            )

        if action_name == "desktop_click_point":
            point = self._parse_point(arguments)
            self._backend.click_point(point)
            return self._empty_result(
                f"Clicked point ({point.x}, {point.y})."
            )

        if action_name == "desktop_type":
            text = self._required_string(
                arguments,
                "text",
            )
            self._backend.type_text(text)
            return self._empty_result(
                "Text entered."
            )

        if action_name == "desktop_press":
            key = self._required_string(
                arguments,
                "key",
            )
            self._backend.press(key)
            return self._empty_result(
                f"Pressed {key}."
            )

        if action_name == "desktop_hotkey":
            keys = self._required_string_sequence(
                arguments,
                "keys",
            )
            self._backend.hotkey(*keys)
            return self._empty_result(
                f"Pressed hotkey: {'+'.join(keys)}."
            )

        if action_name == "desktop_screenshot":
            return self._format_value(
                self._backend.screenshot()
            )

        if action_name == "desktop_health_check":
            healthy = self._backend.health_check()

            if not isinstance(healthy, bool):
                raise DesktopActionAdapterError(
                    "Desktop health_check() must return a boolean."
                )

            return {
                "output": (
                    "Desktop backend is healthy."
                    if healthy
                    else "Desktop backend is not healthy."
                ),
                "data": {
                    "healthy": healthy,
                },
            }

        raise DesktopActionAdapterError(
            f"Unsupported desktop action '{action_name}'."
        )

    @staticmethod
    def _required_string(
        arguments: Mapping[str, Any],
        key: str,
    ) -> str:
        value = arguments.get(key)

        if not isinstance(value, str) or not value.strip():
            raise DesktopActionAdapterError(
                f"Argument '{key}' must be a non-empty string."
            )

        return value.strip()

    @staticmethod
    def _required_string_sequence(
        arguments: Mapping[str, Any],
        key: str,
    ) -> tuple[str, ...]:
        value = arguments.get(key)

        if not isinstance(value, (list, tuple)):
            raise DesktopActionAdapterError(
                f"Argument '{key}' must be a sequence of strings."
            )

        result = tuple(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )

        if not result:
            raise DesktopActionAdapterError(
                f"Argument '{key}' must contain at least one key."
            )

        if len(result) != len(value):
            raise DesktopActionAdapterError(
                f"Argument '{key}' must contain only non-empty strings."
            )

        return result

    @staticmethod
    def _parse_point(
        arguments: Mapping[str, Any],
    ) -> DesktopPoint:
        x = arguments.get("x")
        y = arguments.get("y")

        if not isinstance(x, int) or isinstance(x, bool):
            raise DesktopActionAdapterError(
                "Argument 'x' must be an integer."
            )

        if not isinstance(y, int) or isinstance(y, bool):
            raise DesktopActionAdapterError(
                "Argument 'y' must be an integer."
            )

        return DesktopPoint(
            x=x,
            y=y,
        )

    @staticmethod
    def _parse_locator(
        arguments: Mapping[str, Any],
    ) -> DesktopElementLocator:
        kind = arguments.get("kind")
        value = arguments.get("value")

        if not isinstance(kind, str) or not kind.strip():
            raise DesktopActionAdapterError(
                "Argument 'kind' must be a non-empty string."
            )

        if not isinstance(value, str) or not value.strip():
            raise DesktopActionAdapterError(
                "Argument 'value' must be a non-empty string."
            )

        try:
            return DesktopElementLocator(
                kind=kind.strip(),
                value=value.strip(),
            )
        except (TypeError, ValueError) as exc:
            raise DesktopActionAdapterError(
                f"Invalid desktop element locator: {exc}"
            ) from exc

    @staticmethod
    def _empty_result(output: str) -> dict[str, Any]:
        return {
            "output": output,
            "data": {},
        }

    @classmethod
    def _format_value(
        cls,
        value: Any,
    ) -> dict[str, Any]:
        normalized = cls._normalize_value(value)

        if isinstance(normalized,list):            output = "\n".join(
                str(item)
                for item in normalized
            )
        else:
            output = str(normalized)

        return {
            "output": output,
            "data": {
                "value": normalized,
            },
        }

    @classmethod
    def _normalize_value(
        cls,
        value: Any,
    ) -> Any:
        if is_dataclass(value):
            return cls._normalize_value(
                asdict(value)
            )

        if isinstance(value, Mapping):
            return {
                str(key): cls._normalize_value(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                cls._normalize_value(item)
                for item in value
            ]

        if isinstance(value, set):
            return [
                cls._normalize_value(item)
                for item in sorted(value, key=str)
            ]

        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"

        if isinstance(value, Path):
            return str(value)

        return value
