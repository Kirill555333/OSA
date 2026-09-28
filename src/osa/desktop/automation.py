"""Backend-neutral desktop automation contracts for OSA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class DesktopAutomationError(RuntimeError):
    """Base error for desktop automation."""


class DesktopAutomationConnectionError(DesktopAutomationError):
    """Raised when the desktop backend cannot connect."""


class DesktopAutomationActionError(DesktopAutomationError):
    """Raised when a desktop action fails."""


@dataclass(frozen=True, slots=True)
class DesktopPoint:
    """Screen coordinate."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class DesktopRect:
    """Screen-space rectangle."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError(
                "Desktop rectangle width cannot be negative."
            )

        if self.height < 0:
            raise ValueError(
                "Desktop rectangle height cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class DesktopApplication:
    """Running or launchable desktop application."""

    application_id: str
    name: str
    path: str | None = None
    running: bool = False


@dataclass(frozen=True, slots=True)
class DesktopWindow:
    """Observed desktop window."""

    window_id: str
    title: str
    application_id: str | None = None
    bounds: DesktopRect | None = None
    visible: bool = True
    focused: bool = False
    minimized: bool = False


DesktopElementKind = Literal[
    "name",
    "role",
    "automation_id",
]


@dataclass(frozen=True, slots=True)
class DesktopElementLocator:
    """
    Backend-neutral desktop UI element locator.

    Backends may map these fields to native accessibility/UI automation APIs.
    """

    kind: DesktopElementKind
    value: str
    exact: bool = False

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError(
                "Desktop element locator value cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class DesktopElement:
    """Observed desktop UI element."""

    locator: DesktopElementLocator
    name: str | None = None
    role: str | None = None
    bounds: DesktopRect | None = None
    visible: bool = True
    enabled: bool = True


class DesktopAutomationInterface(ABC):
    """
    Stable interface for interactive desktop automation.

    Concrete implementations must remain platform-specific and must not leak
    platform APIs into OSA core.
    """

    @abstractmethod
    def applications(self) -> tuple[DesktopApplication, ...]:
        """Return visible/running desktop applications."""

    @abstractmethod
    def launch_application(
        self,
        application: str,
    ) -> DesktopApplication:
        """Launch or activate an application."""

    @abstractmethod
    def close_application(
        self,
        application_id: str,
    ) -> None:
        """Close an application."""

    @abstractmethod
    def windows(self) -> tuple[DesktopWindow, ...]:
        """Return accessible desktop windows."""

    @abstractmethod
    def activate_window(
        self,
        window_id: str,
    ) -> DesktopWindow:
        """Activate and focus a desktop window."""

    @abstractmethod
    def close_window(
        self,
        window_id: str,
    ) -> None:
        """Close a desktop window."""

    @abstractmethod
    def find_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> DesktopElement | None:
        """Find and inspect a desktop UI element."""

    @abstractmethod
    def click_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> None:
        """Click a desktop UI element."""

    @abstractmethod
    def click_point(
        self,
        point: DesktopPoint,
    ) -> None:
        """Click a screen coordinate."""

    @abstractmethod
    def type_text(
        self,
        text: str,
    ) -> None:
        """Type text into the currently focused control."""

    @abstractmethod
    def press(
        self,
        key: str,
    ) -> None:
        """Press a keyboard key."""

    @abstractmethod
    def hotkey(
        self,
        *keys: str,
    ) -> None:
        """Press a keyboard shortcut."""

    @abstractmethod
    def screenshot(self) -> bytes:
        """Capture the current desktop."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the desktop backend is available."""


class FakeDesktopAutomation(DesktopAutomationInterface):
    """Deterministic desktop backend for unit tests."""

    def __init__(self) -> None:
        self._applications: dict[
            str,
            DesktopApplication,
        ] = {}

        self._windows: dict[
            str,
            DesktopWindow,
        ] = {}

        self._elements: dict[
            tuple[str | None, DesktopElementLocator],
            DesktopElement,
        ] = {}

        self.actions: list[tuple[str, object]] = []

        self._next_application_id = 1
        self._next_window_id = 1

    def applications(
        self,
    ) -> tuple[DesktopApplication, ...]:
        return tuple(self._applications.values())

    def launch_application(
        self,
        application: str,
    ) -> DesktopApplication:
        if not application.strip():
            raise ValueError(
                "application cannot be empty."
            )

        existing = next(
            (
                item
                for item in self._applications.values()
                if item.name == application
            ),
            None,
        )

        if existing is not None:
            updated = DesktopApplication(
                application_id=existing.application_id,
                name=existing.name,
                path=existing.path,
                running=True,
            )

            self._applications[
                existing.application_id
            ] = updated

            self.actions.append(
                ("launch_application", application)
            )

            return updated

        application_id = (
            f"app-{self._next_application_id}"
        )
        self._next_application_id += 1

        item = DesktopApplication(
            application_id=application_id,
            name=application,
            running=True,
        )

        self._applications[application_id] = item

        self.actions.append(
            ("launch_application", application)
        )

        return item

    def close_application(
        self,
        application_id: str,
    ) -> None:
        application = self._applications.get(
            application_id
        )

        if application is None:
            raise DesktopAutomationError(
                f"Unknown application: {application_id}"
            )

        self._applications[
            application_id
        ] = DesktopApplication(
            application_id=application.application_id,
            name=application.name,
            path=application.path,
            running=False,
        )

        self.actions.append(
            ("close_application", application_id)
        )

    def windows(
        self,
    ) -> tuple[DesktopWindow, ...]:
        return tuple(self._windows.values())

    def activate_window(
        self,
        window_id: str,
    ) -> DesktopWindow:
        if window_id not in self._windows:
            raise DesktopAutomationError(
                f"Unknown window: {window_id}"
            )

        updated: dict[
            str,
            DesktopWindow,
        ] = {}

        for item_id, window in self._windows.items():
            updated[item_id] = DesktopWindow(
                window_id=window.window_id,
                title=window.title,
                application_id=window.application_id,
                bounds=window.bounds,
                visible=window.visible,
                focused=item_id == window_id,
                minimized=False
                if item_id == window_id
                else window.minimized,
            )

        self._windows = updated

        result = self._windows[window_id]

        self.actions.append(
            ("activate_window", window_id)
        )

        return result

    def close_window(
        self,
        window_id: str,
    ) -> None:
        if window_id not in self._windows:
            raise DesktopAutomationError(
                f"Unknown window: {window_id}"
            )

        del self._windows[window_id]

        self.actions.append(
            ("close_window", window_id)
        )

    def find_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> DesktopElement | None:
        self.actions.append(
            (
                "find_element",
                {
                    "locator": locator,
                    "window_id": window_id,
                },
            )
        )

        return self._elements.get(
            (
                window_id,
                locator,
            )
        )

    def click_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
    ) -> None:
        element = self.find_element(
            locator,
            window_id=window_id,
        )

        if element is None:
            raise DesktopAutomationActionError(
                "Desktop element not found: "
                f"{locator.kind}={locator.value}"
            )

        if not element.visible:
            raise DesktopAutomationActionError(
                "Desktop element is not visible: "
                f"{locator.value}"
            )

        if not element.enabled:
            raise DesktopAutomationActionError(
                "Desktop element is not enabled: "
                f"{locator.value}"
            )

        self.actions.append(
            (
                "click_element",
                {
                    "locator": locator,
                    "window_id": window_id,
                },
            )
        )

    def click_point(
        self,
        point: DesktopPoint,
    ) -> None:
        self.actions.append(
            ("click_point", point)
        )

    def type_text(
        self,
        text: str,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        self.actions.append(
            ("type_text", text)
        )

    def press(
        self,
        key: str,
    ) -> None:
        if not key.strip():
            raise ValueError(
                "key cannot be empty."
            )

        self.actions.append(
            ("press", key)
        )

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

        self.actions.append(
            ("hotkey", tuple(keys))
        )

    def screenshot(self) -> bytes:
        self.actions.append(
            ("screenshot", None)
        )
        return b"fake-desktop-screenshot"

    def health_check(self) -> bool:
        return True

    def add_window(
        self,
        *,
        title: str,
        application_id: str | None = None,
        bounds: DesktopRect | None = None,
        visible: bool = True,
    ) -> DesktopWindow:
        """Add a deterministic test window."""
        if not title.strip():
            raise ValueError(
                "title cannot be empty."
            )

        window_id = (
            f"window-{self._next_window_id}"
        )
        self._next_window_id += 1

        window = DesktopWindow(
            window_id=window_id,
            title=title,
            application_id=application_id,
            bounds=bounds,
            visible=visible,
        )

        self._windows[window_id] = window

        return window

    def add_element(
        self,
        locator: DesktopElementLocator,
        *,
        window_id: str | None = None,
        name: str | None = None,
        role: str | None = None,
        bounds: DesktopRect | None = None,
        visible: bool = True,
        enabled: bool = True,
    ) -> DesktopElement:
        """Add a deterministic test UI element."""
        element = DesktopElement(
            locator=locator,
            name=name,
            role=role,
            bounds=bounds,
            visible=visible,
            enabled=enabled,
        )

        self._elements[
            (
                window_id,
                locator,
            )
        ] = element

        return element


def create_fake_desktop_automation() -> FakeDesktopAutomation:
    """Create a deterministic desktop backend for tests."""
    return FakeDesktopAutomation()
