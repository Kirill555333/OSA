from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class BrowserAutomationError(RuntimeError):
    """Base error for interactive browser automation."""


class BrowserAutomationConnectionError(BrowserAutomationError):
    """Raised when the automation backend cannot connect."""


class BrowserAutomationActionError(BrowserAutomationError):
    """Raised when an interactive browser action fails."""


BrowserLocatorKind = Literal[
    "css",
    "role",
    "text",
    "label",
    "placeholder",
    "test_id",
]


@dataclass(frozen=True)
class BrowserLocator:
    """
    Backend-neutral way to identify a page element.

    The selector strategy mirrors the locator concepts supported by modern
    browser automation engines while keeping the OSA API independent of them.
    """

    kind: BrowserLocatorKind
    value: str
    exact: bool = False

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("Browser locator value cannot be empty.")


@dataclass(frozen=True)
class BrowserTab:
    """Current browser tab state."""

    tab_id: str
    url: str
    title: str = ""


@dataclass(frozen=True)
class BrowserElement:
    """Observed browser element."""

    locator: BrowserLocator
    text: str = ""
    role: str | None = None
    tag_name: str | None = None
    visible: bool = True
    enabled: bool = True
    name: str | None = None


@dataclass(frozen=True)
class BrowserPageObservation:
    """Structured observation of the active browser page."""

    tab: BrowserTab
    text: str
    elements: tuple[BrowserElement, ...] = ()
    text_truncated: bool = False
    elements_truncated: bool = False


BrowserWaitState = Literal[
    "attached",
    "detached",
    "visible",
    "hidden",
]


class BrowserAutomationInterface(ABC):
    """
    Backend-neutral interface for a real interactive browser.

    HttpBrowser remains responsible for direct HTTP fetching.
    Implementations of this interface drive a graphical browser.
    """

    @abstractmethod
    def open(self, url: str) -> BrowserTab:
        """Navigate the active page to a URL."""

    @abstractmethod
    def current_tab(self) -> BrowserTab:
        """Return the active tab."""

    @abstractmethod
    def tabs(self) -> tuple[BrowserTab, ...]:
        """Return all open tabs."""

    @abstractmethod
    def close_tab(self, tab_id: str) -> None:
        """Close a browser tab."""

    @abstractmethod
    def find(
        self,
        locator: BrowserLocator,
    ) -> BrowserElement | None:
        """Find and observe an element."""

    @abstractmethod
    def click(
        self,
        locator: BrowserLocator,
    ) -> None:
        """Click an element."""

    @abstractmethod
    def type_text(
        self,
        locator: BrowserLocator,
        text: str,
        *,
        clear: bool = True,
    ) -> None:
        """Fill or type into an editable element."""

    @abstractmethod
    def press(
        self,
        locator: BrowserLocator,
        key: str,
    ) -> None:
        """Press a keyboard key on an element."""

    @abstractmethod
    def get_text(
        self,
        locator: BrowserLocator | None = None,
    ) -> str:
        """Read visible text from the page or selected element."""

    @abstractmethod
    def observe(
        self,
        *,
        max_text_characters: int = 12_000,
        max_elements: int = 100,
    ) -> BrowserPageObservation:
        """Return a bounded structured observation of the active page."""

    @abstractmethod
    def observe(
        self,
        *,
        max_text_characters: int = 12_000,
        max_elements: int = 100,
    ) -> BrowserPageObservation:
        if max_text_characters <= 0:
            raise ValueError(
                "max_text_characters must be greater than zero."
            )

        if max_elements <= 0:
            raise ValueError(
                "max_elements must be greater than zero."
            )

        tab = self.current_tab()
        text = self.get_text()

        return BrowserPageObservation(
            tab=tab,
            text=text[:max_text_characters],
            elements=(),
            text_truncated=len(text) > max_text_characters,
            elements_truncated=False,
        )

    def wait_for(
        self,
        locator: BrowserLocator,
        *,
        state: BrowserWaitState = "visible",
        timeout_ms: int | None = None,
    ) -> None:
        """Wait for an element to reach the requested state."""

    @abstractmethod
    def go_back(self) -> BrowserTab:
        """Navigate the active tab backward in history."""

    @abstractmethod
    def go_forward(self) -> BrowserTab:
        """Navigate the active tab forward in history."""

    @abstractmethod
    def reload(self) -> BrowserTab:
        """Reload the active tab."""

    @abstractmethod
    def screenshot(self) -> bytes:
        """Capture the active page as image bytes."""

    @abstractmethod
    def evaluate(self, script: str) -> object:
        """Evaluate backend-supported JavaScript."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the automation backend is available."""


@dataclass
class FakeBrowserAutomation(BrowserAutomationInterface):
    """
    Deterministic test backend.

    Used by unit tests so the browser contract can be tested without
    launching a graphical browser.
    """

    _tabs: dict[str, BrowserTab]
    _active_tab_id: str | None = None
    actions: list[tuple[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.actions is None:
            self.actions = []

    def open(self, url: str) -> BrowserTab:
        if not url.strip():
            raise ValueError("url cannot be empty.")

        tab_id = f"tab-{len(self._tabs) + 1}"

        tab = BrowserTab(
            tab_id=tab_id,
            url=url,
            title="",
        )

        self._tabs[tab_id] = tab
        self._active_tab_id = tab_id

        self.actions.append(("open", url))

        return tab

    def current_tab(self) -> BrowserTab:
        if self._active_tab_id is None:
            raise BrowserAutomationError(
                "No active browser tab."
            )

        return self._tabs[self._active_tab_id]

    def tabs(self) -> tuple[BrowserTab, ...]:
        return tuple(self._tabs.values())

    def close_tab(self, tab_id: str) -> None:
        if tab_id not in self._tabs:
            raise BrowserAutomationError(
                f"Unknown browser tab: {tab_id}"
            )

        del self._tabs[tab_id]

        if self._active_tab_id == tab_id:
            self._active_tab_id = next(
                iter(self._tabs),
                None,
            )

        self.actions.append(("close_tab", tab_id))

    def find(
        self,
        locator: BrowserLocator,
    ) -> BrowserElement | None:
        self.actions.append(("find", locator))

        return BrowserElement(
            locator=locator,
            text="fake element",
            role=(
                locator.value
                if locator.kind == "role"
                else None
            ),
            tag_name="div",
        )

    def _require_element(
        self,
        locator: BrowserLocator,
    ) -> BrowserElement:
        element = self.find(locator)

        if element is None:
            raise BrowserAutomationActionError(
                f"Element not found: {locator.value}"
            )

        if not element.visible:
            raise BrowserAutomationActionError(
                f"Element is not visible: {locator.value}"
            )

        if not element.enabled:
            raise BrowserAutomationActionError(
                f"Element is not enabled: {locator.value}"
            )

        return element

    def click(
        self,
        locator: BrowserLocator,
    ) -> None:
        self._require_element(locator)
        self.actions.append(("click", locator))

    def type_text(
        self,
        locator: BrowserLocator,
        text: str,
        *,
        clear: bool = True,
    ) -> None:
        self._require_element(locator)

        self.actions.append(
            (
                "type_text",
                {
                    "locator": locator,
                    "text": text,
                    "clear": clear,
                },
            )
        )

    def press(
        self,
        locator: BrowserLocator,
        key: str,
    ) -> None:
        if not key.strip():
            raise ValueError("key cannot be empty.")

        self._require_element(locator)

        self.actions.append(
            (
                "press",
                {
                    "locator": locator,
                    "key": key,
                },
            )
        )

    def get_text(
        self,
        locator: BrowserLocator | None = None,
    ) -> str:
        if locator is not None:
            self._require_element(locator)

        self.actions.append(("get_text", locator))
        return "fake page text"

    def observe(
        self,
        *,
        max_text_characters: int = 12_000,
        max_elements: int = 100,
    ) -> BrowserPageObservation:
        if max_text_characters <= 0:
            raise ValueError(
                "max_text_characters must be greater than zero."
            )

        if max_elements <= 0:
            raise ValueError(
                "max_elements must be greater than zero."
            )

        tab = self.current_tab()
        text = self.get_text()

        return BrowserPageObservation(
            tab=tab,
            text=text[:max_text_characters],
            elements=(),
            text_truncated=len(text) > max_text_characters,
            elements_truncated=False,
        )

    def wait_for(
        self,
        locator: BrowserLocator,
        *,
        state: BrowserWaitState = "visible",
        timeout_ms: int | None = None,
    ) -> None:
        if timeout_ms is not None and timeout_ms < 0:
            raise ValueError(
                "timeout_ms cannot be negative."
            )

        self._require_element(locator)

        self.actions.append(
            (
                "wait_for",
                {
                    "locator": locator,
                    "state": state,
                    "timeout_ms": timeout_ms,
                },
            )
        )

    def go_back(self) -> BrowserTab:
        tab = self.current_tab()
        self.actions.append(("go_back", None))
        return tab

    def go_forward(self) -> BrowserTab:
        tab = self.current_tab()
        self.actions.append(("go_forward", None))
        return tab

    def reload(self) -> BrowserTab:
        tab = self.current_tab()
        self.actions.append(("reload", None))
        return tab

    def screenshot(self) -> bytes:
        self.actions.append(("screenshot", None))
        return b"fake-screenshot"

    def evaluate(self, script: str) -> object:
        if not script.strip():
            raise ValueError(
                "script cannot be empty."
            )

        self.actions.append(("evaluate", script))
        return None

    def health_check(self) -> bool:
        return True


def create_fake_browser_automation() -> FakeBrowserAutomation:
    """Create a deterministic browser backend for tests."""
    return FakeBrowserAutomation(
        _tabs={},
    )
