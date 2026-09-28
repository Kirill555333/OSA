from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from osa.browser.automation import (
    BrowserAutomationActionError,
    BrowserAutomationConnectionError,
    BrowserAutomationError,
    BrowserAutomationInterface,
    BrowserElement,
    BrowserLocator,
    BrowserPageObservation,
    BrowserTab,
    BrowserWaitState,
)


@dataclass(frozen=True)
class PlaywrightBrowserConfig:
    """Configuration for the Playwright browser backend."""

    headless: bool = True
    browser_name: str = "chromium"
    navigation_timeout_ms: int = 30_000
    action_timeout_ms: int = 10_000
    screenshot_full_page: bool = False

    def __post_init__(self) -> None:
        if self.browser_name not in {
            "chromium",
            "firefox",
            "webkit",
        }:
            raise ValueError(
                "browser_name must be chromium, firefox, or webkit."
            )

        if self.navigation_timeout_ms <= 0:
            raise ValueError(
                "navigation_timeout_ms must be greater than zero."
            )

        if self.action_timeout_ms <= 0:
            raise ValueError(
                "action_timeout_ms must be greater than zero."
            )


class PlaywrightBrowser(BrowserAutomationInterface):
    """
    Real interactive browser backend backed by Playwright sync API.

    The Playwright dependency is imported lazily, so importing OSA and running
    unit tests does not require the browser process to be launched.
    """

    def __init__(
        self,
        *,
        config: PlaywrightBrowserConfig | None = None,
        driver_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config or PlaywrightBrowserConfig()
        self._driver_factory = (
            driver_factory
            or self._default_driver_factory
        )

        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None

        self._pages: dict[str, Any] = {}
        self._page_ids: dict[int, str] = {}
        self._active_page_id: str | None = None
        self._next_tab_number = 1

    @staticmethod
    def _default_driver_factory() -> Any:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserAutomationConnectionError(
                "Playwright is not installed. "
                "Run 'python -m pip install playwright'."
            ) from exc

        return sync_playwright()

    @property
    def config(self) -> PlaywrightBrowserConfig:
        """Return backend configuration."""
        return self._config

    def _ensure_started(self) -> None:
        if (
            self._playwright is not None
            and self._browser is not None
            and self._context is not None
        ):
            return

        try:
            self._playwright = self._driver_factory().start()

            browser_type = getattr(
                self._playwright,
                self._config.browser_name,
            )

            self._browser = browser_type.launch(
                headless=self._config.headless,
            )

            self._context = self._browser.new_context()

            self._context.set_default_navigation_timeout(
                self._config.navigation_timeout_ms
            )
            self._context.set_default_timeout(
                self._config.action_timeout_ms
            )

        except BrowserAutomationError:
            raise
        except Exception as exc:
            self._cleanup_runtime()

            raise BrowserAutomationConnectionError(
                f"Failed to start Playwright browser: {exc}"
            ) from exc

    def _cleanup_runtime(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        except Exception:
            pass

        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass

        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass

        self._context = None
        self._browser = None
        self._playwright = None
        self._pages.clear()
        self._page_ids.clear()
        self._active_page_id = None

    def close(self) -> None:
        """Close the Playwright browser and release all resources."""
        self._cleanup_runtime()

    def _require_context(self) -> Any:
        self._ensure_started()

        if self._context is None:
            raise BrowserAutomationConnectionError(
                "Playwright browser context is unavailable."
            )

        return self._context

    def _register_page(self, page: Any) -> str:
        page_key = id(page)

        existing_id = self._page_ids.get(page_key)

        if existing_id is not None:
            return existing_id

        tab_id = f"tab-{self._next_tab_number}"
        self._next_tab_number += 1

        self._pages[tab_id] = page
        self._page_ids[page_key] = tab_id

        return tab_id

    def _remove_page(self, tab_id: str) -> None:
        page = self._pages.pop(tab_id, None)

        if page is not None:
            self._page_ids.pop(id(page), None)

        if self._active_page_id == tab_id:
            self._active_page_id = (
                next(iter(self._pages), None)
            )

    def _require_active_page(self) -> Any:
        self._require_context()

        if self._active_page_id is None:
            raise BrowserAutomationError(
                "No active browser tab."
            )

        page = self._pages.get(self._active_page_id)

        if page is None:
            raise BrowserAutomationError(
                "Active browser tab no longer exists."
            )

        return page

    @staticmethod
    def _tab_from_page(
        tab_id: str,
        page: Any,
    ) -> BrowserTab:
        try:
            url = str(page.url)
            title = str(page.title())
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to read browser tab state: {exc}"
            ) from exc

        return BrowserTab(
            tab_id=tab_id,
            url=url,
            title=title,
        )

    def open(self, url: str) -> BrowserTab:
        if not url.strip():
            raise ValueError("url cannot be empty.")

        context = self._require_context()

        try:
            page = context.new_page()
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self._config.navigation_timeout_ms,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to open URL '{url}': {exc}"
            ) from exc

        tab_id = self._register_page(page)
        self._active_page_id = tab_id

        return self._tab_from_page(
            tab_id,
            page,
        )

    def current_tab(self) -> BrowserTab:
        page = self._require_active_page()

        assert self._active_page_id is not None

        return self._tab_from_page(
            self._active_page_id,
            page,
        )

    def tabs(self) -> tuple[BrowserTab, ...]:
        context = self._require_context()

        for page in context.pages:
            self._register_page(page)

        return tuple(
            self._tab_from_page(tab_id, page)
            for tab_id, page in self._pages.items()
        )

    def close_tab(self, tab_id: str) -> None:
        page = self._pages.get(tab_id)

        if page is None:
            raise BrowserAutomationError(
                f"Unknown browser tab: {tab_id}"
            )

        try:
            page.close()
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to close tab '{tab_id}': {exc}"
            ) from exc
        finally:
            self._remove_page(tab_id)

    def _to_playwright_locator(
        self,
        page: Any,
        locator: BrowserLocator,
    ) -> Any:
        value = locator.value

        try:
            if locator.kind == "css":
                return page.locator(value)

            if locator.kind == "role":
                return page.get_by_role(
                    value,
                    exact=locator.exact,
                )

            if locator.kind == "text":
                return page.get_by_text(
                    value,
                    exact=locator.exact,
                )

            if locator.kind == "label":
                return page.get_by_label(
                    value,
                    exact=locator.exact,
                )

            if locator.kind == "placeholder":
                return page.get_by_placeholder(
                    value,
                    exact=locator.exact,
                )

            if locator.kind == "test_id":
                return page.get_by_test_id(value)

        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to create locator '{value}': {exc}"
            ) from exc

        raise BrowserAutomationActionError(
            f"Unsupported locator kind: {locator.kind}"
        )

    def _resolve_action_locator(
        self,
        page: Any,
        locator: BrowserLocator,
    ) -> Any:
        playwright_locator = self._to_playwright_locator(
            page,
            locator,
        )

        try:
            count = playwright_locator.count()
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to inspect element '{locator.value}': {exc}"
            ) from exc

        if count == 0:
            raise BrowserAutomationActionError(
                f"Element not found: {locator.value}"
            )

        if count > 1:
            raise BrowserAutomationActionError(
                f"Locator is ambiguous and matched {count} "
                f"elements: {locator.value}"
            )

        return playwright_locator

    def find(
        self,
        locator: BrowserLocator,
    ) -> BrowserElement | None:
        page = self._require_active_page()
        playwright_locator = self._to_playwright_locator(
            page,
            locator,
        )

        try:
            count = playwright_locator.count()

            if count == 0:
                return None

            target = playwright_locator.first

            return BrowserElement(
                locator=locator,
                text=target.inner_text(),
                role=target.get_attribute("role"),
                tag_name=str(
                    target.evaluate(
                        "element => element.tagName"
                    )
                ).lower(),
                visible=target.is_visible(),
                enabled=target.is_enabled(),
            )

        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to inspect element '{locator.value}': {exc}"
            ) from exc

    def click(
        self,
        locator: BrowserLocator,
    ) -> None:
        page = self._require_active_page()
        target = self._resolve_action_locator(
            page,
            locator,
        )

        try:
            target.click(
                timeout=self._config.action_timeout_ms,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to click '{locator.value}': {exc}"
            ) from exc

    def type_text(
        self,
        locator: BrowserLocator,
        text: str,
        *,
        clear: bool = True,
    ) -> None:
        page = self._require_active_page()
        target = self._resolve_action_locator(
            page,
            locator,
        )

        try:
            if clear:
                target.fill(
                    text,
                    timeout=self._config.action_timeout_ms,
                )
            else:
                target.press_sequentially(
                    text,
                    timeout=self._config.action_timeout_ms,
                )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to type into '{locator.value}': {exc}"
            ) from exc

    def press(
        self,
        locator: BrowserLocator,
        key: str,
    ) -> None:
        if not key.strip():
            raise ValueError("key cannot be empty.")

        page = self._require_active_page()
        target = self._resolve_action_locator(
            page,
            locator,
        )

        try:
            target.press(
                key,
                timeout=self._config.action_timeout_ms,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to press '{key}' on '{locator.value}': {exc}"
            ) from exc

    def get_text(
        self,
        locator: BrowserLocator | None = None,
    ) -> str:
        page = self._require_active_page()

        try:
            if locator is None:
                return page.locator("body").inner_text()

            target = self._resolve_action_locator(
                page,
                locator,
            )

            return target.inner_text()

        except BrowserAutomationError:
            raise
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to read page text: {exc}"
            ) from exc

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

        page = self._require_active_page()

        try:
            tab = self.current_tab()

            raw_text = page.locator("body").inner_text()
            text = raw_text[:max_text_characters]

            interactive = page.locator(
                "a[href], button, input, textarea, select, "
                "[role], [contenteditable='true']"
            )

            total = interactive.count()
            elements_truncated = total > max_elements

            elements: list[BrowserElement] = []

            for index in range(
                min(total, max_elements)
            ):
                target = interactive.nth(index)

                try:
                    if not target.is_visible():
                        continue

                    tag_name = str(
                        target.evaluate(
                            "element => element.tagName"
                        )
                    ).lower()

                    role = target.get_attribute("role")
                    aria_label = target.get_attribute(
                        "aria-label"
                    )
                    title = target.get_attribute("title")
                    placeholder = target.get_attribute(
                        "placeholder"
                    )

                    try:
                        element_text = target.inner_text()
                    except Exception:
                        element_text = ""

                    name = (
                        aria_label
                        or title
                        or placeholder
                        or element_text
                        or None
                    )

                    locator = BrowserLocator(
                        kind="css",
                        value=(
                            f"{tag_name}:nth-of-type({index + 1})"
                        ),
                    )

                    elements.append(
                        BrowserElement(
                            locator=locator,
                            text=element_text,
                            role=role,
                            tag_name=tag_name,
                            visible=True,
                            enabled=target.is_enabled(),
                            name=name,
                        )
                    )

                except Exception:
                    continue

            return BrowserPageObservation(
                tab=tab,
                text=text,
                elements=tuple(elements),
                text_truncated=(
                    len(raw_text) > max_text_characters
                ),
                elements_truncated=elements_truncated,
            )

        except BrowserAutomationError:
            raise
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to observe browser page: {exc}"
            ) from exc

    def wait_for(
        self,
        locator: BrowserLocator,
        *,
        state: BrowserWaitState = "visible",
        timeout_ms: int | None = None,
    ) -> None:
        page = self._require_active_page()
        playwright_locator = self._to_playwright_locator(
            page,
            locator,
        )

        timeout = (
            timeout_ms
            if timeout_ms is not None
            else self._config.action_timeout_ms
        )

        if timeout < 0:
            raise ValueError(
                "timeout_ms cannot be negative."
            )

        try:
            playwright_locator.wait_for(
                state=state,
                timeout=timeout,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed waiting for '{locator.value}': {exc}"
            ) from exc

    def go_back(self) -> BrowserTab:
        page = self._require_active_page()

        try:
            page.go_back(
                wait_until="domcontentloaded",
                timeout=self._config.navigation_timeout_ms,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to go back: {exc}"
            ) from exc

        assert self._active_page_id is not None

        return self._tab_from_page(
            self._active_page_id,
            page,
        )

    def go_forward(self) -> BrowserTab:
        page = self._require_active_page()

        try:
            page.go_forward(
                wait_until="domcontentloaded",
                timeout=self._config.navigation_timeout_ms,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to go forward: {exc}"
            ) from exc

        assert self._active_page_id is not None

        return self._tab_from_page(
            self._active_page_id,
            page,
        )

    def reload(self) -> BrowserTab:
        page = self._require_active_page()

        try:
            page.reload(
                wait_until="domcontentloaded",
                timeout=self._config.navigation_timeout_ms,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to reload page: {exc}"
            ) from exc

        assert self._active_page_id is not None

        return self._tab_from_page(
            self._active_page_id,
            page,
        )

    def screenshot(self) -> bytes:
        page = self._require_active_page()

        try:
            return page.screenshot(
                full_page=self._config.screenshot_full_page,
            )
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to capture screenshot: {exc}"
            ) from exc

    def evaluate(
        self,
        script: str,
    ) -> object:
        if not script.strip():
            raise ValueError(
                "script cannot be empty."
            )

        page = self._require_active_page()

        try:
            return page.evaluate(script)
        except Exception as exc:
            raise BrowserAutomationActionError(
                f"Failed to evaluate script: {exc}"
            ) from exc

    def health_check(self) -> bool:
        try:
            self._ensure_started()
            return (
                self._browser is not None
                and self._context is not None
            )
        except BrowserAutomationError:
            return False
