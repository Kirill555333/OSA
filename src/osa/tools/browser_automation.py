"""Interactive browser action tools for OSA."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse

from osa.browser import (
    BrowserAutomationError,
    BrowserAutomationInterface,
    BrowserElement,
    BrowserLocator,
    BrowserPageObservation,
    BrowserSafetyError,
    BrowserSafetyPolicy,
    BrowserTab,
)
from osa.tools.registry import (
    ToolInterface,
    ToolResult,
)


def _locator_schema() -> dict[str, Any]:
    """Return the common browser locator JSON schema."""
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "css",
                    "role",
                    "text",
                    "label",
                    "placeholder",
                    "test_id",
                ],
                "description": "Element locator strategy.",
            },
            "value": {
                "type": "string",
                "description": "Value used by the locator.",
            },
            "exact": {
                "type": "boolean",
                "description": (
                    "Require an exact match where supported."
                ),
                "default": False,
            },
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }


def _parse_locator(
    arguments: Mapping[str, Any],
) -> BrowserLocator:
    """Validate tool arguments and construct a BrowserLocator."""
    raw = arguments.get("locator")

    if not isinstance(raw, Mapping):
        raise ValueError(
            "Argument 'locator' must be an object."
        )

    kind = raw.get("kind")
    value = raw.get("value")
    exact = raw.get("exact", False)

    if not isinstance(kind, str):
        raise ValueError(
            "Locator field 'kind' must be a string."
        )

    if not isinstance(value, str):
        raise ValueError(
            "Locator field 'value' must be a string."
        )

    if not isinstance(exact, bool):
        raise ValueError(
            "Locator field 'exact' must be a boolean."
        )

    allowed_kinds = {
        "css",
        "role",
        "text",
        "label",
        "placeholder",
        "test_id",
    }

    if kind not in allowed_kinds:
        raise ValueError(
            f"Unsupported locator kind: {kind}"
        )

    return BrowserLocator(
        kind=kind,
        value=value,
        exact=exact,
    )


def _tab_output(tab: BrowserTab) -> str:
    """Format browser tab information."""
    return (
        f"Tab ID: {tab.tab_id}\n"
        f"URL: {tab.url}\n"
        f"Title: {tab.title or '(untitled)'}"
    )


def _element_output(element: BrowserElement) -> str:
    """Format browser element information."""
    return (
        f"Locator kind: {element.locator.kind}\n"
        f"Locator value: {element.locator.value}\n"
        f"Text: {element.text}\n"
        f"Role: {element.role or '(unknown)'}\n"
        f"Tag: {element.tag_name or '(unknown)'}\n"
        f"Visible: {element.visible}\n"
        f"Enabled: {element.enabled}"
    )


def _execute_browser_action(
    action: Any,
) -> ToolResult:
    """Execute a browser action with normalized tool errors."""
    try:
        result = action()
    except (BrowserAutomationError, ValueError) as exc:
        return ToolResult(
            success=False,
            error=str(exc),
        )

    if isinstance(result, ToolResult):
        return result

    return ToolResult(
        success=True,
        output="Action completed.",
    )


class BrowserOpenTool(ToolInterface):
    """Open a URL in the interactive browser."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
        *,
        safety_policy: BrowserSafetyPolicy | None = None,
    ) -> None:
        self._browser = browser
        self._safety_policy = safety_policy

    @property
    def name(self) -> str:
        return "browser_open"

    @property
    def description(self) -> str:
        return (
            "Open a public HTTP or HTTPS URL in the interactive browser."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "HTTP or HTTPS URL to open."
                    ),
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        url = arguments.get("url")

        if not isinstance(url, str):
            return ToolResult(
                success=False,
                error="Argument 'url' must be a string.",
            )

        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return ToolResult(
                success=False,
                error="Only HTTP and HTTPS URLs are supported.",
            )

        if parsed.username or parsed.password:
            return ToolResult(
                success=False,
                error="URLs with embedded credentials are not allowed.",
            )

        if self._safety_policy is not None:
            try:
                self._safety_policy.validate_url(url)
            except BrowserSafetyError as exc:
                return ToolResult(
                    success=False,
                    error=str(exc),
                )

        return _execute_browser_action(
            lambda: self._open(url),
        )

    def _open(self, url: str) -> ToolResult:
        tab = self._browser.open(url)

        return ToolResult(
            success=True,
            output=_tab_output(tab),
            metadata={
                "tab_id": tab.tab_id,
                "url": tab.url,
                "title": tab.title,
            },
        )


class BrowserFindTool(ToolInterface):
    """Find and inspect an interactive browser element."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
    ) -> None:
        self._browser = browser

    @property
    def name(self) -> str:
        return "browser_find"

    @property
    def description(self) -> str:
        return (
            "Find an element in the active browser page and "
            "return its visible state."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "locator": _locator_schema(),
            },
            "required": ["locator"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        try:
            locator = _parse_locator(arguments)
            element = self._browser.find(locator)
        except (BrowserAutomationError, ValueError) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        if element is None:
            return ToolResult(
                success=False,
                error=(
                    "Element not found: "
                    f"{locator.kind}={locator.value}"
                ),
            )

        return ToolResult(
            success=True,
            output=_element_output(element),
            metadata={
                "locator_kind": locator.kind,
                "locator_value": locator.value,
                "visible": element.visible,
                "enabled": element.enabled,
            },
        )


class BrowserClickTool(ToolInterface):
    """Click a browser element."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
        *,
        safety_policy: BrowserSafetyPolicy | None = None,
    ) -> None:
        self._browser = browser
        self._safety_policy = safety_policy

    @property
    def name(self) -> str:
        return "browser_click"

    @property
    def description(self) -> str:
        return "Click one uniquely identified element in the active page."

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "locator": _locator_schema(),
            },
            "required": ["locator"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        try:
            locator = _parse_locator(arguments)

            if self._safety_policy is not None:
                self._safety_policy.validate_interaction(
                    locator,
                    url=self._browser.current_tab().url,
                )

            self._browser.click(locator)
        except (
            BrowserAutomationError,
            BrowserSafetyError,
            ValueError,
        ) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                "Clicked element: "
                f"{locator.kind}={locator.value}"
            ),
            metadata={
                "locator_kind": locator.kind,
                "locator_value": locator.value,
            },
        )


class BrowserTypeTool(ToolInterface):
    """Enter text into a browser element."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
        *,
        safety_policy: BrowserSafetyPolicy | None = None,
    ) -> None:
        self._browser = browser
        self._safety_policy = safety_policy

    @property
    def name(self) -> str:
        return "browser_type"

    @property
    def description(self) -> str:
        return "Enter text into one uniquely identified browser element."

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "locator": _locator_schema(),
                "text": {
                    "type": "string",
                    "description": "Text to enter.",
                },
                "clear": {
                    "type": "boolean",
                    "description": (
                        "Clear existing content before entering text."
                    ),
                    "default": True,
                },
            },
            "required": ["locator", "text"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        try:
            locator = _parse_locator(arguments)
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        text = arguments.get("text")
        clear = arguments.get("clear", True)

        if not isinstance(text, str):
            return ToolResult(
                success=False,
                error="Argument 'text' must be a string.",
            )

        if not isinstance(clear, bool):
            return ToolResult(
                success=False,
                error="Argument 'clear' must be a boolean.",
            )

        try:
            if self._safety_policy is not None:
                self._safety_policy.validate_interaction(
                    locator,
                    url=self._browser.current_tab().url,
                )

            self._browser.type_text(
                locator,
                text,
                clear=clear,
            )
        except (
            BrowserAutomationError,
            BrowserSafetyError,
        ) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                "Entered text into: "
                f"{locator.kind}={locator.value}"
            ),
            metadata={
                "locator_kind": locator.kind,
                "locator_value": locator.value,
                "clear": clear,
            },
        )


class BrowserPressTool(ToolInterface):
    """Press a keyboard key on a browser element."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
    ) -> None:
        self._browser = browser

    @property
    def name(self) -> str:
        return "browser_press"

    @property
    def description(self) -> str:
        return "Press a keyboard key on a browser element."

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "locator": _locator_schema(),
                "key": {
                    "type": "string",
                    "description": (
                        "Keyboard key, such as Enter, Tab, Escape, "
                        "ArrowDown, or Control+A."
                    ),
                },
            },
            "required": ["locator", "key"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        try:
            locator = _parse_locator(arguments)
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        key = arguments.get("key")

        if not isinstance(key, str):
            return ToolResult(
                success=False,
                error="Argument 'key' must be a string.",
            )

        try:
            self._browser.press(
                locator,
                key,
            )
        except (BrowserAutomationError, ValueError) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                f"Pressed '{key}' on "
                f"{locator.kind}={locator.value}"
            ),
            metadata={
                "locator_kind": locator.kind,
                "locator_value": locator.value,
                "key": key,
            },
        )


class BrowserReadTool(ToolInterface):
    """Read text from the active browser page or an element."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
    ) -> None:
        self._browser = browser

    @property
    def name(self) -> str:
        return "browser_read"

    @property
    def description(self) -> str:
        return (
            "Read visible text from the active browser page "
            "or one selected element."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "locator": _locator_schema(),
            },
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        raw_locator = arguments.get("locator")

        try:
            if raw_locator is None:
                text = self._browser.get_text()
                metadata: dict[str, Any] = {
                    "scope": "page",
                }
            else:
                locator = _parse_locator(arguments)
                text = self._browser.get_text(locator)
                metadata = {
                    "scope": "element",
                    "locator_kind": locator.kind,
                    "locator_value": locator.value,
                }
        except (BrowserAutomationError, ValueError) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=text,
            metadata=metadata,
        )


class BrowserWaitTool(ToolInterface):
    """Wait for a browser element to reach a state."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
    ) -> None:
        self._browser = browser

    @property
    def name(self) -> str:
        return "browser_wait"

    @property
    def description(self) -> str:
        return (
            "Wait for a browser element to become attached, detached, "
            "visible, or hidden."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "locator": _locator_schema(),
                "state": {
                    "type": "string",
                    "enum": [
                        "attached",
                        "detached",
                        "visible",
                        "hidden",
                    ],
                    "default": "visible",
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Maximum wait time in milliseconds.",
                },
            },
            "required": ["locator"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        try:
            locator = _parse_locator(arguments)
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        state = arguments.get("state", "visible")
        timeout_ms = arguments.get("timeout_ms")

        if state not in {
            "attached",
            "detached",
            "visible",
            "hidden",
        }:
            return ToolResult(
                success=False,
                error=f"Unsupported wait state: {state}",
            )

        if timeout_ms is not None:
            if not isinstance(timeout_ms, int):
                return ToolResult(
                    success=False,
                    error="Argument 'timeout_ms' must be an integer.",
                )

            if timeout_ms < 0:
                return ToolResult(
                    success=False,
                    error="Argument 'timeout_ms' cannot be negative.",
                )

        try:
            self._browser.wait_for(
                locator,
                state=state,
                timeout_ms=timeout_ms,
            )
        except BrowserAutomationError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                "Wait condition satisfied for "
                f"{locator.kind}={locator.value}: {state}"
            ),
            metadata={
                "locator_kind": locator.kind,
                "locator_value": locator.value,
                "state": state,
                "timeout_ms": timeout_ms,
            },
        )


class BrowserObserveTool(ToolInterface):
    """Observe the current browser page in a bounded structured format."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
    ) -> None:
        self._browser = browser

    @property
    def name(self) -> str:
        return "browser_observe"

    @property
    def description(self) -> str:
        return (
            "Observe the active browser page and return its URL, title, "
            "visible text, and interactive elements."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_text_characters": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50_000,
                    "default": 12_000,
                },
                "max_elements": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 100,
                },
            },
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        max_text_characters = arguments.get(
            "max_text_characters",
            12_000,
        )
        max_elements = arguments.get(
            "max_elements",
            100,
        )

        if not isinstance(max_text_characters, int):
            return ToolResult(
                success=False,
                error=(
                    "Argument 'max_text_characters' "
                    "must be an integer."
                ),
            )

        if not isinstance(max_elements, int):
            return ToolResult(
                success=False,
                error=(
                    "Argument 'max_elements' "
                    "must be an integer."
                ),
            )

        if not 1 <= max_text_characters <= 50_000:
            return ToolResult(
                success=False,
                error=(
                    "Argument 'max_text_characters' "
                    "must be between 1 and 50000."
                ),
            )

        if not 1 <= max_elements <= 500:
            return ToolResult(
                success=False,
                error=(
                    "Argument 'max_elements' "
                    "must be between 1 and 500."
                ),
            )

        try:
            observation = self._browser.observe(
                max_text_characters=max_text_characters,
                max_elements=max_elements,
            )
        except BrowserAutomationError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        output = self._format_observation(observation)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "tab_id": observation.tab.tab_id,
                "url": observation.tab.url,
                "title": observation.tab.title,
                "element_count": len(
                    observation.elements
                ),
                "text_truncated": observation.text_truncated,
                "elements_truncated": (
                    observation.elements_truncated
                ),
            },
        )

    @staticmethod
    def _format_observation(
        observation: BrowserPageObservation,
    ) -> str:
        lines = [
            "URL: "
            + observation.tab.url,
            "Title: "
            + (
                observation.tab.title
                or "(untitled)"
            ),
            "",
            "Interactive elements:",
        ]

        if not observation.elements:
            lines.append("(none observed)")
        else:
            for index, element in enumerate(
                observation.elements,
                start=1,
            ):
                name = element.name or "(unnamed)"
                text = element.text.strip()

                lines.append(
                    f"[{index}] "
                    f"tag={element.tag_name or '?'} "
                    f"role={element.role or '?'} "
                    f"name={name!r} "
                    f"visible={element.visible} "
                    f"enabled={element.enabled}"
                )

                if text:
                    lines.append(
                        f"    text={text!r}"
                    )

        lines.extend(
            [
                "",
                "Page text:",
                observation.text,
            ]
        )

        if observation.text_truncated:
            lines.append(
                "[Page text truncated by OSA.]"
            )

        if observation.elements_truncated:
            lines.append(
                "[Interactive element list truncated by OSA.]"
            )

        return "\n".join(lines)


class BrowserScreenshotTool(ToolInterface):
    """Capture a screenshot of the active browser page."""

    def __init__(
        self,
        browser: BrowserAutomationInterface,
    ) -> None:
        self._browser = browser

    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of the active browser page. "
            "The image is exposed to the observation layer."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        del arguments

        try:
            image = self._browser.screenshot()
        except BrowserAutomationError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                "Screenshot captured successfully. "
                f"Image size: {len(image)} bytes."
            ),
            metadata={
                "content_type": "image/png",
                "byte_length": len(image),
            },
        )


def create_browser_action_tools(
    browser: BrowserAutomationInterface,
    *,
    safety_policy: BrowserSafetyPolicy | None = None,
) -> tuple[ToolInterface, ...]:
    """Create interactive browser tools with shared safety policy."""
    return (
        BrowserOpenTool(
            browser,
            safety_policy=safety_policy,
        ),
        BrowserFindTool(browser),
        BrowserClickTool(
            browser,
            safety_policy=safety_policy,
        ),
        BrowserTypeTool(
            browser,
            safety_policy=safety_policy,
        ),
        BrowserPressTool(browser),
        BrowserReadTool(browser),
        BrowserWaitTool(browser),
        BrowserObserveTool(browser),
        BrowserScreenshotTool(browser),
    )
