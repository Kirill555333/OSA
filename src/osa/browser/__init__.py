from osa.browser.safety import (
    BrowserSafetyError,
    BrowserSafetyPolicy,
    create_default_browser_safety_policy,
)

from osa.browser.automation import (
    BrowserAutomationActionError,
    BrowserAutomationConnectionError,
    BrowserAutomationError,
    BrowserAutomationInterface,
    BrowserElement,
    BrowserLocator,
    BrowserPageObservation,
    BrowserTab,
    FakeBrowserAutomation,
    create_fake_browser_automation,
)

from osa.browser.playwright import (
    PlaywrightBrowser,
    PlaywrightBrowserConfig,
)

from osa.browser.http import (
    HttpBrowser,
    HttpBrowserConfig,
)
"""Browser components for OSA."""

from osa.browser.interface import (
    BrowserConnectionError,
    BrowserError,
    BrowserInterface,
    BrowserPage,
    BrowserResponseError,
)

__all__ = [
    "BrowserConnectionError",
    "HttpBrowser",
    "HttpBrowserConfig",
    "BrowserError",
    "BrowserInterface",
    "BrowserPage",
    "BrowserResponseError",
]
