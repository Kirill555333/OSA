"""Browser components for OSA."""

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
from osa.browser.http import (
    HttpBrowser,
    HttpBrowserConfig,
)
from osa.browser.interface import (
    BrowserConnectionError,
    BrowserError,
    BrowserInterface,
    BrowserPage,
    BrowserResponseError,
)
from osa.browser.playwright import (
    PlaywrightBrowser,
    PlaywrightBrowserConfig,
)
from osa.browser.safety import (
    BrowserSafetyError,
    BrowserSafetyPolicy,
    create_default_browser_safety_policy,
)

__all__ = [
    "BrowserAutomationActionError",
    "BrowserAutomationConnectionError",
    "BrowserAutomationError",
    "BrowserAutomationInterface",
    "BrowserConnectionError",
    "BrowserElement",
    "BrowserError",
    "BrowserInterface",
    "BrowserLocator",
    "BrowserPage",
    "BrowserPageObservation",
    "BrowserResponseError",
    "BrowserSafetyError",
    "BrowserSafetyPolicy",
    "BrowserTab",
    "FakeBrowserAutomation",
    "HttpBrowser",
    "HttpBrowserConfig",
    "PlaywrightBrowser",
    "PlaywrightBrowserConfig",
    "create_default_browser_safety_policy",
    "create_fake_browser_automation",
]
