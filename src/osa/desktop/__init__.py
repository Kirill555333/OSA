"""Desktop automation package for OSA."""

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
    FakeDesktopAutomation,
    create_fake_desktop_automation,
)
from osa.desktop.factory import create_desktop_automation
from osa.desktop.macos import (
    MacOSDesktopAutomation,
    MacOSDesktopAutomationConfig,
)
from osa.desktop.windows import (
    WindowsDesktopAutomation,
    WindowsDesktopAutomationConfig,
)

__all__ = [
    "DesktopApplication",
    "DesktopAutomationActionError",
    "DesktopAutomationConnectionError",
    "DesktopAutomationError",
    "DesktopAutomationInterface",
    "DesktopElement",
    "DesktopElementLocator",
    "DesktopPoint",
    "DesktopRect",
    "DesktopWindow",
    "FakeDesktopAutomation",
    "MacOSDesktopAutomation",
    "MacOSDesktopAutomationConfig",
    "WindowsDesktopAutomation",
    "WindowsDesktopAutomationConfig",
    "create_desktop_automation",
    "create_fake_desktop_automation",
]
