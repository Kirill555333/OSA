"""Desktop automation package."""

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
    "create_fake_desktop_automation",
]
