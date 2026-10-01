"""Platform-aware desktop automation factory for OSA."""

from __future__ import annotations

import sys

from osa.desktop.automation import (
    DesktopAutomationInterface,
    FakeDesktopAutomation,
)


def create_desktop_automation() -> DesktopAutomationInterface:
    """Create the appropriate desktop automation backend for the current OS."""
    if sys.platform == "darwin":
        from osa.desktop.macos import MacOSDesktopAutomation

        return MacOSDesktopAutomation()

    if sys.platform == "win32":
        from osa.desktop.windows import WindowsDesktopAutomation

        return WindowsDesktopAutomation()

    return FakeDesktopAutomation()
