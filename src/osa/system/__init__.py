"""System information components for OSA."""

from __future__ import annotations

import platform

from osa.system.common import (
    SystemInfo,
    SystemProvider,
)
from osa.system.macos import (
    MacOSSystemProvider,
)
from osa.system.windows import (
    WindowsSystemProvider,
)


def create_system_provider() -> SystemProvider:
    """Create the provider for the current operating system."""
    current_system = platform.system().lower()

    if current_system == "darwin":
        return MacOSSystemProvider()

    if current_system == "windows":
        return WindowsSystemProvider()

    return SystemProvider()


__all__ = [
    "MacOSSystemProvider",
    "SystemInfo",
    "SystemProvider",
    "WindowsSystemProvider",
    "create_system_provider",
]
