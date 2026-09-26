"""Windows-specific system information provider."""

from __future__ import annotations

import platform

from osa.system.common import (
    SystemInfo,
    SystemProvider,
)


class WindowsSystemProvider(SystemProvider):
    """Provide read-only system information for Windows."""

    def get_system_info(self) -> SystemInfo:
        """Return Windows-specific normalized system information."""
        info = super().get_system_info()

        win_release, win_version, _, _ = platform.win32_ver()

        return SystemInfo(
            operating_system="Windows",
            os_release=(
                win_release
                or info.os_release
            ),
            os_version=(
                win_version
                or info.os_version
            ),
            machine=info.machine,
            processor=info.processor,
            architecture=info.architecture,
            cpu_count=info.cpu_count,
            python_version=info.python_version,
            platform_name=info.platform_name,
        )
