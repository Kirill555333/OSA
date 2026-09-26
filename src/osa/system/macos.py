"""macOS-specific system information provider."""

from __future__ import annotations

import platform

from osa.system.common import (
    SystemInfo,
    SystemProvider,
)


class MacOSSystemProvider(SystemProvider):
    """Provide read-only system information for macOS."""

    def get_system_info(self) -> SystemInfo:
        """Return macOS-specific normalized system information."""
        info = super().get_system_info()

        mac_version = platform.mac_ver()[0]

        return SystemInfo(
            operating_system="macOS",
            os_release=(
                mac_version
                or info.os_release
            ),
            os_version=info.os_version,
            machine=info.machine,
            processor=info.processor,
            architecture=info.architecture,
            cpu_count=info.cpu_count,
            python_version=info.python_version,
            platform_name=info.platform_name,
        )
