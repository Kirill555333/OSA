"""Cross-platform system information primitives for OSA."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemInfo:
    """Read-only system information exposed by OSA."""

    operating_system: str
    os_release: str
    os_version: str
    machine: str
    processor: str
    architecture: str
    cpu_count: int
    python_version: str
    platform_name: str


class SystemProvider:
    """Provide read-only information about the host system."""

    def get_system_info(self) -> SystemInfo:
        """Return normalized system information."""
        return SystemInfo(
            operating_system=platform.system() or "Unknown",
            os_release=platform.release() or "Unknown",
            os_version=platform.version() or "Unknown",
            machine=platform.machine() or "Unknown",
            processor=platform.processor() or "Unknown",
            architecture=platform.architecture()[0] or "Unknown",
            cpu_count=self._cpu_count(),
            python_version=platform.python_version(),
            platform_name=sys.platform,
        )

    @staticmethod
    def _cpu_count() -> int:
        """Return the logical CPU count."""
        return max(
            1,
            __import__("os").cpu_count() or 1,
        )
