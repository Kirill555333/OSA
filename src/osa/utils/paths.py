"""Filesystem paths for OSA data (OSA_HOME) and workspace (OSA_WORKSPACE).

Two distinct locations:
- OSA_HOME: OSA's own data — memory, settings, logs, screenshots.
  Default: ~/.osa. Override: $OSA_HOME.
- OSA_WORKSPACE: allowed files for SafeFilesystem and SafeShell.
  Default: current working directory. Override: $OSA_WORKSPACE.

If the corresponding env var is set but points to an unusable location,
an error is raised instead of silently falling back. This keeps the user
aware of where OSA writes data and what files it can touch.
"""

from __future__ import annotations

import os
from pathlib import Path


class OSAHomeError(ValueError):
    """Raised when $OSA_HOME is explicitly set but cannot be used."""


class OSAWorkspaceError(ValueError):
    """Raised when $OSA_WORKSPACE is explicitly set but cannot be used."""


def resolve_osa_home() -> Path:
    """Return the OSA data directory, creating it if necessary.

    Uses $OSA_HOME if set, otherwise ~/.osa. When $OSA_HOME is set but
    points to a file, or cannot be created, raises OSAHomeError.
    """
    raw = os.environ.get("OSA_HOME", "").strip()

    if raw:
        candidate = Path(raw).expanduser()

        if candidate.exists() and not candidate.is_dir():
            raise OSAHomeError(
                f"OSA_HOME points to a file, not a directory: {candidate}"
            )

        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSAHomeError(
                f"Cannot create OSA_HOME directory {candidate}: {exc}"
            ) from exc

        return candidate.resolve()

    default = Path.home() / ".osa"

    try:
        default.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSAHomeError(
            f"Cannot create default OSA_HOME directory {default}: {exc}"
        ) from exc

    return default.resolve()


def resolve_workspace() -> Path:
    """Return the OSA workspace directory.

    Uses $OSA_WORKSPACE if set, otherwise the current working directory.
    When $OSA_WORKSPACE is set but does not exist or is not a directory,
    raises OSAWorkspaceError.
    """
    raw = os.environ.get("OSA_WORKSPACE", "").strip()

    if raw:
        candidate = Path(raw).expanduser()

        if not candidate.exists():
            raise OSAWorkspaceError(
                f"OSA_WORKSPACE does not exist: {candidate}"
            )

        if not candidate.is_dir():
            raise OSAWorkspaceError(
                f"OSA_WORKSPACE is not a directory: {candidate}"
            )

        return candidate.resolve()

    return Path.cwd().resolve()
