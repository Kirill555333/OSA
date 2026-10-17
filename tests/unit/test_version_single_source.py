"""Tests for the single source of version (1.5.1.6.2)."""

from __future__ import annotations

import os
import subprocess
import sys

import osa
import osa.main
import osa.version


def test_version_is_a_string() -> None:
    assert isinstance(osa.version.__version__, str)
    assert osa.version.__version__


def test_osa_package_reexports_version() -> None:
    assert osa.__version__ == osa.version.__version__


def test_main_version_matches_package_version() -> None:
    assert osa.main.VERSION == osa.version.__version__


def test_cli_version_prints_single_source(tmp_path) -> None:
    env = dict(os.environ)
    env.setdefault("OSA_HOME", str(tmp_path / "osa-home"))

    result = subprocess.run(
        [sys.executable, "-m", "osa.main", "--version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert osa.version.__version__ in result.stdout
