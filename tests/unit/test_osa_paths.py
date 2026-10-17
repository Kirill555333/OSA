"""Tests for OSA paths: OSA_HOME and OSA_WORKSPACE (1.5.1.6.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from osa.utils.paths import (
    OSAHomeError,
    OSAWorkspaceError,
    resolve_osa_home,
    resolve_workspace,
)


def test_default_home_is_dot_osa_in_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("OSA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    home = resolve_osa_home()

    assert home == (tmp_path / ".osa").resolve()
    assert home.is_dir()


def test_env_home_wins_and_is_created(monkeypatch, tmp_path):
    target = tmp_path / "custom-osa"
    monkeypatch.setenv("OSA_HOME", str(target))

    home = resolve_osa_home()

    assert home == target.resolve()
    assert home.is_dir()


def test_home_that_is_a_file_raises(monkeypatch, tmp_path):
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x")
    monkeypatch.setenv("OSA_HOME", str(file_path))

    with pytest.raises(OSAHomeError):
        resolve_osa_home()


def test_home_that_cannot_be_created_raises(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("OSA_HOME", str(blocker / "sub"))

    with pytest.raises(OSAHomeError):
        resolve_osa_home()


def test_workspace_defaults_to_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("OSA_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    workspace = resolve_workspace()

    assert workspace == tmp_path.resolve()


def test_env_workspace_wins(monkeypatch, tmp_path):
    target = tmp_path / "custom-workspace"
    target.mkdir()
    monkeypatch.setenv("OSA_WORKSPACE", str(target))

    workspace = resolve_workspace()

    assert workspace == target.resolve()


def test_workspace_that_is_a_file_raises(monkeypatch, tmp_path):
    file_path = tmp_path / "file-not-dir"
    file_path.write_text("x")
    monkeypatch.setenv("OSA_WORKSPACE", str(file_path))

    with pytest.raises(OSAWorkspaceError):
        resolve_workspace()


def test_workspace_that_does_not_exist_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("OSA_WORKSPACE", str(tmp_path / "nope"))

    with pytest.raises(OSAWorkspaceError):
        resolve_workspace()
