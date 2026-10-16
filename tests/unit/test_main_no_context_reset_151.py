"""Tests that main.py no longer resets the agent mid-loop (OSA 1.5.1.5.2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _main_source() -> str | None:
    try:
        spec = importlib.util.find_spec("osa.main")
    except ModuleNotFoundError:
        return None

    if spec is None or spec.origin is None:
        return None

    return Path(spec.origin).read_text(encoding="utf-8")


def test_main_does_not_call_agent_reset() -> None:
    source = _main_source()

    if source is None:
        return

    assert "agent.reset()" not in source


def test_main_does_not_contain_anti_loop_comment() -> None:
    source = _main_source()

    if source is None:
        return

    assert "Anti-Loop & Context Bounding" not in source
