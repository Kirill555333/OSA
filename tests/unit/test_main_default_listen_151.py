"""Tests for the default listening mode wiring in main.py (OSA 1.5.2.5)."""

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


def test_background_listening_is_the_default_mode() -> None:
    source = _main_source()

    if source is None:
        return

    assert "if not args.push_to_talk:" in source
    assert 'run_listening_loop(args.listen or "wake", voice_enabled=True)' in source


def test_legacy_press_enter_mode_stays_available_behind_a_flag() -> None:
    source = _main_source()

    if source is None:
        return

    assert '"--push-to-talk"' in source
    assert "run_interactive_loop(voice_enabled=True)" in source
