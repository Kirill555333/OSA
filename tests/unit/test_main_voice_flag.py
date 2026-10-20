"""Tests for --no-voice flag (5.2)."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def _run_main(monkeypatch, argv: list[str]) -> dict:
    """Run osa.main.main() with argv; return captured run_* calls."""
    from osa import main as main_module

    captured: dict[str, object] = {}

    def fake_listening(listen_mode, voice_enabled=True):
        captured["listening"] = {"listen_mode": listen_mode, "voice_enabled": voice_enabled}

    def fake_interactive(voice_enabled=True):
        captured["interactive"] = {"voice_enabled": voice_enabled}

    monkeypatch.setattr(sys, "argv", ["osa"] + argv)

    with patch.object(main_module, "run_listening_loop", side_effect=fake_listening), \
         patch.object(main_module, "run_interactive_loop", side_effect=fake_interactive):
        main_module.main()

    return captured


def test_no_voice_disables_tts_in_listening_loop(monkeypatch):
    captured = _run_main(monkeypatch, ["--no-voice"])
    assert captured["listening"]["voice_enabled"] is False


def test_no_voice_disables_tts_in_interactive_loop(monkeypatch):
    captured = _run_main(monkeypatch, ["--no-voice", "--push-to-talk"])
    assert captured["interactive"]["voice_enabled"] is False


def test_default_voice_enabled(monkeypatch):
    captured = _run_main(monkeypatch, [])
    assert captured["listening"]["voice_enabled"] is True


def test_voice_and_no_voice_conflict_warns_and_no_voice_wins(monkeypatch, capsys):
    captured = _run_main(monkeypatch, ["--voice", "--no-voice"])
    assert captured["listening"]["voice_enabled"] is False
    err = capsys.readouterr().err
    assert "--voice" in err and "--no-voice" in err


def test_listen_mode_passed_through(monkeypatch):
    captured = _run_main(monkeypatch, ["--listen", "always"])
    assert captured["listening"]["listen_mode"] == "always"
    assert captured["listening"]["voice_enabled"] is True
