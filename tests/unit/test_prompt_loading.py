"""Tests for system prompt loading (4.6)."""

from __future__ import annotations



from osa.prompts import load_system_prompt


def test_bundled_prompt_loads() -> None:
    prompt = load_system_prompt()
    assert "OSA" in prompt
    assert "launch_application" in prompt
    assert "Правила" in prompt


def test_env_override_wins(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom_prompt.txt"
    custom.write_text("Ты — мой особый ассистент.", encoding="utf-8")
    monkeypatch.setenv("OSA_SYSTEM_PROMPT", str(custom))

    assert load_system_prompt() == "Ты — мой особый ассистент."


def test_env_points_to_missing_file_falls_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OSA_SYSTEM_PROMPT", str(tmp_path / "missing.txt"))
    prompt = load_system_prompt()
    assert "OSA" in prompt
    assert "launch_application" in prompt


def test_explicit_path_argument_wins(tmp_path, monkeypatch) -> None:
    env_custom = tmp_path / "env.txt"
    env_custom.write_text("env version", encoding="utf-8")
    monkeypatch.setenv("OSA_SYSTEM_PROMPT", str(env_custom))

    explicit = tmp_path / "explicit.txt"
    explicit.write_text("explicit version", encoding="utf-8")

    assert load_system_prompt(explicit) == "explicit version"


def test_empty_file_falls_back(tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n", encoding="utf-8")
    monkeypatch.setenv("OSA_SYSTEM_PROMPT", str(empty))

    prompt = load_system_prompt()
    assert "OSA" in prompt


def test_utf8_decoding_preserves_russian() -> None:
    prompt = load_system_prompt()
    assert "Оса" in prompt
    assert "открой калькулятор" in prompt
