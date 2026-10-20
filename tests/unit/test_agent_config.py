"""Tests for load_agent_config (5.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from osa.utils.config import AgentConfig, ConfigError, load_agent_config


def _write(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / "default.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def test_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_TOOL_CHOICE", raising=False)
    cfg = load_agent_config(tmp_path / "does-not-exist.yaml")
    assert cfg == AgentConfig()


def test_full_override_from_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_TOOL_CHOICE", raising=False)
    cfg_path = _write(tmp_path, """
agent:
  temperature: 0.5
  max_tokens: 1024
  max_tool_rounds: 3
  tool_choice: "required"
""")
    cfg = load_agent_config(cfg_path)
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1024
    assert cfg.max_tool_rounds == 3
    assert cfg.tool_choice == "required"


def test_env_tool_choice_wins_over_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_TOOL_CHOICE", "none")
    cfg_path = _write(tmp_path, """
agent:
  tool_choice: "required"
""")
    cfg = load_agent_config(cfg_path)
    assert cfg.tool_choice == "none"


def test_invalid_tool_choice_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_TOOL_CHOICE", raising=False)
    cfg_path = _write(tmp_path, """
agent:
  tool_choice: "bogus"
""")
    with pytest.raises(ConfigError, match="tool_choice"):
        load_agent_config(cfg_path)


def test_invalid_temperature_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_TOOL_CHOICE", raising=False)
    cfg_path = _write(tmp_path, """
agent:
  temperature: 5.0
""")
    with pytest.raises(ConfigError, match="temperature"):
        load_agent_config(cfg_path)


def test_main_uses_agent_config_values(tmp_path, monkeypatch):
    """Source-level check: create_agent wires AgentConfig into Agent()."""
    from osa import main as main_module

    source = Path(main_module.__file__).read_text(encoding="utf-8")
    assert "load_agent_config()" in source
    assert "agent_config.temperature" in source
    assert "agent_config.max_tokens" in source
    assert "agent_config.max_tool_rounds" in source
    assert "agent_config.tool_choice" in source
