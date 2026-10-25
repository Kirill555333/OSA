"""Tests for load_models_config from config/default.yaml (4.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from osa.utils.config import ConfigError, load_models_config


def _write(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / "default.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def test_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_LLAMA_URL", raising=False)
    missing = tmp_path / "does-not-exist.yaml"

    cfg = load_models_config(missing)

    assert set(cfg) == {"reasoning", "vision"}
    assert cfg["reasoning"].base_url == "http://127.0.0.1:8080"
    assert cfg["vision"].base_url == "http://127.0.0.1:8080"
    assert cfg["reasoning"].timeout == 45.0
    assert cfg["vision"].timeout == 60.0


def test_two_models_with_different_urls(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_LLAMA_URL", raising=False)
    cfg_path = _write(tmp_path, """
models:
  reasoning:
    base_url: "http://127.0.0.1:8081"
    model_name: "qwen3-8b"
    timeout: 60.0
  vision:
    base_url: "http://127.0.0.1:8080"
    model_name: "qwen2.5-vl-3b"
""")

    cfg = load_models_config(cfg_path)

    assert cfg["reasoning"].base_url == "http://127.0.0.1:8081"
    assert cfg["reasoning"].model_name == "qwen3-8b"
    assert cfg["reasoning"].timeout == 60.0
    assert cfg["vision"].base_url == "http://127.0.0.1:8080"
    assert cfg["vision"].model_name == "qwen2.5-vl-3b"
    assert cfg["vision"].timeout == 60.0
    assert "router" not in cfg


def test_env_fills_missing_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_LLAMA_URL", "http://10.0.0.1:9090")
    cfg_path = _write(tmp_path, """
models:
  reasoning:
    timeout: 30.0
  vision:
    timeout: 40.0
""")

    cfg = load_models_config(cfg_path)

    assert cfg["reasoning"].base_url == "http://10.0.0.1:9090"
    assert cfg["vision"].base_url == "http://10.0.0.1:9090"


def test_yaml_explicit_url_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_LLAMA_URL", "http://10.0.0.1:9090")
    cfg_path = _write(tmp_path, """
models:
  reasoning:
    base_url: "http://127.0.0.1:8081"
  vision:
    base_url: "http://127.0.0.1:8080"
""")

    cfg = load_models_config(cfg_path)

    assert cfg["reasoning"].base_url == "http://127.0.0.1:8081"
    assert cfg["vision"].base_url == "http://127.0.0.1:8080"


def test_router_section_optional(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_LLAMA_URL", raising=False)
    cfg_path = _write(tmp_path, """
models:
  reasoning: {}
  vision: {}
  router:
    base_url: "http://127.0.0.1:8082"
    model_name: "qwen3-1b"
""")

    cfg = load_models_config(cfg_path)

    assert "router" in cfg
    assert cfg["router"].base_url == "http://127.0.0.1:8082"
    assert cfg["router"].model_name == "qwen3-1b"


def test_invalid_yaml_raises(tmp_path):
    cfg_path = _write(tmp_path, "models: [unclosed")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_models_config(cfg_path)


def test_models_section_not_a_mapping_raises(tmp_path):
    cfg_path = _write(tmp_path, "models: [1, 2, 3]")
    with pytest.raises(ConfigError, match="'models' section must be a mapping"):
        load_models_config(cfg_path)

# --- 6.1: ModelConfig fields and provider behavior -------------------------


def test_default_provider_is_llama_cpp(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_LLAMA_URL", raising=False)
    cfg_path = _write(tmp_path, """
models:
  reasoning: {}
  vision: {}
""")
    cfg = load_models_config(cfg_path)
    assert cfg["reasoning"].provider == "llama_cpp"
    assert cfg["vision"].provider == "llama_cpp"


def test_explicit_openai_provider_parses(tmp_path):
    cfg_path = _write(tmp_path, """
models:
  reasoning:
    provider: "openai"
    base_url: "https://api.openai.com"
    model_name: "gpt-4o-mini"
  vision: {}
""")
    cfg = load_models_config(cfg_path)
    assert cfg["reasoning"].provider == "openai"
    assert cfg["reasoning"].base_url == "https://api.openai.com"
    assert cfg["reasoning"].model_name == "gpt-4o-mini"


def test_unknown_provider_raises(tmp_path):
    cfg_path = _write(tmp_path, """
models:
  reasoning:
    provider: "foobar"
  vision: {}
""")
    with pytest.raises(ConfigError, match="provider"):
        load_models_config(cfg_path)


def test_openai_without_base_url_does_not_inherit_llama_url(tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_LLAMA_URL", "http://10.0.0.1:9090")
    cfg_path = _write(tmp_path, """
models:
  reasoning:
    provider: "openai"
  vision: {}
""")
    cfg = load_models_config(cfg_path)
    assert cfg["reasoning"].base_url is None
    assert cfg["vision"].base_url == "http://10.0.0.1:9090"


def test_llamacpp_without_base_url_uses_llama_url_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_LLAMA_URL", "http://10.0.0.1:9090")
    cfg_path = _write(tmp_path, """
models:
  reasoning: {}
  vision: {}
""")
    cfg = load_models_config(cfg_path)
    assert cfg["reasoning"].base_url == "http://10.0.0.1:9090"


def test_existing_default_yaml_still_works(tmp_path, monkeypatch):
    monkeypatch.delenv("OSA_LLAMA_URL", raising=False)
    cfg_path = _write(tmp_path, """
agent:
  temperature: 0.1

models:
  reasoning:
    timeout: 45.0
  vision:
    timeout: 60.0
""")
    cfg = load_models_config(cfg_path)
    assert cfg["reasoning"].provider == "llama_cpp"
    assert cfg["vision"].provider == "llama_cpp"
    assert cfg["reasoning"].timeout == 45.0
    assert cfg["vision"].timeout == 60.0
