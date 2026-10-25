"""Tests for create_agent model wiring (4.5, updated in 6.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from osa.models import LlamaCppModel, ModelConfig


def _read_source() -> str:
    from osa import main as main_module

    return Path(main_module.__file__).read_text(encoding="utf-8")


def test_create_agent_loads_models_config() -> None:
    source = _read_source()
    assert "load_models_config()" in source
    assert "models_config['reasoning']" in source
    assert "models_config['vision']" in source


def test_inspect_screen_uses_vision_model() -> None:
    source = _read_source()
    assert "InspectScreenTool(vision_model)" in source
    assert "InspectScreenTool(model)" not in source


def test_create_agent_runtime_uses_reasoning_client(tmp_path, monkeypatch):
    """Behavioral check: reasoning ModelConfig must reach LlamaCppModel.

    Compared by value (base_url), not by object identity: _model_from_config
    constructs a fresh LlamaCppConfig from ModelConfig.
    """
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    monkeypatch.setenv("OSA_WORKSPACE", str(tmp_path))

    from osa import main as main_module

    reasoning_cfg = ModelConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8081",
        model_name="qwen3-8b",
        timeout=45.0,
    )
    vision_cfg = ModelConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8080",
        model_name="qwen2.5-vl-3b",
        timeout=60.0,
    )

    fake_reasoning = MagicMock(spec=LlamaCppModel)
    fake_vision = MagicMock(spec=LlamaCppModel)

    captured: dict[str, list[str]] = {"seen_urls": []}

    def fake_factory(llamacpp_config):
        captured["seen_urls"].append(llamacpp_config.base_url)
        if llamacpp_config.base_url == "http://127.0.0.1:8081":
            return fake_reasoning
        return fake_vision

    with patch.object(main_module, "load_models_config") as mock_cfg, \
         patch.object(main_module, "LlamaCppModel", side_effect=fake_factory):
        mock_cfg.return_value = {"reasoning": reasoning_cfg, "vision": vision_cfg}
        agent = main_module.create_agent()

    assert agent.model is fake_reasoning
    assert "http://127.0.0.1:8081" in captured["seen_urls"]
    assert "http://127.0.0.1:8080" in captured["seen_urls"]
