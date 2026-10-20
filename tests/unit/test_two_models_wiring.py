"""Tests for create_agent model wiring (4.5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from osa.models import LlamaCppConfig, LlamaCppModel


def _read_source() -> str:
    from osa import main as main_module

    return Path(main_module.__file__).read_text(encoding="utf-8")


def test_create_agent_loads_models_config() -> None:
    source = _read_source()
    assert "load_models_config()" in source
    assert "models_config['reasoning']" in source
    assert "models_config['vision']" in source


def test_create_agent_builds_two_llama_models() -> None:
    source = _read_source()
    assert "LlamaCppModel(models_config['reasoning'])" in source
    assert "LlamaCppModel(models_config['vision'])" in source


def test_inspect_screen_uses_vision_model() -> None:
    source = _read_source()
    assert "InspectScreenTool(vision_model)" in source
    assert "InspectScreenTool(model)" not in source


def test_create_agent_runtime_uses_reasoning_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    monkeypatch.setenv("OSA_WORKSPACE", str(tmp_path))

    from osa import main as main_module

    fake_reasoning = MagicMock(spec=LlamaCppModel)
    fake_vision = MagicMock(spec=LlamaCppModel)

    created: dict[str, object] = {}

    def fake_model_factory(config):
        if isinstance(config, LlamaCppConfig):
            if id(config) == id(created.get("reasoning_cfg")):
                return fake_reasoning
            if id(config) == id(created.get("vision_cfg")):
                return fake_vision
        return MagicMock(spec=LlamaCppModel)

    reasoning_cfg = LlamaCppConfig()
    vision_cfg = LlamaCppConfig()
    created["reasoning_cfg"] = reasoning_cfg
    created["vision_cfg"] = vision_cfg

    with patch.object(main_module, "load_models_config") as mock_cfg, \
         patch.object(main_module, "LlamaCppModel", side_effect=fake_model_factory):
        mock_cfg.return_value = {"reasoning": reasoning_cfg, "vision": vision_cfg}
        agent = main_module.create_agent()

    assert agent.model is fake_reasoning
