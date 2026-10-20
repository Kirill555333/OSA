"""Tests for check_all_models (4.7)."""

from __future__ import annotations

from unittest.mock import patch


def test_check_all_models_both_ok(monkeypatch, tmp_path):
    from osa import main as main_module
    from osa.models import LlamaCppConfig

    reasoning_cfg = LlamaCppConfig(
        base_url="http://127.0.0.1:8081", model_name="qwen3-8b"
    )
    vision_cfg = LlamaCppConfig(
        base_url="http://127.0.0.1:8080", model_name="qwen2.5-vl-3b"
    )

    with patch.object(
        main_module, "load_models_config",
        return_value={"reasoning": reasoning_cfg, "vision": vision_cfg},
    ), patch.object(main_module, "check_llama_server", return_value=True):
        ok, lines = main_module.check_all_models()

    assert ok is True
    assert any("reasoning" in line and "qwen3-8b" in line for line in lines)
    assert any("vision" in line and "qwen2.5-vl-3b" in line for line in lines)


def test_check_all_models_reasoning_down(monkeypatch):
    from osa import main as main_module
    from osa.models import LlamaCppConfig

    reasoning_cfg = LlamaCppConfig(base_url="http://127.0.0.1:8081")
    vision_cfg = LlamaCppConfig(base_url="http://127.0.0.1:8080")

    def fake_check(url):
        return "8080" in url

    with patch.object(
        main_module, "load_models_config",
        return_value={"reasoning": reasoning_cfg, "vision": vision_cfg},
    ), patch.object(main_module, "check_llama_server", side_effect=fake_check):
        ok, lines = main_module.check_all_models()

    assert ok is False
    assert any("\u2717" in line and "reasoning" in line for line in lines)
    assert any("\u2713" in line and "vision" in line for line in lines)


def test_check_all_models_config_error():
    from osa import main as main_module
    from osa.utils.config import ConfigError

    with patch.object(
        main_module, "load_models_config", side_effect=ConfigError("bad yaml")
    ):
        ok, lines = main_module.check_all_models()

    assert ok is False
    assert any("Config error" in line for line in lines)
