"""Tests for _model_from_config bridge (6.1)."""

from __future__ import annotations

import pytest

from osa.models import LlamaCppModel, ModelConfig
from osa.utils.config import ConfigError


def test_model_from_config_returns_llamacpp_for_llama_cpp():
    from osa import main as main_module

    config = ModelConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8081",
        model_name="qwen3-8b",
        timeout=45.0,
    )

    model = main_module._model_from_config(config)

    assert isinstance(model, LlamaCppModel)
    assert model._config.base_url == "http://127.0.0.1:8081"
    assert model._config.model_name == "qwen3-8b"
    assert model._config.timeout == 45.0


def test_model_from_config_openai_raises():
    from osa import main as main_module

    config = ModelConfig(provider="openai", base_url="https://api.openai.com")

    with pytest.raises(ConfigError, match="openai"):
        main_module._model_from_config(config)


def test_model_from_config_default_model_name_for_llama_cpp():
    """The 'default' sentinel resolves to LlamaCppConfig historical default."""
    from osa import main as main_module

    config = ModelConfig(provider="llama_cpp", base_url="http://127.0.0.1:8080")

    model = main_module._model_from_config(config)

    assert model._config.model_name == "qwen2.5-vl-3b"
