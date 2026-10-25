"""Tests for create_model factory (6.2)."""

from __future__ import annotations

import pytest

from osa.models import LlamaCppModel, ModelConfig, create_model
from osa.utils.config import ConfigError


def test_create_model_llama_cpp_returns_llama_model():
    config = ModelConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8081",
        model_name="qwen3-8b",
        timeout=45.0,
    )

    model = create_model(config)

    assert isinstance(model, LlamaCppModel)
    assert model._config.base_url == "http://127.0.0.1:8081"
    assert model._config.model_name == "qwen3-8b"
    assert model._config.timeout == 45.0


def test_create_model_llama_cpp_default_model_name():
    """The 'default' sentinel resolves to LlamaCppConfig historical default."""
    config = ModelConfig(provider="llama_cpp", base_url="http://127.0.0.1:8080")

    model = create_model(config)

    assert model._config.model_name == "qwen2.5-vl-3b"


def test_create_model_openai_raises():
    config = ModelConfig(provider="openai", base_url="https://api.openai.com")

    with pytest.raises(ConfigError, match="openai"):
        create_model(config)


def test_create_model_unknown_provider_raises(monkeypatch):
    """Validate the factory's own else-branch, not ModelConfig.__post_init__.

    ModelConfig rejects unknown providers before create_model is called, so we
    temporarily widen SUPPORTED_PROVIDERS to let a stub config through; the
    factory itself must still refuse it (no fallback to llama_cpp).
    """
    import osa.models.config as config_module

    monkeypatch.setattr(
        config_module,
        "SUPPORTED_PROVIDERS",
        frozenset({"llama_cpp", "openai", "stub_provider"}),
    )

    config = config_module.ModelConfig(provider="stub_provider")

    with pytest.raises(ConfigError, match="stub_provider"):
        create_model(config)
