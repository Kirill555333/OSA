"""Model client factory (6.2).

``create_model`` dispatches a ``ModelConfig`` to a concrete ``ModelInterface``.
In 6.2 only ``provider="llama_cpp"`` is implemented; ``openai`` and any other
provider explicitly raise ``ConfigError`` (no silent fallback).
"""

from __future__ import annotations

from osa.models.config import ModelConfig
from osa.models.interface import ModelInterface
from osa.models.local import LlamaCppConfig, LlamaCppModel
from osa.utils.config import ConfigError

_LLAMA_CPP_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_MODEL_NAME_SENTINEL = "default"


def create_model(config: ModelConfig) -> ModelInterface:
    """Build a model client from a ModelConfig.

    Providers implemented in 6.2:
      - ``llama_cpp`` -> LlamaCppModel(LlamaCppConfig(...))

    ``openai`` is accepted by ModelConfig but not yet implemented; it raises
    ConfigError with a stable message. Any other provider that ModelConfig
    might allow (after a future SUPPORTED_PROVIDERS expansion) also raises
    ConfigError - never a fallback to llama_cpp.
    """
    if config.provider == "llama_cpp":
        default_model_name = LlamaCppConfig().model_name
        model_name = (
            config.model_name
            if config.model_name
            and config.model_name != _DEFAULT_MODEL_NAME_SENTINEL
            else default_model_name
        )

        return LlamaCppModel(
            LlamaCppConfig(
                base_url=config.base_url or _LLAMA_CPP_DEFAULT_BASE_URL,
                model_name=model_name,
                timeout=config.timeout,
            )
        )

    if config.provider == "openai":
        raise ConfigError(
            f"provider '{config.provider}' is not implemented until 6.3"
        )

    raise ConfigError(
        f"provider '{config.provider}' is not implemented"
    )
