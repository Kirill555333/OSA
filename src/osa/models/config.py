"""Universal model configuration for OSA (6.1).

ModelConfig decouples the model layer from the local llama.cpp backend.
Provider-specific settings live here; LlamaCppConfig remains untouched
as the concrete configuration for LlamaCppModel (bridge until 6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from osa.utils.config import ConfigError

SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"llama_cpp", "openai"})


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Role-level model configuration (provider, endpoint, credentials name).

    api_key_env stores the *name* of the environment variable that holds
    the API key, never the value. Reads/validates happen in the provider client.
    """

    provider: str = "llama_cpp"
    base_url: str | None = None
    model_name: str = "default"
    api_key_env: str | None = None
    timeout: float = 60.0
    max_context: int | None = None
    supports_vision: bool = False
    supports_tools: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"Unknown model provider '{self.provider}'. "
                f"Supported: {sorted(SUPPORTED_PROVIDERS)}."
            )

        if self.timeout <= 0:
            raise ConfigError("ModelConfig.timeout must be > 0.")

        if self.max_context is not None and self.max_context <= 0:
            raise ConfigError("ModelConfig.max_context must be > 0 when set.")

        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))
