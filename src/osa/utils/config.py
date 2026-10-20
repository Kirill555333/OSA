"""Common configuration primitives for OSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when OSA configuration cannot be parsed safely."""


@dataclass(frozen=True)
class AgentConfig:
    """Runtime knobs for the OSA agent (5.1)."""

    temperature: float = 0.1
    max_tokens: int = 512
    max_tool_rounds: int = 8
    tool_choice: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(
                "agent.temperature must be between 0.0 and 2.0."
            )

        if self.max_tokens <= 0:
            raise ConfigError("agent.max_tokens must be > 0.")

        if self.max_tool_rounds <= 0:
            raise ConfigError("agent.max_tool_rounds must be > 0.")

        if self.tool_choice not in {"auto", "required", "none"}:
            raise ConfigError(
                "agent.tool_choice must be 'auto', 'required' or 'none'."
            )


def load_agent_config(
    path: Path | None = None,
) -> AgentConfig:
    """Load agent: section from config/default.yaml.

    Env var OSA_TOOL_CHOICE always wins over the YAML tool_choice value.
    """
    import os

    config_path = path or Path('config') / 'default.yaml'
    agent_section: dict[str, Any] = {}

    if config_path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "pyyaml is required to read config/default.yaml."
            ) from exc

        try:
            raw = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Invalid YAML in {config_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ConfigError(
                f"Config root must be a mapping, got {type(raw).__name__}."
            )

        agent_section = raw.get('agent') or {}

        if not isinstance(agent_section, dict):
            raise ConfigError(
                "'agent' section must be a mapping."
            )

    temperature = float(agent_section.get('temperature', 0.1))
    max_tokens = int(agent_section.get('max_tokens', 512))
    max_tool_rounds = int(agent_section.get('max_tool_rounds', 8))

    yaml_tool_choice = str(agent_section.get('tool_choice', 'auto')).strip()
    env_tool_choice = os.environ.get('OSA_TOOL_CHOICE', '').strip()
    tool_choice = env_tool_choice or yaml_tool_choice

    return AgentConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        max_tool_rounds=max_tool_rounds,
        tool_choice=tool_choice,
    )


def load_models_config(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load per-role model configs from config/default.yaml.

    Returns a dict with keys 'reasoning' and 'vision' (both present),
    and optionally 'router'. Each value is an LlamaCppConfig.

    Precedence for base_url: explicit YAML > OSA_LLAMA_URL env > 127.0.0.1:8080.
    """
    from osa.models.local import LlamaCppConfig

    config_path = path or Path('config') / 'default.yaml'
    models_section: dict[str, Any] = {}

    if config_path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "pyyaml is required to read config/default.yaml."
            ) from exc

        try:
            raw = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Invalid YAML in {config_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ConfigError(
                f"Config root must be a mapping, got {type(raw).__name__}."
            )

        models_section = raw.get('models') or {}

        if not isinstance(models_section, dict):
            raise ConfigError(
                "'models' section must be a mapping."
            )

    def _build(role_name: str, default_timeout: float) -> LlamaCppConfig:
        role_raw = models_section.get(role_name) or {}

        if not isinstance(role_raw, dict):
            raise ConfigError(
                f"'models.{role_name}' must be a mapping."
            )

        kwargs: dict[str, Any] = {
            'timeout': float(role_raw.get('timeout', default_timeout)),
        }

        if 'base_url' in role_raw:
            kwargs['base_url'] = str(role_raw['base_url'])

        if 'model_name' in role_raw:
            kwargs['model_name'] = str(role_raw['model_name'])

        return LlamaCppConfig(**kwargs)

    result: dict[str, Any] = {
        'reasoning': _build('reasoning', 45.0),
        'vision': _build('vision', 60.0),
    }

    if 'router' in models_section:
        result['router'] = _build('router', 30.0)

    return result


@dataclass(frozen=True)
class EnvironmentConfig:
    """Immutable, typed access to environment-style configuration values."""

    values: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> EnvironmentConfig:
        """Create configuration from the supplied mapping or process environment."""
        import os

        return cls(dict(os.environ if environ is None else environ))

    def get(
        self,
        name: str,
        default: str | None = None,
    ) -> str | None:
        """Return a stripped text value, falling back when unset or empty."""
        value = self.values.get(name)
        if value is None:
            return default

        normalized = value.strip()
        return normalized if normalized else default

    def require(self, name: str) -> str:
        """Return a required non-empty text value."""
        value = self.get(name)
        if value is None:
            raise ConfigError(f"Required configuration '{name}' is not set.")
        return value

    def boolean(
        self,
        name: str,
        default: bool = False,
    ) -> bool:
        """Parse a conventional boolean environment value."""
        value = self.get(name)
        if value is None:
            return default

        normalized = value.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

        raise ConfigError(
            f"Configuration '{name}' must be a boolean "
            f"(true/false, yes/no, on/off, or 1/0)."
        )

    def integer(
        self,
        name: str,
        default: int,
    ) -> int:
        """Parse an integer environment value."""
        value = self.get(name)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(
                f"Configuration '{name}' must be an integer."
            ) from exc

    def floating(
        self,
        name: str,
        default: float,
    ) -> float:
        """Parse a floating-point environment value."""
        value = self.get(name)
        if value is None:
            return default

        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(
                f"Configuration '{name}' must be a number."
            ) from exc
