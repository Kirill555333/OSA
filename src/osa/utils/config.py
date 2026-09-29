"""Common configuration primitives for OSA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


class ConfigError(ValueError):
    """Raised when OSA configuration cannot be parsed safely."""


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
