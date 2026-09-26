"""Error recovery policies for OSA."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Callable, TypeVar

from osa.models import (
    ModelConnectionError,
    ModelError,
)


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Configuration for bounded error recovery."""

    max_model_retries: int = 1
    model_retry_delay_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_model_retries < 0:
            raise ValueError(
                "max_model_retries cannot be negative."
            )

        if self.model_retry_delay_seconds < 0:
            raise ValueError(
                "model_retry_delay_seconds cannot be negative."
            )


class ErrorRecovery:
    """Provide bounded recovery for retry-safe failures."""

    def __init__(
        self,
        config: RecoveryConfig | None = None,
    ) -> None:
        self._config = (
            config
            or RecoveryConfig()
        )

    @property
    def config(self) -> RecoveryConfig:
        """Return the recovery configuration."""
        return self._config

    def run_model(
        self,
        operation: Callable[[], T],
    ) -> tuple[T, int]:
        """
        Execute a model operation with bounded retries.

        Only model connection failures are retried automatically.
        Other model errors are raised immediately.
        """
        attempts = 0
        max_attempts = (
            self._config.max_model_retries + 1
        )

        while attempts < max_attempts:
            attempts += 1

            try:
                return operation(), attempts
            except ModelConnectionError:
                if attempts >= max_attempts:
                    raise

                if self._config.model_retry_delay_seconds > 0:
                    sleep(
                        self._config.model_retry_delay_seconds
                    )
            except ModelError:
                raise

        raise RuntimeError(
            "Model recovery loop exited unexpectedly."
        )

    @staticmethod
    def tool_exception_message(
        tool_name: str,
        error: Exception,
    ) -> str:
        """Convert a tool exception into model-readable text."""
        return (
            f"Tool '{tool_name}' failed with an unexpected error: "
            f"{error}"
        )
