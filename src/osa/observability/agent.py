from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter
from typing import Any, Protocol

from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)


class AgentObservabilityError(RuntimeError):
    """Raised when Agent observability is configured incorrectly."""


class AgentLike(Protocol):
    """Minimal Agent interface required by ObservableAgent."""

    def chat(self, user_input: str) -> Any:
        ...

    def chat_stream(self, user_input: str) -> Iterator[str]:
        ...

    def run_autonomous(self, goal: str) -> Any:
        ...


class ObservableAgent:
    """
    Observability wrapper around the existing Agent.

    The wrapped Agent remains authoritative:
    - chat semantics are unchanged;
    - streaming semantics are unchanged;
    - autonomous semantics are unchanged;
    - results are returned unchanged;
    - exceptions are re-raised;
    - logging failures never break Agent execution.
    """

    def __init__(
        self,
        agent: AgentLike,
        *,
        logger: ObservabilityLogger | None = None,
    ) -> None:
        if agent is None:
            raise AgentObservabilityError(
                "agent is required"
            )

        if logger is None:
            logger = InMemoryObservabilityLogger()

        self._agent = agent
        self._logger = logger

    @property
    def agent(self) -> AgentLike:
        return self._agent

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    def chat(self, user_input: str) -> Any:
        normalized_input = self._normalize_input(
            user_input,
            "user_input",
        )

        started = perf_counter()

        self._safe_log(
            ObservabilityEvent(
                event="agent.chat.started",
                source="agent",
                status="started",
                duration_ms=0.0,
                metadata={
                    "input_length": len(normalized_input),
                },
            )
        )

        try:
            result = self._agent.chat(
                normalized_input
            )
        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="agent.chat.failed",
                    source="agent",
                    status="failed",
                    duration_ms=self._duration_ms(started),
                    error=str(exc),
                )
            )
            raise

        self._safe_log(
            ObservabilityEvent(
                event="agent.chat.completed",
                source="agent",
                status="completed",
                duration_ms=self._duration_ms(started),
                metadata={
                    "response_present": result is not None,
                    "response_length": self._response_length(
                        result
                    ),
                },
            )
        )

        return result

    def chat_stream(
        self,
        user_input: str,
    ) -> Iterator[str]:
        normalized_input = self._normalize_input(
            user_input,
            "user_input",
        )

        started = perf_counter()
        emitted_length = 0

        self._safe_log(
            ObservabilityEvent(
                event="agent.chat.started",
                source="agent",
                status="started",
                duration_ms=0.0,
                metadata={
                    "input_length": len(normalized_input),
                    "streaming": True,
                },
            )
        )

        try:
            for chunk in self._agent.chat_stream(
                normalized_input
            ):
                if isinstance(chunk, str):
                    emitted_length += len(chunk)

                yield chunk

        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="agent.chat.failed",
                    source="agent",
                    status="failed",
                    duration_ms=self._duration_ms(started),
                    error=str(exc),
                    metadata={
                        "streaming": True,
                        "emitted_length": emitted_length,
                    },
                )
            )
            raise

        self._safe_log(
            ObservabilityEvent(
                event="agent.chat.completed",
                source="agent",
                status="completed",
                duration_ms=self._duration_ms(started),
                metadata={
                    "streaming": True,
                    "emitted_length": emitted_length,
                    "response_present": emitted_length > 0,
                },
            )
        )

    def run_autonomous(self, goal: str) -> Any:
        normalized_goal = self._normalize_input(
            goal,
            "goal",
        )

        started = perf_counter()

        self._safe_log(
            ObservabilityEvent(
                event="agent.autonomous.started",
                source="agent",
                status="started",
                duration_ms=0.0,
                metadata={
                    "goal_length": len(normalized_goal),
                },
            )
        )

        try:
            result = self._agent.run_autonomous(
                normalized_goal
            )
        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="agent.autonomous.failed",
                    source="agent",
                    status="failed",
                    duration_ms=self._duration_ms(started),
                    error=str(exc),
                )
            )
            raise

        success = self._report_success(result)

        self._safe_log(
            ObservabilityEvent(
                event=(
                    "agent.autonomous.completed"
                    if success
                    else "agent.autonomous.failed"
                ),
                source="agent",
                status=(
                    "completed"
                    if success
                    else "failed"
                ),
                duration_ms=self._duration_ms(started),
                metadata={
                    "success": success,
                    "report_present": result is not None,
                },
            )
        )

        return result

    @staticmethod
    def _normalize_input(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise AgentObservabilityError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized

    @staticmethod
    def _response_length(
        response: Any,
    ) -> int | None:
        if response is None:
            return None

        content = getattr(
            response,
            "content",
            None,
        )

        if isinstance(content, str):
            return len(content)

        return None

    @staticmethod
    def _report_success(
        report: Any,
    ) -> bool:
        success = getattr(
            report,
            "success",
            None,
        )

        return success is True

    @staticmethod
    def _duration_ms(
        started: float,
    ) -> float:
        return round(
            max(
                0.0,
                (perf_counter() - started) * 1000.0,
            ),
            3,
        )

    def _safe_log(
        self,
        event: ObservabilityEvent,
    ) -> None:
        try:
            self._logger.log(event)
        except Exception:
            # Observability must never break Agent execution.
            return
