"""Continuous voice runtime foundation for OSA 0.7.6."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock, Thread, current_thread
from typing import Callable

from osa.voice.runtime import VoiceRuntime
from osa.voice.session import VoiceSessionResult


class ContinuousVoiceRuntimeError(
    RuntimeError
):
    """Raised when the continuous voice runtime cannot operate."""


class ContinuousVoiceRuntimeState(str, Enum):
    """Lifecycle state of a continuous voice runtime."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class ContinuousVoiceRuntimeConfig:
    """Configuration for the continuous voice worker."""

    thread_name: str = "osa-voice"
    daemon: bool = True
    join_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        normalized_thread_name = self.thread_name.strip()

        if not normalized_thread_name:
            raise ValueError(
                "thread_name cannot be empty."
            )

        if not isinstance(
            self.daemon,
            bool,
        ):
            raise ValueError(
                "daemon must be boolean."
            )

        if self.join_timeout_seconds < 0:
            raise ValueError(
                "join_timeout_seconds cannot be negative."
            )

        object.__setattr__(
            self,
            "thread_name",
            normalized_thread_name,
        )


VoiceResultCallback = Callable[
    [VoiceSessionResult],
    None,
]


class ContinuousVoiceRuntime:
    """
    Run VoiceRuntime repeatedly in a dedicated worker thread.

    Each cycle uses the existing VoiceRuntime.run_once() pipeline, so safety,
    VAD, STT, Agent, TTS, normalization, and playback remain unchanged.
    """

    def __init__(
        self,
        runtime: VoiceRuntime,
        *,
        config: ContinuousVoiceRuntimeConfig | None = None,
        on_result: VoiceResultCallback | None = None,
    ) -> None:
        if runtime is None:
            raise ValueError(
                "runtime is required."
            )

        if not isinstance(
            runtime,
            VoiceRuntime,
        ):
            raise TypeError(
                "runtime must be a VoiceRuntime."
            )

        if on_result is not None and not callable(
            on_result
        ):
            raise TypeError(
                "on_result must be callable."
            )

        self._runtime = runtime
        self._config = (
            config
            if config is not None
            else ContinuousVoiceRuntimeConfig()
        )
        self._on_result = on_result
        self._state = ContinuousVoiceRuntimeState.STOPPED
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_result: VoiceSessionResult | None = None
        self._last_error: Exception | None = None
        self._lock = RLock()

    @property
    def runtime(self) -> VoiceRuntime:
        """Return the wrapped VoiceRuntime."""
        return self._runtime

    @property
    def config(self) -> ContinuousVoiceRuntimeConfig:
        """Return immutable continuous runtime configuration."""
        return self._config

    @property
    def state(self) -> ContinuousVoiceRuntimeState:
        """Return the worker lifecycle state."""
        with self._lock:
            return self._state

    @property
    def running(self) -> bool:
        """Return whether the worker thread is active."""
        return self.state in {
            ContinuousVoiceRuntimeState.STARTING,
            ContinuousVoiceRuntimeState.RUNNING,
        }

    @property
    def last_result(self) -> VoiceSessionResult | None:
        """Return the most recent completed voice result."""
        with self._lock:
            return self._last_result

    @property
    def last_error(self) -> Exception | None:
        """Return the most recent worker error."""
        with self._lock:
            return self._last_error

    @property
    def thread(self) -> Thread | None:
        """Return the worker thread, if started."""
        with self._lock:
            return self._thread

    def start(self) -> None:
        """Start the continuous voice worker."""
        with self._lock:
            if self._state in {
                ContinuousVoiceRuntimeState.STARTING,
                ContinuousVoiceRuntimeState.RUNNING,
            }:
                return

            if self._state == ContinuousVoiceRuntimeState.ERROR:
                raise ContinuousVoiceRuntimeError(
                    "Continuous runtime is in an error state; reset it first."
                )

            self._stop_event.clear()
            self._last_error = None
            self._state = ContinuousVoiceRuntimeState.STARTING

        try:
            self._runtime.start()

            thread = Thread(
                target=self._run_worker,
                name=self._config.thread_name,
                daemon=self._config.daemon,
            )

            with self._lock:
                self._thread = thread
                self._state = ContinuousVoiceRuntimeState.RUNNING

            thread.start()

        except Exception as exc:
            with self._lock:
                self._last_error = exc
                self._state = ContinuousVoiceRuntimeState.ERROR

            raise ContinuousVoiceRuntimeError(
                f"Could not start continuous voice runtime: {exc}"
            ) from exc

    def stop(self) -> None:
        """
        Stop the continuous worker and release audio capture.

        The underlying VoiceSession remains reusable.
        """
        with self._lock:
            if self._state == ContinuousVoiceRuntimeState.STOPPED:
                return

            self._state = ContinuousVoiceRuntimeState.STOPPING
            self._stop_event.set()
            thread = self._thread

        try:
            self._runtime.stop_capture()
        except Exception as exc:
            with self._lock:
                self._last_error = exc
                self._state = ContinuousVoiceRuntimeState.ERROR

            raise ContinuousVoiceRuntimeError(
                f"Could not stop continuous voice runtime: {exc}"
            ) from exc

        if (
            thread is not None
            and thread is not current_thread()
        ):
            thread.join(
                timeout=self._config.join_timeout_seconds
            )

        with self._lock:
            self._thread = None

            if self._state != ContinuousVoiceRuntimeState.ERROR:
                self._state = ContinuousVoiceRuntimeState.STOPPED

    def reset(self) -> None:
        """
        Reset session state and prepare the continuous runtime for another
        start().
        """
        self.stop()

        try:
            self._runtime.session.reset()
        except Exception as exc:
            with self._lock:
                self._last_error = exc
                self._state = ContinuousVoiceRuntimeState.ERROR

            raise ContinuousVoiceRuntimeError(
                f"Could not reset continuous voice runtime: {exc}"
            ) from exc

        with self._lock:
            self._last_result = None
            self._last_error = None
            self._state = ContinuousVoiceRuntimeState.STOPPED

    def interrupt(self) -> None:
        """Interrupt active TTS without stopping the worker."""
        try:
            self._runtime.interrupt()
        except Exception as exc:
            with self._lock:
                self._last_error = exc
                self._state = ContinuousVoiceRuntimeState.ERROR

            raise ContinuousVoiceRuntimeError(
                f"Could not interrupt continuous voice runtime: {exc}"
            ) from exc

    def _run_worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                result = self._runtime.run_once()

                if result is None:
                    continue

                with self._lock:
                    self._last_result = result

                if self._on_result is not None:
                    try:
                        self._on_result(result)
                    except Exception as exc:
                        raise ContinuousVoiceRuntimeError(
                            f"Voice result callback failed: {exc}"
                        ) from exc

        except Exception as exc:
            with self._lock:
                self._last_error = exc
                self._state = ContinuousVoiceRuntimeState.ERROR

            self._stop_event.set()

            try:
                self._runtime.stop_capture()
            except Exception:
                pass

            return

        with self._lock:
            self._thread = None

            if self._state != ContinuousVoiceRuntimeState.ERROR:
                self._state = ContinuousVoiceRuntimeState.STOPPED


def create_continuous_voice_runtime(
    runtime: VoiceRuntime,
    *,
    config: ContinuousVoiceRuntimeConfig | None = None,
    on_result: VoiceResultCallback | None = None,
) -> ContinuousVoiceRuntime:
    """Create a continuous voice runtime wrapper."""
    return ContinuousVoiceRuntime(
        runtime,
        config=config,
        on_result=on_result,
    )
