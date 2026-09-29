"""Barge-in monitoring for OSA 0.7.7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock, Thread, current_thread
from typing import Callable

from osa.voice.audio import MicrophoneInput
from osa.voice.vad import VoiceActivityDetector


class BargeInError(
    RuntimeError
):
    """Raised when barge-in monitoring cannot operate."""


class BargeInState(str, Enum):
    """Lifecycle state of a barge-in monitor."""

    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


@dataclass(frozen=True)
class BargeInConfig:
    """Configuration for speech monitoring during TTS playback."""

    poll_interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.poll_interval_seconds,
                bool,
            )
            or not isinstance(
                self.poll_interval_seconds,
                (int, float),
            )
        ):
            raise ValueError(
                "poll_interval_seconds must be numeric."
            )

        if self.poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be greater than zero."
            )


BargeInCallback = Callable[[], None]


class VoiceBargeInMonitor:
    """
    Monitor microphone input while TTS playback is active.

    The monitor does not own microphone lifecycle. The surrounding
    VoiceRuntime remains responsible for start/stop. A detected speech event
    invokes the configured callback exactly once per monitoring session.
    """

    def __init__(
        self,
        microphone: MicrophoneInput,
        vad: VoiceActivityDetector,
        on_barge_in: BargeInCallback,
        *,
        config: BargeInConfig | None = None,
    ) -> None:
        if microphone is None:
            raise ValueError(
                "microphone is required."
            )

        if not isinstance(
            microphone,
            MicrophoneInput,
        ):
            raise TypeError(
                "microphone must provide start(), read(), and stop()."
            )

        if vad is None:
            raise ValueError(
                "vad is required."
            )

        if not isinstance(
            vad,
            VoiceActivityDetector,
        ):
            raise TypeError(
                "vad must provide detect() and reset()."
            )

        if not callable(
            on_barge_in
        ):
            raise TypeError(
                "on_barge_in must be callable."
            )

        self._microphone = microphone
        self._vad = vad
        self._on_barge_in = on_barge_in
        self._config = (
            config
            if config is not None
            else BargeInConfig()
        )
        self._state = BargeInState.STOPPED
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_error: Exception | None = None
        self._lock = RLock()

    @property
    def state(self) -> BargeInState:
        """Return the monitor state."""
        with self._lock:
            return self._state

    @property
    def config(self) -> BargeInConfig:
        """Return immutable monitor configuration."""
        return self._config

    @property
    def last_error(self) -> Exception | None:
        """Return the latest monitor error, if any."""
        with self._lock:
            return self._last_error

    @property
    def running(self) -> bool:
        """Return whether the monitor worker is active."""
        return self.state == BargeInState.RUNNING

    def start(self) -> None:
        """Start monitoring microphone input."""
        with self._lock:
            if self._state == BargeInState.RUNNING:
                return

            if self._state == BargeInState.ERROR:
                raise BargeInError(
                    "Barge-in monitor is in an error state; reset it first."
                )

            self._stop_event.clear()
            self._last_error = None

            thread = Thread(
                target=self._run,
                name="osa-barge-in",
                daemon=True,
            )

            self._thread = thread
            self._state = BargeInState.RUNNING

        thread.start()

    def stop(self) -> None:
        """Stop monitoring without changing microphone lifecycle."""
        with self._lock:
            thread = self._thread
            self._stop_event.set()

            if self._state != BargeInState.ERROR:
                self._state = BargeInState.STOPPED

        if (
            thread is not None
            and thread is not current_thread()
        ):
            thread.join(timeout=1.0)

        with self._lock:
            self._thread = None

    def reset(self) -> None:
        """Clear monitor state and return to STOPPED."""
        self.stop()

        with self._lock:
            self._last_error = None
            self._state = BargeInState.STOPPED

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                voice_input = self._microphone.read()

                if self._stop_event.is_set():
                    return

                is_speech = self._vad.detect(
                    voice_input.audio,
                    sample_rate_hz=voice_input.sample_rate_hz,
                    channels=voice_input.channels,
                )

                if not is_speech:
                    self._stop_event.wait(
                        self._config.poll_interval_seconds
                    )
                    continue

                try:
                    self._on_barge_in()
                except Exception as exc:
                    raise BargeInError(
                        f"Barge-in callback failed: {exc}"
                    ) from exc

                self._stop_event.set()
                return

        except Exception as exc:
            if self._stop_event.is_set():
                return

            with self._lock:
                self._last_error = exc
                self._state = BargeInState.ERROR

            self._stop_event.set()
            return

        with self._lock:
            if self._state != BargeInState.ERROR:
                self._state = BargeInState.STOPPED


def create_barge_in_monitor(
    microphone: MicrophoneInput,
    vad: VoiceActivityDetector,
    on_barge_in: BargeInCallback,
    *,
    config: BargeInConfig | None = None,
) -> VoiceBargeInMonitor:
    """Create a provider-neutral barge-in monitor."""
    return VoiceBargeInMonitor(
        microphone,
        vad,
        on_barge_in,
        config=config,
    )
