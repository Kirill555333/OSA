"""Real audio runtime integration for OSA voice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock

from osa.voice.audio import (
    MicrophoneInput,
    SpeakerOutput,
)
from osa.voice.session import (
    VoiceSession,
    VoiceSessionResult,
)


class VoiceRuntimeError(
    RuntimeError
):
    """Raised when the real voice runtime cannot operate."""


class VoiceRuntimeState(str, Enum):
    """Lifecycle state of the audio runtime."""

    STOPPED = "stopped"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceRuntimeConfig:
    """Configuration for runtime speaker playback."""

    output_sample_rate_hz: int = 16_000
    output_channels: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.output_sample_rate_hz,
                bool,
            )
            or not isinstance(
                self.output_sample_rate_hz,
                int,
            )
        ):
            raise ValueError(
                "output_sample_rate_hz must be an integer."
            )

        if self.output_sample_rate_hz <= 0:
            raise ValueError(
                "output_sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(
                self.output_channels,
                bool,
            )
            or not isinstance(
                self.output_channels,
                int,
            )
        ):
            raise ValueError(
                "output_channels must be an integer."
            )

        if self.output_channels <= 0:
            raise ValueError(
                "output_channels must be greater than zero."
            )


class VoiceRuntime:
    """
    Bridge VoiceSession with actual microphone and speaker devices.

    The runtime owns device lifecycle. VoiceSession remains responsible for
    VAD, STT, Agent, TTS, and voice state semantics.
    """

    def __init__(
        self,
        session: VoiceSession,
        microphone: MicrophoneInput,
        speaker: SpeakerOutput,
        *,
        config: VoiceRuntimeConfig | None = None,
    ) -> None:
        if session is None:
            raise ValueError(
                "session is required."
            )

        if not isinstance(
            session,
            VoiceSession,
        ):
            raise TypeError(
                "session must be a VoiceSession."
            )

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

        if speaker is None:
            raise ValueError(
                "speaker is required."
            )

        if not isinstance(
            speaker,
            SpeakerOutput,
        ):
            raise TypeError(
                "speaker must provide play() and stop()."
            )

        self._session = session
        self._microphone = microphone
        self._speaker = speaker
        self._config = (
            config
            if config is not None
            else VoiceRuntimeConfig()
        )
        self._state = VoiceRuntimeState.STOPPED
        self._lock = RLock()

    @property
    def session(self) -> VoiceSession:
        """Return the configured voice session."""
        return self._session

    @property
    def microphone(self) -> MicrophoneInput:
        """Return the configured microphone."""
        return self._microphone

    @property
    def speaker(self) -> SpeakerOutput:
        """Return the configured speaker."""
        return self._speaker

    @property
    def config(self) -> VoiceRuntimeConfig:
        """Return immutable runtime configuration."""
        return self._config

    @property
    def state(self) -> VoiceRuntimeState:
        """Return the runtime state."""
        with self._lock:
            return self._state

    def start(self) -> None:
        """Start microphone capture and make the runtime ready."""
        with self._lock:
            if self._state in {
                VoiceRuntimeState.READY,
                VoiceRuntimeState.RUNNING,
            }:
                return

            if self._state == VoiceRuntimeState.ERROR:
                raise VoiceRuntimeError(
                    "Voice runtime is in an error state; reset it first."
                )

            try:
                self._microphone.start()
            except Exception:
                self._state = VoiceRuntimeState.ERROR
                raise

            self._state = VoiceRuntimeState.READY

    def listen_once(
        self,
    ) -> VoiceSessionResult | None:
        """
        Read one microphone chunk and process it through VoiceSession.

        The runtime must be started first.
        """
        with self._lock:
            if self._state != VoiceRuntimeState.READY:
                raise VoiceRuntimeError(
                    (
                        "Voice runtime must be READY before listening; "
                        f"current state is '{self._state.value}'."
                    )
                )

            self._state = VoiceRuntimeState.RUNNING

        try:
            voice_input = self._microphone.read()

            result = self._session.process_audio(
                voice_input.audio,
                sample_rate_hz=voice_input.sample_rate_hz,
                channels=voice_input.channels,
            )

            with self._lock:
                self._state = VoiceRuntimeState.READY

            return result

        except Exception:
            with self._lock:
                self._state = VoiceRuntimeState.ERROR

            raise

    def speak(
        self,
        result: VoiceSessionResult,
    ) -> None:
        """
        Play a processed VoiceSessionResult through the speaker.

        Successful playback completes the VoiceSession speaking state.
        """
        if not isinstance(
            result,
            VoiceSessionResult,
        ):
            raise TypeError(
                "result must be a VoiceSessionResult."
            )

        if self._session.state.value != "speaking":
            raise VoiceRuntimeError(
                (
                    "VoiceSession must be SPEAKING before playback; "
                    f"current state is '{self._session.state.value}'."
                )
            )

        try:
            self._speaker.play(
                result.audio,
                sample_rate_hz=self._config.output_sample_rate_hz,
                channels=self._config.output_channels,
            )

            self._session.complete_speaking()

            with self._lock:
                self._state = VoiceRuntimeState.READY

        except Exception:
            with self._lock:
                self._state = VoiceRuntimeState.ERROR

            raise

    def run_once(
        self,
    ) -> VoiceSessionResult | None:
        """
        Capture, process, and synchronously play one utterance.

        Non-speech returns None. Speech is sent to the speaker and the
        VoiceSession returns to idle after successful playback.
        """
        result = self.listen_once()

        if result is None:
            return None

        self.speak(result)

        return result

    def interrupt(self) -> None:
        """
        Stop current voice presentation and speaker playback.

        This operation never modifies the completed Agent result.
        """
        try:
            if self._session.state.value == "speaking":
                self._session.interrupt()

            self._speaker.stop()

        except Exception:
            with self._lock:
                self._state = VoiceRuntimeState.ERROR

            raise

        with self._lock:
            if self._state != VoiceRuntimeState.STOPPED:
                self._state = VoiceRuntimeState.READY

    def reset(self) -> None:
        """Reset the session and return the runtime to READY."""
        try:
            self._speaker.stop()
            self._session.reset()

            with self._lock:
                self._state = VoiceRuntimeState.READY

        except Exception:
            with self._lock:
                self._state = VoiceRuntimeState.ERROR

            raise

    def stop(self) -> None:
        """Stop speaker playback, microphone capture, and voice session."""
        microphone_error: Exception | None = None
        speaker_error: Exception | None = None
        session_error: Exception | None = None

        try:
            self._speaker.stop()
        except Exception as exc:
            speaker_error = exc

        try:
            self._microphone.stop()
        except Exception as exc:
            microphone_error = exc

        try:
            self._session.stop()
        except Exception as exc:
            session_error = exc

        with self._lock:
            self._state = (
                VoiceRuntimeState.ERROR
                if (
                    microphone_error is not None
                    or speaker_error is not None
                    or session_error is not None
                )
                else VoiceRuntimeState.STOPPED
            )

        first_error = (
            microphone_error
            or speaker_error
            or session_error
        )

        if first_error is not None:
            raise first_error


def create_voice_runtime(
    session: VoiceSession,
    microphone: MicrophoneInput,
    speaker: SpeakerOutput,
    *,
    config: VoiceRuntimeConfig | None = None,
) -> VoiceRuntime:
    """Create a configured voice runtime."""
    return VoiceRuntime(
        session,
        microphone,
        speaker,
        config=config,
    )
