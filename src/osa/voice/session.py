"""Voice session orchestration for OSA 0.6.x."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceSessionConfig,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.stt import SpeechToText
from osa.voice.tts import TextToSpeech
from osa.voice.vad import VoiceActivityDetector


class VoiceSessionError(
    RuntimeError
):
    """Raised when a voice session cannot complete an operation."""


class VoiceSessionStateError(
    VoiceSessionError
):
    """Raised when an operation is invalid for the current state."""


@runtime_checkable
class VoiceAgent(Protocol):
    """Minimal Agent interface required by the voice layer."""

    def chat(
        self,
        user_input: str,
    ) -> Any:
        ...


@dataclass(frozen=True)
class VoiceSessionResult:
    """Result produced when a voice utterance reaches TTS synthesis."""

    transcript: VoiceTranscript
    response: VoiceOutput
    audio: bytes


class VoiceSession:
    """
    Coordinate VAD, STT, Agent, and TTS.

    The session does not own a microphone or speaker. Audio capture and
    playback remain platform/interface concerns outside this orchestration
    layer.
    """

    def __init__(
        self,
        agent: VoiceAgent,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        *,
        config: VoiceSessionConfig | None = None,
    ) -> None:
        if agent is None:
            raise ValueError(
                "agent is required."
            )

        if vad is None:
            raise ValueError(
                "vad is required."
            )

        if stt is None:
            raise ValueError(
                "stt is required."
            )

        if tts is None:
            raise ValueError(
                "tts is required."
            )

        self._agent = agent
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._config = (
            config
            if config is not None
            else VoiceSessionConfig()
        )
        self._state = VoiceSessionState.IDLE

    @property
    def agent(self) -> VoiceAgent:
        """Return the configured Agent."""
        return self._agent

    @property
    def vad(self) -> VoiceActivityDetector:
        """Return the configured VAD."""
        return self._vad

    @property
    def stt(self) -> SpeechToText:
        """Return the configured STT adapter."""
        return self._stt

    @property
    def tts(self) -> TextToSpeech:
        """Return the configured TTS adapter."""
        return self._tts

    @property
    def config(self) -> VoiceSessionConfig:
        """Return the immutable voice session configuration."""
        return self._config

    @property
    def state(self) -> VoiceSessionState:
        """Return the current session state."""
        return self._state

    def process_audio(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> VoiceSessionResult | None:
        """
        Process one complete audio chunk.

        Non-speech returns None and leaves the session idle.

        Speech follows:
            IDLE -> LISTENING -> PROCESSING -> SPEAKING

        SPEAKING is intentionally retained until the caller confirms that
        playback has completed with complete_speaking(), or interrupts it
        with interrupt().
        """
        self._require_state(
            VoiceSessionState.IDLE
        )

        self._state = VoiceSessionState.LISTENING

        try:
            is_speech = self._vad.detect(
                audio,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )

            if not is_speech:
                self._state = VoiceSessionState.IDLE
                return None

            self._state = VoiceSessionState.PROCESSING

            voice_input = VoiceInput(
                audio=audio,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )

            transcript = self._stt.transcribe(
                voice_input
            )

            response = self._agent.chat(
                transcript.text
            )

            response_text = self._extract_response_text(
                response
            )

            output = VoiceOutput(
                text=response_text,
                language=(
                    transcript.language
                    if transcript.language is not None
                    else self._config.language
                    if self._config.language != "auto"
                    else None
                ),
            )

            self._state = VoiceSessionState.SPEAKING

            synthesized_audio = self._tts.synthesize(
                output
            )

            if not isinstance(
                synthesized_audio,
                bytes,
            ):
                raise VoiceSessionError(
                    "TTS returned an invalid audio payload."
                )

            if not synthesized_audio:
                raise VoiceSessionError(
                    "TTS returned an empty audio payload."
                )

            return VoiceSessionResult(
                transcript=transcript,
                response=output,
                audio=synthesized_audio,
            )

        except Exception:
            self._state = VoiceSessionState.ERROR
            raise

    def complete_speaking(self) -> None:
        """
        Mark current speech playback as completed.

        No speaker is invoked here; the speaker remains outside this layer.
        """
        self._require_state(
            VoiceSessionState.SPEAKING
        )

        self._state = VoiceSessionState.IDLE

    def interrupt(self) -> None:
        """
        Stop current speech presentation.

        Interruption only affects TTS presentation. It does not cancel an
        already completed Agent operation or alter any action result.
        """
        self._require_state(
            VoiceSessionState.SPEAKING
        )

        self._state = VoiceSessionState.STOPPING

        try:
            self._tts.stop()
        except Exception:
            self._state = VoiceSessionState.ERROR
            raise

        self._state = VoiceSessionState.IDLE

    def reset(self) -> None:
        """Reset the session and underlying VAD state to idle."""
        if self._state == VoiceSessionState.SPEAKING:
            try:
                self._tts.stop()
            except Exception:
                self._state = VoiceSessionState.ERROR
                raise

        self._vad.reset()
        self._state = VoiceSessionState.IDLE

    def stop(self) -> None:
        """
        Stop the session.

        SPEAKING is interrupted before entering STOPPED.
        """
        if self._state == VoiceSessionState.STOPPED:
            return

        if self._state == VoiceSessionState.SPEAKING:
            try:
                self._tts.stop()
            except Exception:
                self._state = VoiceSessionState.ERROR
                raise

        self._state = VoiceSessionState.STOPPED

    def _require_state(
        self,
        expected: VoiceSessionState,
    ) -> None:
        if self._state != expected:
            raise VoiceSessionStateError(
                (
                    "Operation requires voice session state "
                    f"'{expected.value}', but current state is "
                    f"'{self._state.value}'."
                )
            )

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:
        content = getattr(
            response,
            "content",
            None,
        )

        if not isinstance(
            content,
            str,
        ):
            raise VoiceSessionError(
                "Agent response must expose string content."
            )

        if not content.strip():
            raise VoiceSessionError(
                "Agent response content cannot be empty."
            )

        return content.strip()
