"""Agent-facing voice integration for OSA 0.6.x."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from osa.voice.contracts import (
    VoiceSessionConfig,
)
from osa.voice.session import (
    VoiceSession,
    VoiceSessionError,
    VoiceSessionResult,
)
from osa.voice.stt import SpeechToText
from osa.voice.tts import TextToSpeech
from osa.voice.vad import VoiceActivityDetector


class AgentVoiceIntegrationError(
    RuntimeError
):
    """Raised when Agent voice integration is invalid."""


@runtime_checkable
class VoiceCapableAgent(Protocol):
    """Minimal Agent contract required by the voice integration."""

    def chat(
        self,
        user_input: str,
    ) -> Any:
        ...


class AgentVoiceIntegration:
    """
    Bind an existing Agent to the VoiceSession layer.

    The integration is explicitly opt-in and does not modify the Agent's
    existing chat, streaming, tool, permission, recovery, or autonomous
    execution paths.
    """

    def __init__(
        self,
        agent: VoiceCapableAgent,
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

        if not isinstance(
            agent,
            VoiceCapableAgent,
        ):
            raise TypeError(
                "agent must provide a chat(user_input) method."
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

        try:
            session = VoiceSession(
                agent=agent,
                vad=vad,
                stt=stt,
                tts=tts,
                config=config,
            )
        except Exception as exc:
            raise AgentVoiceIntegrationError(
                "Failed to create voice session."
            ) from exc

        self._agent = agent
        self._session = session

    @property
    def agent(self) -> VoiceCapableAgent:
        """Return the Agent bound to this voice integration."""
        return self._agent

    @property
    def session(self) -> VoiceSession:
        """Return the underlying voice session."""
        return self._session

    @property
    def state(self):
        """Return the current voice session state."""
        return self._session.state

    @property
    def config(self) -> VoiceSessionConfig:
        """Return the voice session configuration."""
        return self._session.config

    def process_audio(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> VoiceSessionResult | None:
        """Send captured audio through VAD, STT, Agent, and TTS."""
        try:
            return self._session.process_audio(
                audio,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )
        except VoiceSessionError:
            raise
        except Exception as exc:
            raise AgentVoiceIntegrationError(
                "Voice processing failed."
            ) from exc

    def complete_speaking(self) -> None:
        """Notify the integration that TTS playback has completed."""
        self._session.complete_speaking()

    def interrupt(self) -> None:
        """Interrupt the current TTS presentation."""
        self._session.interrupt()

    def reset(self) -> None:
        """Reset the voice integration to idle."""
        self._session.reset()

    def stop(self) -> None:
        """Permanently stop the voice integration."""
        self._session.stop()
