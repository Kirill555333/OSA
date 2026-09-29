"""Speech-to-text interfaces for OSA 0.6.x."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from osa.voice.contracts import VoiceInput, VoiceTranscript


class SpeechToTextError(
    RuntimeError
):
    """Raised when speech-to-text processing fails."""


@runtime_checkable
class SpeechToText(Protocol):
    """Provider-neutral speech-to-text interface."""

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        ...


class FixedSpeechToText:
    """
    Deterministic STT implementation for development and testing.

    It does not inspect or decode audio. A fixed transcript is returned for
    every valid VoiceInput.
    """

    def __init__(
        self,
        transcript: VoiceTranscript,
    ) -> None:
        if not isinstance(
            transcript,
            VoiceTranscript,
        ):
            raise TypeError(
                "transcript must be a VoiceTranscript."
            )

        self._transcript = transcript

    @property
    def transcript(self) -> VoiceTranscript:
        """Return the configured deterministic transcript."""
        return self._transcript

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        """Return the configured transcript for valid voice input."""
        if not isinstance(
            voice_input,
            VoiceInput,
        ):
            raise TypeError(
                "voice_input must be a VoiceInput."
            )

        return self._transcript
