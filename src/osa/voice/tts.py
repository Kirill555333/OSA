"""Text-to-speech interfaces for OSA 0.6.x."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from osa.voice.contracts import VoiceOutput


class TextToSpeechError(
    RuntimeError
):
    """Raised when text-to-speech processing fails."""


@runtime_checkable
class TextToSpeech(Protocol):
    """Provider-neutral text-to-speech interface."""

    def synthesize(
        self,
        output: VoiceOutput,
    ) -> bytes:
        ...

    def stop(self) -> None:
        ...


class FixedTextToSpeech:
    """
    Deterministic TTS implementation for development and testing.

    It does not synthesize real speech. It returns a fixed byte payload for
    every valid VoiceOutput and records stop requests.
    """

    def __init__(
        self,
        audio: bytes = b"voice-audio",
    ) -> None:
        if not isinstance(audio, bytes):
            raise TypeError(
                "audio must be bytes."
            )

        if not audio:
            raise ValueError(
                "audio cannot be empty."
            )

        self._audio = audio
        self._stop_calls = 0

    @property
    def audio(self) -> bytes:
        """Return the deterministic audio payload."""
        return self._audio

    @property
    def stop_calls(self) -> int:
        """Return the number of stop requests."""
        return self._stop_calls

    def synthesize(
        self,
        output: VoiceOutput,
    ) -> bytes:
        """Return deterministic audio for valid voice output."""
        if not isinstance(
            output,
            VoiceOutput,
        ):
            raise TypeError(
                "output must be a VoiceOutput."
            )

        return self._audio

    def stop(self) -> None:
        """Record a request to stop current speech."""
        self._stop_calls += 1
