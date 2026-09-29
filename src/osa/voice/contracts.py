"""Immutable contracts for the OSA voice interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class VoiceContractError(ValueError):
    """Raised when a voice contract contains invalid data."""


class VoiceSessionState(str, Enum):
    """Lifecycle states of a voice session."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceInput:
    """
    Immutable captured voice input.

    The object represents a complete audio payload that can be passed to
    a speech-to-text adapter. Provider-specific decoding remains outside
    this contract.
    """

    audio: bytes
    sample_rate_hz: int
    channels: int = 1
    encoding: str = "pcm_s16le"

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes):
            raise VoiceContractError(
                "audio must be bytes."
            )

        if not self.audio:
            raise VoiceContractError(
                "audio cannot be empty."
            )

        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
        ):
            raise VoiceContractError(
                "sample_rate_hz must be an integer."
            )

        if self.sample_rate_hz <= 0:
            raise VoiceContractError(
                "sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
        ):
            raise VoiceContractError(
                "channels must be an integer."
            )

        if self.channels <= 0:
            raise VoiceContractError(
                "channels must be greater than zero."
            )

        if not isinstance(self.encoding, str):
            raise VoiceContractError(
                "encoding must be a string."
            )

        if not self.encoding.strip():
            raise VoiceContractError(
                "encoding cannot be empty."
            )

        object.__setattr__(
            self,
            "encoding",
            self.encoding.strip().lower(),
        )


@dataclass(frozen=True)
class VoiceTranscript:
    """Immutable speech-to-text result."""

    text: str
    language: str | None = None
    confidence: float | None = None
    is_final: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise VoiceContractError(
                "text must be a string."
            )

        normalized_text = self.text.strip()

        if not normalized_text:
            raise VoiceContractError(
                "text cannot be empty."
            )

        if self.language is not None:
            if not isinstance(self.language, str):
                raise VoiceContractError(
                    "language must be a string or None."
                )

            normalized_language = self.language.strip().lower()

            if not normalized_language:
                raise VoiceContractError(
                    "language cannot be empty."
                )

            object.__setattr__(
                self,
                "language",
                normalized_language,
            )

        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(
                    self.confidence,
                    (int, float),
                )
            ):
                raise VoiceContractError(
                    "confidence must be numeric or None."
                )

            confidence = float(self.confidence)

            if not math.isfinite(confidence):
                raise VoiceContractError(
                    "confidence must be finite."
                )

            if not 0.0 <= confidence <= 1.0:
                raise VoiceContractError(
                    "confidence must be between 0.0 and 1.0."
                )

            object.__setattr__(
                self,
                "confidence",
                confidence,
            )

        if not isinstance(self.is_final, bool):
            raise VoiceContractError(
                "is_final must be boolean."
            )

        object.__setattr__(
            self,
            "text",
            normalized_text,
        )


@dataclass(frozen=True)
class VoiceOutput:
    """Immutable text response that the voice layer may send to TTS."""

    text: str
    language: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise VoiceContractError(
                "text must be a string."
            )

        normalized_text = self.text.strip()

        if not normalized_text:
            raise VoiceContractError(
                "text cannot be empty."
            )

        if self.language is not None:
            if not isinstance(self.language, str):
                raise VoiceContractError(
                    "language must be a string or None."
                )

            normalized_language = self.language.strip().lower()

            if not normalized_language:
                raise VoiceContractError(
                    "language cannot be empty."
                )

            object.__setattr__(
                self,
                "language",
                normalized_language,
            )

        object.__setattr__(
            self,
            "text",
            normalized_text,
        )


@dataclass(frozen=True)
class VoiceSessionConfig:
    """Immutable provider-neutral configuration for a voice session."""

    language: str = "auto"
    max_utterance_seconds: float = 30.0
    interrupt_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.language, str):
            raise VoiceContractError(
                "language must be a string."
            )

        normalized_language = self.language.strip().lower()

        if not normalized_language:
            raise VoiceContractError(
                "language cannot be empty."
            )

        if (
            isinstance(
                self.max_utterance_seconds,
                bool,
            )
            or not isinstance(
                self.max_utterance_seconds,
                (int, float),
            )
        ):
            raise VoiceContractError(
                "max_utterance_seconds must be numeric."
            )

        max_utterance_seconds = float(
            self.max_utterance_seconds
        )

        if not math.isfinite(
            max_utterance_seconds
        ):
            raise VoiceContractError(
                "max_utterance_seconds must be finite."
            )

        if max_utterance_seconds <= 0:
            raise VoiceContractError(
                "max_utterance_seconds must be greater than zero."
            )

        if not isinstance(
            self.interrupt_enabled,
            bool,
        ):
            raise VoiceContractError(
                "interrupt_enabled must be boolean."
            )

        object.__setattr__(
            self,
            "language",
            normalized_language,
        )
        object.__setattr__(
            self,
            "max_utterance_seconds",
            max_utterance_seconds,
        )
