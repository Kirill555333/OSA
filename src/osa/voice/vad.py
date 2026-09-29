"""Voice Activity Detection interfaces for OSA 0.6.x."""

from __future__ import annotations

import math
import struct
from typing import Protocol, runtime_checkable

from osa.voice.contracts import VoiceContractError


class VoiceActivityDetectionError(
    RuntimeError
):
    """Raised when voice activity detection fails."""


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """
    Provider-neutral interface for voice activity detection.

    Implementations receive independent PCM audio chunks and return whether
    speech is currently detected in that chunk.
    """

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        ...

    def reset(self) -> None:
        ...


class EnergyVoiceActivityDetector:
    """
    Deterministic PCM16 energy-based VAD.

    This implementation is intentionally lightweight and provider-neutral.
    It is suitable for local development and testing; production deployments
    may replace it with a platform/provider-specific VAD implementation
    without changing the VoiceSession API.
    """

    def __init__(
        self,
        *,
        threshold: float = 500.0,
    ) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(
                threshold,
                (int, float),
            )
        ):
            raise VoiceContractError(
                "threshold must be numeric."
            )

        normalized_threshold = float(threshold)

        if not math.isfinite(
            normalized_threshold
        ):
            raise VoiceContractError(
                "threshold must be finite."
            )

        if normalized_threshold <= 0:
            raise VoiceContractError(
                "threshold must be greater than zero."
            )

        self._threshold = normalized_threshold

    @property
    def threshold(self) -> float:
        """Return the configured RMS speech threshold."""
        return self._threshold

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        """
        Return whether a PCM16 little-endian chunk contains speech energy.

        The sample rate is validated as part of the public VAD contract even
        though this simple energy detector does not currently resample.
        """
        self._validate_audio_arguments(
            audio,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )

        if not audio:
            return False

        if len(audio) % 2 != 0:
            raise VoiceActivityDetectionError(
                "PCM16 audio must contain an even number of bytes."
            )

        samples = struct.unpack(
            f"<{len(audio) // 2}h",
            audio,
        )

        if not samples:
            return False

        mean_square = sum(
            sample * sample
            for sample in samples
        ) / len(samples)

        rms = math.sqrt(mean_square)

        return rms >= self._threshold

    def reset(self) -> None:
        """Reset detector state."""
        return None

    @staticmethod
    def _validate_audio_arguments(
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        if not isinstance(audio, bytes):
            raise VoiceActivityDetectionError(
                "audio must be bytes."
            )

        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(
                sample_rate_hz,
                int,
            )
        ):
            raise VoiceActivityDetectionError(
                "sample_rate_hz must be an integer."
            )

        if sample_rate_hz <= 0:
            raise VoiceActivityDetectionError(
                "sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(channels, bool)
            or not isinstance(
                channels,
                int,
            )
        ):
            raise VoiceActivityDetectionError(
                "channels must be an integer."
            )

        if channels <= 0:
            raise VoiceActivityDetectionError(
                "channels must be greater than zero."
            )
