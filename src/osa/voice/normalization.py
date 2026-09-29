"""Provider-neutral audio normalization for OSA 0.7.4."""

from __future__ import annotations

import io
import math
import wave
from array import array
from dataclasses import dataclass
from struct import pack
from typing import Any

from osa.voice.contracts import VoiceInput


class AudioNormalizationError(
    RuntimeError
):
    """Raised when an audio payload cannot be normalized safely."""


@dataclass(frozen=True)
class AudioNormalizationConfig:
    """Target format for microphone audio consumed by the voice stack."""

    target_sample_rate_hz: int = 16_000
    target_channels: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_sample_rate_hz, bool)
            or not isinstance(
                self.target_sample_rate_hz,
                int,
            )
        ):
            raise ValueError(
                "target_sample_rate_hz must be an integer."
            )

        if self.target_sample_rate_hz <= 0:
            raise ValueError(
                "target_sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(self.target_channels, bool)
            or not isinstance(
                self.target_channels,
                int,
            )
        ):
            raise ValueError(
                "target_channels must be an integer."
            )

        if self.target_channels <= 0:
            raise ValueError(
                "target_channels must be greater than zero."
            )


@dataclass(frozen=True)
class NormalizedAudio:
    """Normalized PCM16 audio plus the format required for playback."""

    audio: bytes
    sample_rate_hz: int
    channels: int
    encoding: str = "pcm_s16le"

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes):
            raise AudioNormalizationError(
                "Normalized audio must be bytes."
            )

        if not self.audio:
            raise AudioNormalizationError(
                "Normalized audio cannot be empty."
            )

        if self.sample_rate_hz <= 0:
            raise AudioNormalizationError(
                "Normalized sample rate must be positive."
            )

        if self.channels <= 0:
            raise AudioNormalizationError(
                "Normalized channels must be positive."
            )

        if self.encoding != "pcm_s16le":
            raise AudioNormalizationError(
                "Normalized audio must use pcm_s16le."
            )


class VoiceAudioNormalizer:
    """
    Normalize provider audio at the runtime I/O boundary.

    Input audio is converted to the STT target format, normally 16 kHz mono
    PCM16. TTS output may be either raw PCM16 or a PCM16 WAV payload. WAV
    containers are unwrapped and converted to the runtime speaker format.
    """

    SUPPORTED_INPUT_ENCODING = "pcm_s16le"
    SUPPORTED_WAV_CODEC = "NONE"
    SUPPORTED_WAV_SAMPLE_WIDTH = 2

    def __init__(
        self,
        config: AudioNormalizationConfig | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else AudioNormalizationConfig()
        )

    @property
    def config(self) -> AudioNormalizationConfig:
        """Return immutable normalization configuration."""
        return self._config

    def normalize_input(
        self,
        voice_input: VoiceInput,
    ) -> VoiceInput:
        """Normalize microphone audio into the STT target format."""
        if not isinstance(
            voice_input,
            VoiceInput,
        ):
            raise AudioNormalizationError(
                "voice_input must be a VoiceInput."
            )

        if voice_input.encoding != self.SUPPORTED_INPUT_ENCODING:
            raise AudioNormalizationError(
                (
                    "Unsupported input encoding "
                    f"'{voice_input.encoding}'. Expected "
                    f"'{self.SUPPORTED_INPUT_ENCODING}'."
                )
            )

        normalized_audio = self.normalize_pcm16(
            voice_input.audio,
            sample_rate_hz=voice_input.sample_rate_hz,
            channels=voice_input.channels,
            target_sample_rate_hz=self._config.target_sample_rate_hz,
            target_channels=self._config.target_channels,
        )

        return VoiceInput(
            audio=normalized_audio.audio,
            sample_rate_hz=normalized_audio.sample_rate_hz,
            channels=normalized_audio.channels,
            encoding=normalized_audio.encoding,
        )

    def normalize_tts_output(
        self,
        audio: bytes,
        *,
        fallback_sample_rate_hz: int,
        fallback_channels: int,
        target_sample_rate_hz: int,
        target_channels: int,
    ) -> NormalizedAudio:
        """
        Normalize a TTS payload for SpeakerOutput.

        Valid PCM16 WAV payloads are unwrapped. Valid raw PCM16 payloads are
        normalized using the fallback metadata. Unknown legacy byte payloads
        are preserved unchanged for backward compatibility with existing
        TTS implementations.
        """
        if not isinstance(
            audio,
            bytes,
        ):
            raise AudioNormalizationError(
                "TTS audio must be bytes."
            )

        if not audio:
            raise AudioNormalizationError(
                "TTS audio cannot be empty."
            )

        if _looks_like_wav(audio):
            return self._normalize_wav(
                audio,
                target_sample_rate_hz=target_sample_rate_hz,
                target_channels=target_channels,
            )

        if _valid_raw_pcm16(
            audio,
            channels=fallback_channels,
        ):
            return self.normalize_pcm16(
                audio,
                sample_rate_hz=fallback_sample_rate_hz,
                channels=fallback_channels,
                target_sample_rate_hz=target_sample_rate_hz,
                target_channels=target_channels,
            )

        # Existing OSA TTS adapters may return opaque bytes. Do not break
        # that public behavior before every provider exposes audio metadata.
        return NormalizedAudio(
            audio=audio,
            sample_rate_hz=fallback_sample_rate_hz,
            channels=fallback_channels,
        )

    def normalize_pcm16(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
        target_sample_rate_hz: int,
        target_channels: int,
    ) -> NormalizedAudio:
        """Convert PCM16 LE audio between sample rates and channel layouts."""
        self._validate_pcm16_metadata(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        self._validate_pcm16_metadata(
            sample_rate_hz=target_sample_rate_hz,
            channels=target_channels,
        )

        if not isinstance(
            audio,
            bytes,
        ):
            raise AudioNormalizationError(
                "PCM16 audio must be bytes."
            )

        if not audio:
            raise AudioNormalizationError(
                "PCM16 audio cannot be empty."
            )

        frame_width = channels * 2

        if len(audio) % frame_width:
            raise AudioNormalizationError(
                (
                    "PCM16 payload size is not aligned to the configured "
                    "channel count."
                )
            )

        if (
            sample_rate_hz == target_sample_rate_hz
            and channels == target_channels
        ):
            return NormalizedAudio(
                audio=audio,
                sample_rate_hz=target_sample_rate_hz,
                channels=target_channels,
            )

        frames = _decode_pcm16le(
            audio,
            channels=channels,
        )

        converted_channels = _convert_channels(
            frames,
            source_channels=channels,
            target_channels=target_channels,
        )

        resampled = _resample_frames(
            converted_channels,
            source_sample_rate_hz=sample_rate_hz,
            target_sample_rate_hz=target_sample_rate_hz,
        )

        encoded = _encode_pcm16le(resampled)

        return NormalizedAudio(
            audio=encoded,
            sample_rate_hz=target_sample_rate_hz,
            channels=target_channels,
        )

    def _normalize_wav(
        self,
        audio: bytes,
        *,
        target_sample_rate_hz: int,
        target_channels: int,
    ) -> NormalizedAudio:
        """Read a PCM16 WAV container and normalize its payload."""
        try:
            with wave.open(
                io.BytesIO(audio),
                "rb",
            ) as wav_file:
                sample_rate_hz = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                compression = wav_file.getcomptype()
                frame_count = wav_file.getnframes()
                raw_audio = wav_file.readframes(
                    frame_count
                )

        except (wave.Error, EOFError) as exc:
            raise AudioNormalizationError(
                f"Invalid WAV audio payload: {exc}"
            ) from exc

        if compression != self.SUPPORTED_WAV_CODEC:
            raise AudioNormalizationError(
                (
                    "Unsupported WAV compression "
                    f"'{compression}'. Only PCM WAV is supported."
                )
            )

        if sample_width != self.SUPPORTED_WAV_SAMPLE_WIDTH:
            raise AudioNormalizationError(
                (
                    "Unsupported WAV sample width "
                    f"{sample_width * 8} bits. "
                    "Only PCM16 WAV is supported."
                )
            )

        return self.normalize_pcm16(
            raw_audio,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            target_sample_rate_hz=target_sample_rate_hz,
            target_channels=target_channels,
        )

    @staticmethod
    def _validate_pcm16_metadata(
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(
                sample_rate_hz,
                int,
            )
        ):
            raise AudioNormalizationError(
                "sample_rate_hz must be an integer."
            )

        if sample_rate_hz <= 0:
            raise AudioNormalizationError(
                "sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(channels, bool)
            or not isinstance(
                channels,
                int,
            )
        ):
            raise AudioNormalizationError(
                "channels must be an integer."
            )

        if channels <= 0:
            raise AudioNormalizationError(
                "channels must be greater than zero."
            )


def create_voice_audio_normalizer(
    config: AudioNormalizationConfig | None = None,
) -> VoiceAudioNormalizer:
    """Create a provider-neutral audio normalizer."""
    return VoiceAudioNormalizer(config=config)


def _looks_like_wav(audio: bytes) -> bool:
    return (
        len(audio) >= 12
        and audio[:4] == b"RIFF"
        and audio[8:12] == b"WAVE"
    )


def _valid_raw_pcm16(
    audio: bytes,
    *,
    channels: int,
) -> bool:
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels <= 0
    ):
        return False

    frame_width = channels * 2
    return bool(audio) and len(audio) % frame_width == 0


def _decode_pcm16le(
    audio: bytes,
    *,
    channels: int,
) -> list[list[int]]:
    samples = array("h")
    samples.frombytes(audio)

    if _needs_byteswap():
        samples.byteswap()

    frame_count = len(samples) // channels

    return [
        [
            int(samples[
                frame_index * channels + channel_index
            ])
            for channel_index in range(channels)
        ]
        for frame_index in range(frame_count)
    ]


def _convert_channels(
    frames: list[list[int]],
    *,
    source_channels: int,
    target_channels: int,
) -> list[list[int]]:
    if source_channels == target_channels:
        return [
            list(frame)
            for frame in frames
        ]

    converted: list[list[int]] = []

    for frame in frames:
        if target_channels == 1:
            average = round(
                sum(frame) / source_channels
            )
            converted.append(
                [_clip_pcm16(average)]
            )
            continue

        if source_channels == 1:
            converted.append(
                [
                    frame[0]
                    for _ in range(target_channels)
                ]
            )
            continue

        output_frame: list[int] = []

        for target_index in range(target_channels):
            start = math.floor(
                target_index * source_channels / target_channels
            )
            end = math.floor(
                (target_index + 1)
                * source_channels
                / target_channels
            )

            end = max(
                end,
                start + 1,
            )

            selected = frame[
                start:min(end, source_channels)
            ]

            output_frame.append(
                _clip_pcm16(
                    round(
                        sum(selected) / len(selected)
                    )
                )
            )

        converted.append(output_frame)

    return converted


def _resample_frames(
    frames: list[list[int]],
    *,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> list[list[int]]:
    if not frames:
        raise AudioNormalizationError(
            "Cannot resample empty PCM audio."
        )

    if source_sample_rate_hz == target_sample_rate_hz:
        return [
            list(frame)
            for frame in frames
        ]

    source_frame_count = len(frames)
    channel_count = len(frames[0])

    target_frame_count = max(
        1,
        int(
            round(
                source_frame_count
                * target_sample_rate_hz
                / source_sample_rate_hz
            )
        ),
    )

    ratio = (
        source_sample_rate_hz
        / target_sample_rate_hz
    )

    output: list[list[int]] = []

    for target_index in range(target_frame_count):
        position = target_index * ratio
        left_index = int(math.floor(position))
        fraction = position - left_index

        if left_index >= source_frame_count - 1:
            left_index = source_frame_count - 1
            right_index = left_index
            fraction = 0.0
        else:
            right_index = left_index + 1

        frame: list[int] = []

        for channel_index in range(channel_count):
            left = frames[left_index][channel_index]
            right = frames[right_index][channel_index]

            value = round(
                left
                + (
                    right - left
                )
                * fraction
            )

            frame.append(
                _clip_pcm16(value)
            )

        output.append(frame)

    return output


def _encode_pcm16le(
    frames: list[list[int]],
) -> bytes:
    if not frames:
        raise AudioNormalizationError(
            "Cannot encode empty PCM audio."
        )

    output = bytearray()

    for frame in frames:
        for sample in frame:
            output.extend(
                pack(
                    "<h",
                    _clip_pcm16(sample),
                )
            )

    return bytes(output)


def _clip_pcm16(value: int) -> int:
    return max(
        -32768,
        min(
            32767,
            int(value),
        ),
    )


def _needs_byteswap() -> bool:
    import sys

    return sys.byteorder != "little"
