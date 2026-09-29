from __future__ import annotations

import io
import wave

import pytest

from osa.voice.contracts import VoiceInput
from osa.voice.normalization import (
    AudioNormalizationConfig,
    AudioNormalizationError,
    VoiceAudioNormalizer,
)


def pcm16(values: list[int]) -> bytes:
    import struct

    return b"".join(
        struct.pack("<h", value)
        for value in values
    )


def pcm16_frames(frames: list[list[int]]) -> bytes:
    return b"".join(
        pcm16(frame)
        for frame in frames
    )


def make_wav(
    audio: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
) -> bytes:
    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(audio)

    return buffer.getvalue()


def test_normalizer_defaults_to_16khz_mono() -> None:
    normalizer = VoiceAudioNormalizer()

    assert normalizer.config == AudioNormalizationConfig(
        target_sample_rate_hz=16_000,
        target_channels=1,
    )


def test_input_is_left_unchanged_when_already_normalized() -> None:
    audio = VoiceInput(
        audio=pcm16([100, -100]),
        sample_rate_hz=16_000,
        channels=1,
    )

    result = VoiceAudioNormalizer().normalize_input(audio)

    assert result == audio


def test_stereo_input_is_downmixed_to_mono() -> None:
    audio = VoiceInput(
        audio=pcm16_frames(
            [
                [1000, 3000],
                [-1000, -3000],
            ]
        ),
        sample_rate_hz=16_000,
        channels=2,
    )

    result = VoiceAudioNormalizer().normalize_input(audio)

    assert result.sample_rate_hz == 16_000
    assert result.channels == 1
    assert result.audio == pcm16(
        [2000, -2000]
    )


def test_8khz_input_is_resampled_to_16khz() -> None:
    audio = VoiceInput(
        audio=pcm16([0, 1000, 2000, 3000]),
        sample_rate_hz=8_000,
        channels=1,
    )

    result = VoiceAudioNormalizer().normalize_input(audio)

    assert result.sample_rate_hz == 16_000
    assert result.channels == 1
    assert len(result.audio) == 8 * 2
    assert result.audio[:4] == pcm16(
        [0, 500]
    )


def test_input_rejects_unsupported_encoding() -> None:
    audio = VoiceInput(
        audio=b"\x00\x00",
        sample_rate_hz=16_000,
        channels=1,
        encoding="pcm_s16be",
    )

    with pytest.raises(
        AudioNormalizationError,
        match="Unsupported input encoding",
    ):
        VoiceAudioNormalizer().normalize_input(audio)


def test_pcm16_requires_frame_alignment() -> None:
    with pytest.raises(
        AudioNormalizationError,
        match="not aligned",
    ):
        VoiceAudioNormalizer().normalize_pcm16(
            b"\x00",
            sample_rate_hz=16_000,
            channels=1,
            target_sample_rate_hz=16_000,
            target_channels=1,
        )


def test_wav_output_is_unwrapped_and_normalized() -> None:
    wav = make_wav(
        pcm16([100, -100]),
        sample_rate_hz=22_050,
        channels=1,
    )

    result = VoiceAudioNormalizer().normalize_tts_output(
        wav,
        fallback_sample_rate_hz=16_000,
        fallback_channels=1,
        target_sample_rate_hz=16_000,
        target_channels=1,
    )

    assert result.sample_rate_hz == 16_000
    assert result.channels == 1
    assert result.audio[:4] == pcm16(
        [100]
    )[:4]
    assert not result.audio.startswith(b"RIFF")


def test_wav_stereo_output_is_converted_to_mono() -> None:
    wav = make_wav(
        pcm16_frames(
            [
                [1000, 3000],
                [-1000, -3000],
            ]
        ),
        sample_rate_hz=16_000,
        channels=2,
    )

    result = VoiceAudioNormalizer().normalize_tts_output(
        wav,
        fallback_sample_rate_hz=16_000,
        fallback_channels=1,
        target_sample_rate_hz=16_000,
        target_channels=1,
    )

    assert result.channels == 1
    assert result.audio == pcm16(
        [2000, -2000]
    )


def test_unknown_legacy_tts_payload_is_preserved() -> None:
    result = VoiceAudioNormalizer().normalize_tts_output(
        b"speech",
        fallback_sample_rate_hz=16_000,
        fallback_channels=1,
        target_sample_rate_hz=16_000,
        target_channels=1,
    )

    assert result.audio == b"speech"
    assert result.sample_rate_hz == 16_000
    assert result.channels == 1


def test_malformed_wav_is_rejected() -> None:
    payload = b"RIFFxxxxWAVEbroken"

    with pytest.raises(
        AudioNormalizationError,
        match="Invalid WAV",
    ):
        VoiceAudioNormalizer().normalize_tts_output(
            payload,
            fallback_sample_rate_hz=16_000,
            fallback_channels=1,
            target_sample_rate_hz=16_000,
            target_channels=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_sample_rate_hz", 0),
        ("target_sample_rate_hz", -1),
        ("target_channels", 0),
        ("target_channels", -1),
    ],
)
def test_invalid_normalizer_config_is_rejected(
    field: str,
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=field,
    ):
        AudioNormalizationConfig(
            **{field: value}
        )


def test_normalizer_rejects_invalid_voice_input_type() -> None:
    with pytest.raises(
        AudioNormalizationError,
        match="VoiceInput",
    ):
        VoiceAudioNormalizer().normalize_input(
            object()
        )
