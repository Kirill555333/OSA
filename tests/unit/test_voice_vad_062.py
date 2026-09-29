from __future__ import annotations

import struct

import pytest

from osa.voice.vad import (
    EnergyVoiceActivityDetector,
    VoiceActivityDetectionError,
    VoiceActivityDetector,
)


def _pcm16(
    *samples: int,
) -> bytes:
    return struct.pack(
        f"<{len(samples)}h",
        *samples,
    )


def test_energy_vad_satisfies_protocol_shape():
    detector = EnergyVoiceActivityDetector()

    assert isinstance(
        detector,
        VoiceActivityDetector,
    )


def test_silence_is_not_detected_as_speech():
    detector = EnergyVoiceActivityDetector(
        threshold=500.0,
    )

    audio = _pcm16(
        0,
        0,
        0,
        0,
        0,
        0,
    )

    assert detector.detect(
        audio,
        sample_rate_hz=16000,
    ) is False


def test_high_energy_audio_is_detected_as_speech():
    detector = EnergyVoiceActivityDetector(
        threshold=500.0,
    )

    audio = _pcm16(
        2000,
        -2000,
        1800,
        -1800,
        2200,
        -2200,
    )

    assert detector.detect(
        audio,
        sample_rate_hz=16000,
    ) is True


def test_threshold_is_exposed():
    detector = EnergyVoiceActivityDetector(
        threshold=750.0,
    )

    assert detector.threshold == 750.0


def test_detector_can_be_reset():
    detector = EnergyVoiceActivityDetector()

    assert detector.reset() is None


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
        "500",
    ],
)
def test_invalid_threshold_is_rejected(
    threshold,
):
    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        EnergyVoiceActivityDetector(
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "audio",
    [
        "audio",
        bytearray(b"audio"),
        None,
    ],
)
def test_invalid_audio_type_is_rejected(
    audio,
):
    detector = EnergyVoiceActivityDetector()

    with pytest.raises(
        VoiceActivityDetectionError,
        match="audio",
    ):
        detector.detect(
            audio,
            sample_rate_hz=16000,
        )


@pytest.mark.parametrize(
    "sample_rate_hz",
    [
        0,
        -1,
        16000.0,
        True,
    ],
)
def test_invalid_sample_rate_is_rejected(
    sample_rate_hz,
):
    detector = EnergyVoiceActivityDetector()

    with pytest.raises(
        VoiceActivityDetectionError,
        match="sample_rate_hz",
    ):
        detector.detect(
            b"\x00\x00",
            sample_rate_hz=sample_rate_hz,
        )


@pytest.mark.parametrize(
    "channels",
    [
        0,
        -1,
        1.0,
        True,
    ],
)
def test_invalid_channels_are_rejected(
    channels,
):
    detector = EnergyVoiceActivityDetector()

    with pytest.raises(
        VoiceActivityDetectionError,
        match="channels",
    ):
        detector.detect(
            b"\x00\x00",
            sample_rate_hz=16000,
            channels=channels,
        )


def test_odd_pcm16_payload_is_rejected():
    detector = EnergyVoiceActivityDetector()

    with pytest.raises(
        VoiceActivityDetectionError,
        match="even number of bytes",
    ):
        detector.detect(
            b"\x00",
            sample_rate_hz=16000,
        )


def test_empty_audio_is_not_speech():
    detector = EnergyVoiceActivityDetector()

    assert detector.detect(
        b"",
        sample_rate_hz=16000,
    ) is False


def test_detector_is_deterministic():
    detector = EnergyVoiceActivityDetector(
        threshold=1000.0,
    )

    audio = _pcm16(
        1000,
        -1000,
        1000,
        -1000,
    )

    first = detector.detect(
        audio,
        sample_rate_hz=16000,
    )
    second = detector.detect(
        audio,
        sample_rate_hz=16000,
    )

    assert first is True
    assert second is True
