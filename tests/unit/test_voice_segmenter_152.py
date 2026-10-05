"""Tests for adaptive VAD and utterance segmentation (OSA 1.5.2.1)."""

from __future__ import annotations

import struct

import pytest

from osa.voice.segmenter import (
    AdaptiveEnergyVad,
    SegmenterConfig,
    UtteranceSegmenter,
    pcm16_rms,
)

CONFIG = SegmenterConfig()
SILENCE = struct.pack("<480h", *([0] * 480))


def tone(amplitude: int) -> bytes:
    """One 30 ms frame of a square wave whose RMS equals the amplitude."""
    return struct.pack("<480h", *([amplitude, -amplitude] * 240))


class _ThresholdVad:
    """Deterministic stand-in for tests that only exercise segmentation timing."""

    def __init__(self, threshold: float = 100.0) -> None:
        self._threshold = threshold

    def detect(self, audio: bytes, *, sample_rate_hz: int, channels: int = 1) -> bool:
        return pcm16_rms(audio) >= self._threshold

    def reset(self) -> None:
        return None


def _segmenter(config: SegmenterConfig | None = None) -> UtteranceSegmenter:
    return UtteranceSegmenter(_ThresholdVad(), config)


def _collect(segmenter: UtteranceSegmenter, frames: list[bytes]) -> list[bytes]:
    utterances = []

    for frame in frames:
        result = segmenter.feed(frame)

        if result is not None:
            utterances.append(result)

    return utterances


# --- pcm16_rms ---------------------------------------------------------------------


def test_rms_of_silence_and_tone() -> None:
    assert pcm16_rms(b"") == 0.0
    assert pcm16_rms(SILENCE) == 0.0
    assert pcm16_rms(tone(1000)) == pytest.approx(1000.0)


def test_rms_rejects_odd_length() -> None:
    with pytest.raises(ValueError):
        pcm16_rms(b"\x00\x01\x02")


# --- adaptive VAD ------------------------------------------------------------------


def test_vad_calibrates_on_ambient_noise_then_detects_speech() -> None:
    vad = AdaptiveEnergyVad()

    for _ in range(10):
        assert vad.detect(tone(80), sample_rate_hz=16000) is False

    assert vad.detect(tone(80), sample_rate_hz=16000) is False
    assert vad.detect(tone(2000), sample_rate_hz=16000) is True


def test_vad_threshold_follows_a_noisier_room() -> None:
    quiet = AdaptiveEnergyVad()
    noisy = AdaptiveEnergyVad()

    for _ in range(11):
        quiet.detect(tone(50), sample_rate_hz=16000)
        noisy.detect(tone(400), sample_rate_hz=16000)

    assert noisy.threshold > quiet.threshold
    assert noisy.detect(tone(600), sample_rate_hz=16000) is False
    assert noisy.detect(tone(3000), sample_rate_hz=16000) is True


def test_vad_adopts_constant_loud_sound_as_new_noise_floor() -> None:
    vad = AdaptiveEnergyVad(stuck_frames=5)

    for _ in range(11):
        vad.detect(tone(50), sample_rate_hz=16000)

    results = [vad.detect(tone(1500), sample_rate_hz=16000) for _ in range(12)]

    assert results[0] is True
    assert results[-1] is False


def test_vad_reset_restarts_calibration() -> None:
    vad = AdaptiveEnergyVad()

    for _ in range(11):
        vad.detect(tone(50), sample_rate_hz=16000)

    vad.reset()

    assert vad.detect(tone(5000), sample_rate_hz=16000) is False


def test_vad_validates_arguments() -> None:
    with pytest.raises(ValueError):
        AdaptiveEnergyVad(ratio=1.0)

    with pytest.raises(ValueError):
        AdaptiveEnergyVad().detect(SILENCE, sample_rate_hz=0)


# --- config ------------------------------------------------------------------------


def test_config_frame_math() -> None:
    assert CONFIG.frame_samples == 480
    assert CONFIG.frame_bytes == 960
    assert CONFIG.frames(90) == 3
    assert CONFIG.frames(100) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_ms": 0},
        {"sample_rate_hz": 11025, "frame_ms": 7},
        {"tail_ms": 2000},
        {"min_speech_ms": 20000},
        {"preroll_ms": True},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SegmenterConfig(**kwargs)


# --- segmentation ------------------------------------------------------------------


def test_silence_produces_nothing() -> None:
    assert _collect(_segmenter(), [SILENCE] * 200) == []


def test_utterance_is_returned_after_trailing_silence() -> None:
    speech = [tone(1000)] * 20
    frames = [SILENCE] * 10 + speech + [SILENCE] * 40

    utterances = _collect(_segmenter(), frames)

    assert len(utterances) == 1
    assert len(utterances[0]) % CONFIG.frame_bytes == 0


def test_utterance_includes_preroll_speech_and_short_tail_only() -> None:
    frames = [SILENCE] * 20 + [tone(1000)] * 20 + [SILENCE] * 40

    audio = _collect(_segmenter(), frames)[0]
    frame_count = len(audio) // CONFIG.frame_bytes

    preroll = CONFIG.frames(CONFIG.preroll_ms)
    tail = CONFIG.frames(CONFIG.tail_ms)

    assert frame_count == preroll + 20 + tail


def test_short_blip_is_discarded() -> None:
    frames = [SILENCE] * 10 + [tone(1000)] * 5 + [SILENCE] * 60

    assert _collect(_segmenter(), frames) == []


def test_pause_inside_speech_does_not_split_the_utterance() -> None:
    pause = [SILENCE] * 10
    frames = [SILENCE] * 5 + [tone(1000)] * 12 + pause + [tone(1000)] * 12 + [SILENCE] * 40

    utterances = _collect(_segmenter(), frames)

    assert len(utterances) == 1


def test_long_pause_splits_into_two_utterances() -> None:
    frames = (
        [SILENCE] * 5
        + [tone(1000)] * 20
        + [SILENCE] * 40
        + [tone(1000)] * 20
        + [SILENCE] * 40
    )

    assert len(_collect(_segmenter(), frames)) == 2


def test_isolated_loud_frames_do_not_start_an_utterance() -> None:
    frames = [SILENCE] * 5
    for _ in range(10):
        frames += [tone(1000), tone(1000), SILENCE]
    frames += [SILENCE] * 50

    assert _collect(_segmenter(), frames) == []


def test_max_length_forces_an_utterance_to_end() -> None:
    config = SegmenterConfig(max_utterance_ms=1500, min_speech_ms=300)
    frames = [SILENCE] * 5 + [tone(1000)] * 120

    utterances = _collect(_segmenter(config), frames)

    assert len(utterances) >= 1
    assert all(len(item) // config.frame_bytes <= config.frames(1500) for item in utterances)


def test_flush_returns_pending_utterance() -> None:
    segmenter = _segmenter()

    _collect(segmenter, [SILENCE] * 5 + [tone(1000)] * 20)

    assert segmenter.active is True
    assert segmenter.flush() is not None
    assert segmenter.active is False
    assert segmenter.flush() is None


def test_flush_discards_too_short_pending_audio() -> None:
    segmenter = _segmenter()

    _collect(segmenter, [SILENCE] * 5 + [tone(1000)] * 4)

    assert segmenter.flush() is None


def test_reset_clears_partial_state() -> None:
    segmenter = _segmenter()

    _collect(segmenter, [tone(1000)] * 10)
    segmenter.reset()

    assert segmenter.active is False
    assert segmenter.flush() is None


def test_wrong_frame_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        _segmenter().feed(b"\x00" * 100)


def test_end_to_end_with_adaptive_vad() -> None:
    segmenter = UtteranceSegmenter(AdaptiveEnergyVad())
    frames = [tone(60)] * 20 + [tone(2500)] * 25 + [tone(60)] * 40

    utterances = _collect(segmenter, frames)

    assert len(utterances) == 1
