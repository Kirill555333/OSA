"""Tests for the background listening hardware adapters (OSA 1.5.2.2)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from osa.voice.background_audio import (
    FasterWhisperTranscriber,
    SoundDeviceFrameSource,
    keep_segment,
)

FRAME = b"\x01\x00" * 480


class _FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.aborted = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


class _BrokenStream(_FakeStream):
    def abort(self) -> None:
        raise RuntimeError("driver hung up")

    def close(self) -> None:
        raise RuntimeError("driver hung up")


def _install_fake_sounddevice(stream_class: type) -> tuple[list, object]:
    created: list = []

    def raw_input_stream(**kwargs):
        stream = stream_class(**kwargs)
        created.append(stream)
        return stream

    previous = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = SimpleNamespace(RawInputStream=raw_input_stream)

    return created, previous


def _restore_sounddevice(previous: object) -> None:
    if previous is None:
        sys.modules.pop("sounddevice", None)
    else:
        sys.modules["sounddevice"] = previous


# --- frame source ------------------------------------------------------------------


def test_frame_geometry() -> None:
    assert SoundDeviceFrameSource().frame_bytes == 960
    assert SoundDeviceFrameSource(sample_rate_hz=8000, frame_ms=20).frame_bytes == 320


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate_hz": 0},
        {"frame_ms": 0},
        {"max_queue_frames": 0},
        {"sample_rate_hz": 11025, "frame_ms": 7},
    ],
)
def test_frame_source_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SoundDeviceFrameSource(**kwargs)


def test_callback_frames_are_read_in_order() -> None:
    source = SoundDeviceFrameSource()

    source._callback(FRAME, 480, None, None)
    source._callback(b"\x02\x00" * 480, 480, None, None)

    assert source.read(0.01) == FRAME
    assert source.read(0.01) == b"\x02\x00" * 480
    assert source.read(0.01) is None


def test_full_queue_drops_the_oldest_frame() -> None:
    source = SoundDeviceFrameSource(max_queue_frames=2)

    for value in (1, 2, 3):
        source._callback(bytes([value, 0]) * 480, 480, None, None)

    assert source.read(0.01)[0] == 2
    assert source.read(0.01)[0] == 3
    assert source.read(0.01) is None


def test_drain_discards_everything_captured() -> None:
    source = SoundDeviceFrameSource()

    for _ in range(5):
        source._callback(FRAME, 480, None, None)

    source.drain()

    assert source.read(0.01) is None


def test_start_opens_one_int16_mono_stream_with_exact_blocksize() -> None:
    created, previous = _install_fake_sounddevice(_FakeStream)

    try:
        source = SoundDeviceFrameSource(device=3)
        source.start()
        source.start()
    finally:
        _restore_sounddevice(previous)

    assert len(created) == 1
    stream = created[0]
    assert stream.started is True
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["blocksize"] == 480
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["device"] == 3
    assert stream.kwargs["callback"] == source._callback


def test_stop_aborts_and_closes_the_stream() -> None:
    created, previous = _install_fake_sounddevice(_FakeStream)

    try:
        source = SoundDeviceFrameSource()
        source.start()
        source.stop()
    finally:
        _restore_sounddevice(previous)

    assert created[0].aborted is True
    assert created[0].closed is True


def test_stop_survives_a_failing_driver_and_is_repeatable() -> None:
    created, previous = _install_fake_sounddevice(_BrokenStream)

    try:
        source = SoundDeviceFrameSource()
        source.start()
        source.stop()
        source.stop()
    finally:
        _restore_sounddevice(previous)

    assert len(created) == 1


def test_stop_without_start_is_a_no_op() -> None:
    SoundDeviceFrameSource().stop()


# --- segment filtering -------------------------------------------------------------


def test_keep_segment_filters_silence_and_hallucinations() -> None:
    assert keep_segment(0.1, -0.3) is True
    assert keep_segment(None, None) is True
    assert keep_segment(0.9, -0.3) is False
    assert keep_segment(0.1, -1.6) is False
    assert keep_segment(0.6, -1.0) is True


# --- transcriber -------------------------------------------------------------------


class _FakeModel:
    def __init__(self, segments) -> None:
        self._segments = segments
        self.calls: list = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        return iter(self._segments), None


def _segment(text: str, no_speech_prob: float = 0.1, avg_logprob: float = -0.3):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


def test_transcriber_joins_trusted_segments_only() -> None:
    model = _FakeModel(
        [
            _segment(" Оса, "),
            _segment("Продолжение следует", no_speech_prob=0.95),
            _segment(" открой хром. "),
            _segment("Субтитры", avg_logprob=-2.0),
        ]
    )
    transcriber = FasterWhisperTranscriber(model=model)

    assert transcriber.transcribe(FRAME * 10, 16000) == "Оса, открой хром."


def test_transcriber_converts_pcm16_to_normalized_float32() -> None:
    model = _FakeModel([_segment("привет")])
    loud = np.array([32767, -32768, 0, 16384], dtype="<i2").tobytes()

    FasterWhisperTranscriber(model=model, language="ru", beam_size=2).transcribe(loud, 16000)

    samples, kwargs = model.calls[0]
    assert samples.dtype == np.float32
    assert float(np.max(np.abs(samples))) <= 1.0
    assert samples[3] == pytest.approx(0.5)
    assert kwargs["language"] == "ru"
    assert kwargs["beam_size"] == 2
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is False


def test_transcriber_returns_empty_text_for_empty_audio_without_touching_the_model() -> None:
    model = _FakeModel([_segment("не должно вызываться")])

    assert FasterWhisperTranscriber(model=model).transcribe(b"", 16000) == ""
    assert model.calls == []


def test_transcriber_loads_the_model_lazily_and_once() -> None:
    loads: list[str] = []
    model = _FakeModel([_segment("привет")])

    def factory(size: str):
        loads.append(size)
        return model

    transcriber = FasterWhisperTranscriber("tiny", model_factory=factory)

    assert loads == []

    transcriber.preload()
    transcriber.transcribe(FRAME, 16000)
    transcriber.transcribe(FRAME, 16000)

    assert loads == ["tiny"]
    assert transcriber.model_size == "tiny"


def test_transcriber_rejects_invalid_beam_size() -> None:
    with pytest.raises(ValueError):
        FasterWhisperTranscriber(beam_size=0)
