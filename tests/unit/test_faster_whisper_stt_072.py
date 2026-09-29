from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.voice.contracts import VoiceContractError, VoiceInput
from osa.voice.faster_whisper_stt import (
    FasterWhisperConfig,
    FasterWhisperSpeechToText,
    create_faster_whisper_stt,
)
from osa.voice.stt import SpeechToTextError


@dataclass
class FakeSegment:
    text: str


@dataclass
class FakeInfo:
    language: str


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(
            [
                FakeSegment(" Hello"),
                FakeSegment(" OSA"),
            ]
        ), FakeInfo("en")


def fake_audio_converter(audio: bytes) -> list[float]:
    return [sample / 32768.0 for sample in range(0, len(audio), 2)]


def make_audio() -> VoiceInput:
    return VoiceInput(
        audio=b"\x00\x00" * 1600,
        sample_rate_hz=16_000,
        channels=1,
        encoding="pcm_s16le",
    )


def make_test_stt(model) -> FasterWhisperSpeechToText:
    return FasterWhisperSpeechToText(
        model=model,
        audio_converter=fake_audio_converter,
    )


def test_faster_whisper_config_defaults() -> None:
    config = FasterWhisperConfig()

    assert config.model == "small"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.beam_size == 5
    assert config.vad_filter is True
    assert config.condition_on_previous_text is False


def test_faster_whisper_transcribes_using_injected_model() -> None:
    model = FakeWhisperModel()
    stt = FasterWhisperSpeechToText(
        FasterWhisperConfig(
            language="en",
            beam_size=3,
        ),
        model=model,
        audio_converter=fake_audio_converter,
    )

    result = stt.transcribe(make_audio())

    assert result.text == "Hello OSA"
    assert result.language == "en"
    assert result.confidence is None
    assert result.is_final is True
    assert len(model.calls) == 1

    samples, kwargs = model.calls[0]
    assert isinstance(samples, list)
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 3
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False


def test_faster_whisper_model_is_loaded_lazily() -> None:
    model = FakeWhisperModel()
    stt = create_faster_whisper_stt(
        model=model,
        audio_converter=fake_audio_converter,
    )

    assert stt.model_loaded is True
    stt.transcribe(make_audio())

    assert stt.model_loaded is True
    assert len(model.calls) == 1


def test_empty_voice_input_is_rejected_by_voice_contract() -> None:
    with pytest.raises(VoiceContractError, match="audio cannot be empty"):
        VoiceInput(
            audio=b"",
            sample_rate_hz=16_000,
            channels=1,
            encoding="pcm_s16le",
        )


def test_faster_whisper_rejects_wrong_encoding() -> None:
    stt = make_test_stt(FakeWhisperModel())

    with pytest.raises(SpeechToTextError, match="pcm_s16le"):
        stt.transcribe(
            VoiceInput(
                audio=b"\x00\x00",
                sample_rate_hz=16_000,
                channels=1,
                encoding="wav",
            )
        )


def test_faster_whisper_rejects_non_mono_audio() -> None:
    stt = make_test_stt(FakeWhisperModel())

    with pytest.raises(SpeechToTextError, match="mono"):
        stt.transcribe(
            VoiceInput(
                audio=b"\x00\x00" * 2,
                sample_rate_hz=16_000,
                channels=2,
                encoding="pcm_s16le",
            )
        )


def test_faster_whisper_rejects_wrong_sample_rate() -> None:
    stt = make_test_stt(FakeWhisperModel())

    with pytest.raises(SpeechToTextError, match="16000 Hz"):
        stt.transcribe(
            VoiceInput(
                audio=b"\x00\x00",
                sample_rate_hz=48_000,
                channels=1,
                encoding="pcm_s16le",
            )
        )


def test_faster_whisper_rejects_odd_pcm_byte_count() -> None:
    stt = make_test_stt(FakeWhisperModel())

    with pytest.raises(SpeechToTextError, match="even number of bytes"):
        stt.transcribe(
            VoiceInput(
                audio=b"\x00",
                sample_rate_hz=16_000,
                channels=1,
                encoding="pcm_s16le",
            )
        )


def test_faster_whisper_wraps_model_errors() -> None:
    class BrokenModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("boom")

    stt = make_test_stt(BrokenModel())

    with pytest.raises(SpeechToTextError, match="transcription failed"):
        stt.transcribe(make_audio())


def test_faster_whisper_accepts_protocol_usage() -> None:
    model = FakeWhisperModel()
    stt = make_test_stt(model)

    assert stt.transcribe(make_audio()).text == "Hello OSA"


def test_faster_whisper_config_rejects_invalid_beam_size() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        FasterWhisperConfig(beam_size=0)


def test_faster_whisper_wraps_converter_errors() -> None:
    def broken_converter(audio: bytes):
        raise RuntimeError("converter failed")

    stt = FasterWhisperSpeechToText(
        model=FakeWhisperModel(),
        audio_converter=broken_converter,
    )

    with pytest.raises(
        SpeechToTextError,
        match="transcription failed",
    ):
        stt.transcribe(make_audio())
