from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest

from osa.voice.contracts import VoiceOutput
from osa.voice.piper_tts import (
    PiperTTSConfig,
    PiperTextToSpeech,
    create_piper_tts,
)
from osa.voice.tts import TextToSpeechError


@dataclass
class FakeAudioChunk:
    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_int16_bytes: bytes


@dataclass
class FakeSynthesisConfig:
    speaker_id: int | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None


class FakePiperVoice:
    def __init__(
        self,
        chunks: list[FakeAudioChunk],
        *,
        on_first_chunk=None,
    ) -> None:
        self.chunks = chunks
        self.on_first_chunk = on_first_chunk
        self.calls: list[tuple[str, object]] = []

    def synthesize(self, text: str, syn_config=None):
        self.calls.append((text, syn_config))

        for index, chunk in enumerate(self.chunks):
            yield chunk

            if index == 0 and self.on_first_chunk is not None:
                self.on_first_chunk()


def fake_synthesis_config_factory(**kwargs) -> FakeSynthesisConfig:
    return FakeSynthesisConfig(**kwargs)


def make_voice() -> FakePiperVoice:
    return FakePiperVoice(
        [
            FakeAudioChunk(
                sample_rate=22_050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=struct.pack("<hh", 100, -100),
            ),
            FakeAudioChunk(
                sample_rate=22_050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=struct.pack("<hh", 200, -200),
            ),
        ]
    )


def test_piper_config_defaults() -> None:
    config = PiperTTSConfig()

    assert config.model_path is None
    assert config.use_cuda is False
    assert config.length_scale is None
    assert config.noise_scale is None
    assert config.noise_w_scale is None
    assert config.speaker_id is None


def test_piper_synthesizes_wav_bytes_from_injected_voice() -> None:
    voice = make_voice()
    tts = create_piper_tts(
        voice=voice,
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    result = tts.synthesize(VoiceOutput(text="Hello OSA"))

    assert result.startswith(b"RIFF")
    assert b"WAVE" in result
    assert len(result) > 44
    assert voice.calls[0][0] == "Hello OSA"


def test_piper_passes_synthesis_configuration() -> None:
    voice = make_voice()
    tts = PiperTextToSpeech(
        PiperTTSConfig(
            length_scale=0.9,
            noise_scale=0.4,
            noise_w_scale=0.7,
            speaker_id=2,
        ),
        voice=voice,
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    tts.synthesize(VoiceOutput(text="Hello"))

    _, syn_config = voice.calls[0]

    assert syn_config.length_scale == 0.9
    assert syn_config.noise_scale == 0.4
    assert syn_config.noise_w_scale == 0.7
    assert syn_config.speaker_id == 2


def test_piper_model_is_loaded_lazily_when_injected_voice_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"fake-model")

    tts = PiperTextToSpeech(
        PiperTTSConfig(model_path=str(model_path)),
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    fake_voice = make_voice()

    class FakePiperVoiceLoader:
        @staticmethod
        def load(path, use_cuda=False):
            assert path == model_path
            assert use_cuda is False
            return fake_voice

    class FakePiperModule:
        SynthesisConfig = FakeSynthesisConfig
        PiperVoice = FakePiperVoiceLoader

    monkeypatch.setitem(
        __import__("sys").modules,
        "piper",
        FakePiperModule,
    )

    assert tts.model_loaded is False

    result = tts.synthesize(VoiceOutput(text="Hello"))

    assert result.startswith(b"RIFF")
    assert tts.model_loaded is True


def test_piper_rejects_missing_model_path() -> None:
    tts = create_piper_tts()

    with pytest.raises(
        TextToSpeechError,
        match="model_path is required",
    ):
        tts.synthesize(VoiceOutput(text="Hello"))


def test_piper_rejects_missing_model_file() -> None:
    tts = PiperTextToSpeech(
        PiperTTSConfig(model_path="/does/not/exist/voice.onnx")
    )

    with pytest.raises(
        TextToSpeechError,
        match="model was not found",
    ):
        tts.synthesize(VoiceOutput(text="Hello"))


def test_piper_rejects_wrong_output_type() -> None:
    tts = create_piper_tts(
        voice=make_voice(),
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    with pytest.raises(
        TextToSpeechError,
        match="VoiceOutput",
    ):
        tts.synthesize(object())


def test_piper_stop_interrupts_synthesis() -> None:
    tts: PiperTextToSpeech

    voice = FakePiperVoice(
        [
            FakeAudioChunk(
                sample_rate=22_050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=struct.pack("<h", 100),
            ),
            FakeAudioChunk(
                sample_rate=22_050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=struct.pack("<h", 200),
            ),
        ],
    )

    original_synthesize = voice.synthesize

    def controlled_synthesize(text, syn_config=None):
        generator = original_synthesize(text, syn_config)

        first = next(generator)
        yield first
        tts.stop()

        yield from generator

    voice.synthesize = controlled_synthesize
    tts = create_piper_tts(
        voice=voice,
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    with pytest.raises(
        TextToSpeechError,
        match="was stopped",
    ):
        tts.synthesize(VoiceOutput(text="Hello"))


def test_piper_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="length_scale"):
        PiperTTSConfig(length_scale=0)

    with pytest.raises(ValueError, match="noise_scale"):
        PiperTTSConfig(noise_scale=-1)

    with pytest.raises(ValueError, match="noise_w_scale"):
        PiperTTSConfig(noise_w_scale=-1)

    with pytest.raises(ValueError, match="speaker_id"):
        PiperTTSConfig(speaker_id=-1)


def test_piper_empty_synthesis_is_reported() -> None:
    tts = create_piper_tts(
        voice=FakePiperVoice([]),
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    with pytest.raises(
        TextToSpeechError,
        match="no audio",
    ):
        tts.synthesize(VoiceOutput(text="Hello"))


def test_piper_wraps_voice_errors() -> None:
    class BrokenVoice:
        def synthesize(self, text, syn_config=None):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    tts = create_piper_tts(
        voice=BrokenVoice(),
        synthesis_config_factory=fake_synthesis_config_factory,
    )

    with pytest.raises(
        TextToSpeechError,
        match="Piper synthesis failed",
    ):
        tts.synthesize(VoiceOutput(text="Hello"))
