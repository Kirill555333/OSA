from __future__ import annotations

import pytest

from osa.voice.contracts import VoiceOutput
from osa.voice.tts import (
    FixedTextToSpeech,
    TextToSpeech,
    TextToSpeechError,
)


def _output() -> VoiceOutput:
    return VoiceOutput(
        text="hello OSA",
        language="en",
    )


def test_fixed_tts_satisfies_protocol_shape():
    tts = FixedTextToSpeech()

    assert isinstance(
        tts,
        TextToSpeech,
    )


def test_fixed_tts_returns_configured_audio():
    tts = FixedTextToSpeech(
        audio=b"test-audio",
    )

    result = tts.synthesize(
        _output()
    )

    assert result == b"test-audio"


def test_fixed_tts_preserves_binary_payload():
    payload = bytes(
        range(16)
    )

    tts = FixedTextToSpeech(
        audio=payload,
    )

    assert tts.synthesize(
        _output()
    ) is payload


def test_fixed_tts_requires_voice_output():
    tts = FixedTextToSpeech()

    with pytest.raises(
        TypeError,
        match="VoiceOutput",
    ):
        tts.synthesize(
            object()
        )


@pytest.mark.parametrize(
    "audio",
    [
        b"",
        "audio",
        bytearray(b"audio"),
        None,
    ],
)
def test_fixed_tts_rejects_invalid_audio(
    audio,
):
    with pytest.raises(
        (TypeError, ValueError),
        match="audio",
    ):
        FixedTextToSpeech(
            audio=audio,
        )


def test_fixed_tts_exposes_audio():
    payload = b"configured"

    tts = FixedTextToSpeech(
        audio=payload,
    )

    assert tts.audio == payload


def test_stop_records_interrupt_request():
    tts = FixedTextToSpeech()

    assert tts.stop_calls == 0

    tts.stop()
    tts.stop()

    assert tts.stop_calls == 2


def test_text_to_speech_error_is_runtime_error():
    assert issubclass(
        TextToSpeechError,
        RuntimeError,
    )
