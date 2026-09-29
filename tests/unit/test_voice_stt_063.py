from __future__ import annotations

import pytest

from osa.voice.contracts import (
    VoiceInput,
    VoiceTranscript,
)
from osa.voice.stt import (
    FixedSpeechToText,
    SpeechToText,
    SpeechToTextError,
)


def _voice_input() -> VoiceInput:
    return VoiceInput(
        audio=b"\x00\x00",
        sample_rate_hz=16000,
    )


def test_fixed_stt_satisfies_protocol_shape():
    transcript = VoiceTranscript(
        text="hello OSA",
        language="en",
    )

    stt = FixedSpeechToText(
        transcript,
    )

    assert isinstance(
        stt,
        SpeechToText,
    )


def test_fixed_stt_preserves_transcript():
    transcript = VoiceTranscript(
        text="hello OSA",
        language="en",
        confidence=0.95,
    )

    stt = FixedSpeechToText(
        transcript,
    )

    result = stt.transcribe(
        _voice_input()
    )

    assert result is transcript
    assert result.text == "hello OSA"
    assert result.language == "en"
    assert result.confidence == 0.95


def test_fixed_stt_requires_voice_input():
    stt = FixedSpeechToText(
        VoiceTranscript(
            text="hello",
        )
    )

    with pytest.raises(
        TypeError,
        match="VoiceInput",
    ):
        stt.transcribe(
            object()
        )


def test_fixed_stt_requires_valid_transcript():
    with pytest.raises(
        TypeError,
        match="VoiceTranscript",
    ):
        FixedSpeechToText(
            object()
        )


def test_fixed_stt_exposes_configured_transcript():
    transcript = VoiceTranscript(
        text="test",
    )

    stt = FixedSpeechToText(
        transcript,
    )

    assert stt.transcript is transcript


def test_speech_to_text_error_is_runtime_error():
    assert issubclass(
        SpeechToTextError,
        RuntimeError,
    )
