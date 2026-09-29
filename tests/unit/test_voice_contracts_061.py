from __future__ import annotations

import pytest

from osa.voice.contracts import (
    VoiceContractError,
    VoiceInput,
    VoiceOutput,
    VoiceSessionConfig,
    VoiceSessionState,
    VoiceTranscript,
)


def test_voice_session_states_are_stable():
    assert VoiceSessionState.IDLE.value == "idle"
    assert VoiceSessionState.LISTENING.value == "listening"
    assert VoiceSessionState.PROCESSING.value == "processing"
    assert VoiceSessionState.SPEAKING.value == "speaking"
    assert VoiceSessionState.STOPPING.value == "stopping"
    assert VoiceSessionState.STOPPED.value == "stopped"
    assert VoiceSessionState.ERROR.value == "error"


def test_voice_input_is_immutable():
    value = VoiceInput(
        audio=b"audio",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        AttributeError,
    ):
        value.sample_rate_hz = 8000


def test_voice_input_normalizes_encoding():
    value = VoiceInput(
        audio=b"audio",
        sample_rate_hz=16000,
        encoding=" PCM_S16LE ",
    )

    assert value.encoding == "pcm_s16le"


@pytest.mark.parametrize(
    ("audio", "sample_rate_hz", "message"),
    [
        (b"", 16000, "audio"),
        ("audio", 16000, "audio"),
        (b"audio", 0, "sample_rate_hz"),
        (b"audio", -1, "sample_rate_hz"),
    ],
)
def test_voice_input_rejects_invalid_values(
    audio,
    sample_rate_hz,
    message: str,
):
    with pytest.raises(
        VoiceContractError,
        match=message,
    ):
        VoiceInput(
            audio=audio,
            sample_rate_hz=sample_rate_hz,
        )


def test_voice_input_rejects_invalid_channels():
    with pytest.raises(
        VoiceContractError,
        match="channels",
    ):
        VoiceInput(
            audio=b"audio",
            sample_rate_hz=16000,
            channels=0,
        )


def test_voice_transcript_normalizes_text_and_language():
    value = VoiceTranscript(
        text="  Hello OSA  ",
        language=" EN-US ",
        confidence=0.8,
    )

    assert value.text == "Hello OSA"
    assert value.language == "en-us"
    assert value.confidence == 0.8


def test_voice_transcript_confidence_defaults_to_none():
    value = VoiceTranscript(
        text="Hello",
    )

    assert value.confidence is None
    assert value.is_final is True


@pytest.mark.parametrize(
    "confidence",
    [
        -0.1,
        1.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_voice_transcript_rejects_invalid_confidence(
    confidence: float,
):
    with pytest.raises(
        VoiceContractError,
        match="confidence",
    ):
        VoiceTranscript(
            text="Hello",
            confidence=confidence,
        )


def test_voice_transcript_rejects_empty_text():
    with pytest.raises(
        VoiceContractError,
        match="text",
    ):
        VoiceTranscript(
            text="   ",
        )


def test_voice_transcript_is_immutable():
    value = VoiceTranscript(
        text="Hello",
    )

    with pytest.raises(
        AttributeError,
    ):
        value.text = "Changed"


def test_voice_output_normalizes_text_and_language():
    value = VoiceOutput(
        text="  Hello there  ",
        language=" EN ",
    )

    assert value.text == "Hello there"
    assert value.language == "en"


def test_voice_output_rejects_empty_text():
    with pytest.raises(
        VoiceContractError,
        match="text",
    ):
        VoiceOutput(
            text=" ",
        )


def test_voice_output_is_immutable():
    value = VoiceOutput(
        text="Hello",
    )

    with pytest.raises(
        AttributeError,
    ):
        value.text = "Changed"


def test_voice_session_config_defaults():
    value = VoiceSessionConfig()

    assert value.language == "auto"
    assert value.max_utterance_seconds == 30.0
    assert value.interrupt_enabled is True


@pytest.mark.parametrize(
    ("max_utterance_seconds", "message"),
    [
        (0, "greater than zero"),
        (-1, "greater than zero"),
        (float("inf"), "finite"),
        (float("nan"), "finite"),
    ],
)
def test_voice_session_config_rejects_invalid_duration(
    max_utterance_seconds: float,
    message: str,
):
    with pytest.raises(
        VoiceContractError,
        match=message,
    ):
        VoiceSessionConfig(
            max_utterance_seconds=max_utterance_seconds,
        )


def test_voice_session_config_normalizes_language():
    value = VoiceSessionConfig(
        language=" EN-US ",
    )

    assert value.language == "en-us"


def test_voice_session_config_rejects_invalid_interrupt_flag():
    with pytest.raises(
        VoiceContractError,
        match="interrupt_enabled",
    ):
        VoiceSessionConfig(
            interrupt_enabled=1,
        )


def test_voice_session_config_is_immutable():
    value = VoiceSessionConfig()

    with pytest.raises(
        AttributeError,
    ):
        value.language = "ru"


def test_voice_contract_module_exports_expected_types():
    assert VoiceContractError
    assert VoiceInput
    assert VoiceTranscript
    assert VoiceOutput
    assert VoiceSessionState
    assert VoiceSessionConfig
