from __future__ import annotations

import pytest

from osa.voice.activation import (
    AlwaysActiveVoiceActivationDetector,
    KeywordVoiceActivationDetector,
    VoiceActivationResult,
    create_keyword_activation_detector,
)
from osa.voice.contracts import VoiceTranscript


def transcript(text: str) -> VoiceTranscript:
    return VoiceTranscript(
        text=text,
        language="ru",
    )


def test_always_active_preserves_existing_behavior() -> None:
    detector = AlwaysActiveVoiceActivationDetector()

    result = detector.detect(
        transcript("open browser")
    )

    assert result == VoiceActivationResult(
        activated=True,
        command_text="open browser",
    )


def test_keyword_activation_matches_phrase() -> None:
    detector = KeywordVoiceActivationDetector(
        ("OSA",)
    )

    result = detector.detect(
        transcript("OSA open browser")
    )

    assert result.activated is True
    assert result.command_text == "open browser"


def test_keyword_activation_is_case_insensitive() -> None:
    detector = create_keyword_activation_detector(
        "Hey OSA"
    )

    result = detector.detect(
        transcript("hey osa, open browser")
    )

    assert result.activated is True
    assert result.command_text == "open browser"


def test_keyword_activation_rejects_embedded_word() -> None:
    detector = create_keyword_activation_detector(
        "osa"
    )

    result = detector.detect(
        transcript("osak open browser")
    )

    assert result.activated is False
    assert result.command_text == ""


def test_keyword_activation_accepts_phrase_after_text() -> None:
    detector = create_keyword_activation_detector(
        "OSA"
    )

    result = detector.detect(
        transcript("hello OSA, open browser")
    )

    assert result.activated is True
    assert result.command_text == "hello, open browser"


def test_keyword_activation_can_preserve_phrase() -> None:
    detector = create_keyword_activation_detector(
        "OSA",
        strip_phrase=False,
    )

    result = detector.detect(
        transcript("OSA open browser")
    )

    assert result.activated is True
    assert result.command_text == "OSA open browser"


def test_keyword_activation_rejects_silent_transcript() -> None:
    detector = create_keyword_activation_detector(
        "OSA"
    )

    result = detector.detect(
        transcript("something else")
    )

    assert result.activated is False
    assert result.command_text == ""


def test_activation_detector_rejects_invalid_transcript() -> None:
    detector = create_keyword_activation_detector(
        "OSA"
    )

    with pytest.raises(
        TypeError,
        match="VoiceTranscript",
    ):
        detector.detect(
            object()
        )


def test_activation_requires_at_least_one_phrase() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        KeywordVoiceActivationDetector(())


def test_activation_result_normalizes_command_text() -> None:
    result = VoiceActivationResult(
        activated=True,
        command_text="  open   browser  ",
    )

    assert result.command_text == "open browser"
