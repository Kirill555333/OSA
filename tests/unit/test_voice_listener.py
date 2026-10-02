"""Targeted unit tests for OSAVoiceListener."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np

from osa.voice.listener import OSAVoiceListener


def test_listener_initialization() -> None:
    listener = OSAVoiceListener(model_size="medium", sample_rate=16000)
    assert listener.model_size == "medium"
    assert listener.sample_rate == 16000
    assert listener._model is None


def test_transcribe_empty_audio() -> None:
    listener = OSAVoiceListener()
    empty = np.array([], dtype=np.float32)
    assert listener.transcribe(empty) == ""


def test_transcribe_mocked_audio() -> None:
    listener = OSAVoiceListener()
    fake_segment = MagicMock()
    fake_segment.text = "Привет OSA"
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([fake_segment], MagicMock())
    listener._model = fake_model

    dummy_audio = np.ones(16000, dtype=np.float32)
    result = listener.transcribe(dummy_audio)
    assert result == "Привет OSA"
    fake_model.transcribe.assert_called_once()
