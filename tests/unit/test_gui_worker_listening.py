"""Tests for GUI AgentWorker background listening (1.5.3.0)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from osa.gui.worker import AgentWorker
from osa.shell.input_hub import InputEvent


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_worker(agent=None):
    return AgentWorker(agent=agent or MagicMock())


def test_start_listening_creates_and_starts_listener(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    worker = _make_worker()

    with patch("osa.gui.worker.BackgroundListener") as mock_bl, \
         patch("osa.gui.worker.FasterWhisperTranscriber"), \
         patch("osa.gui.worker.SoundDeviceFrameSource"), \
         patch("osa.gui.worker.WakeWordGate"), \
         patch("osa.gui.worker.InputHub"):
        assert worker.start_listening() is True

    assert worker.is_listening is True
    mock_bl.assert_called_once()
    mock_bl.return_value.start.assert_called_once()


def test_stop_listening_stops_listener(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    worker = _make_worker()

    with patch("osa.gui.worker.BackgroundListener") as mock_bl, \
         patch("osa.gui.worker.FasterWhisperTranscriber"), \
         patch("osa.gui.worker.SoundDeviceFrameSource"), \
         patch("osa.gui.worker.WakeWordGate"), \
         patch("osa.gui.worker.InputHub"):
        worker.start_listening()
        worker.stop_listening()

    assert worker.is_listening is False
    mock_bl.return_value.stop.assert_called_once()


def test_poll_voice_events_emits_text(qapp):
    worker = _make_worker()
    received = []
    worker.voice_text_received.connect(received.append)

    hub = MagicMock()
    hub.next_event.side_effect = [
        InputEvent("voice", "привет", "wake"),
        None,
    ]
    worker._input_hub = hub
    worker._poll_voice_events()

    assert received == ["привет"]


def test_poll_voice_events_ignores_empty_text(qapp):
    worker = _make_worker()
    received = []
    worker.voice_text_received.connect(received.append)

    hub = MagicMock()
    hub.next_event.side_effect = [
        InputEvent("voice", "   ", "wake_only"),
        None,
    ]
    worker._input_hub = hub
    worker._poll_voice_events()

    assert received == []


def test_execute_text_query_mutes_listener_while_speaking(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    agent = MagicMock()
    agent.chat.return_value = MagicMock(content="ответ")
    worker = _make_worker(agent=agent)
    worker._voice_engine = MagicMock()
    worker._voice_engine.is_speaking.return_value = False

    listener = MagicMock()
    worker._background_listener = listener
    worker._listening = True

    worker.execute_text_query("привет")

    listener.mute.assert_called_once()
    listener.unmute.assert_called_once()
    listener.open_follow_up.assert_called_once()
