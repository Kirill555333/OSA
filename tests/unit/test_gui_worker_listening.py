"""Tests for GUI AgentWorker background listening (1.5.3.0)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from osa.gui.worker import AgentWorker  # noqa: E402
from osa.shell.input_hub import InputEvent  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_worker(agent=None):
    return AgentWorker(agent=agent or MagicMock())


def test_build_listener_creates_components(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    worker = _make_worker()
    with patch("osa.gui.worker.BackgroundListener") as mock_bl, \
         patch("osa.gui.worker.FasterWhisperTranscriber") as mock_tr, \
         patch("osa.gui.worker.SoundDeviceFrameSource"), \
         patch("osa.gui.worker.WakeWordGate"), \
         patch("osa.gui.worker.InputHub") as mock_hub:
        listener, hub = worker._build_listener("small")

    mock_tr.assert_called_once_with("small")
    mock_tr.return_value.preload.assert_called_once()
    mock_bl.assert_called_once()
    assert listener is mock_bl.return_value
    assert hub is mock_hub.return_value


def test_start_listening_bg_success(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    worker = _make_worker()
    mock_listener = MagicMock()
    mock_hub = MagicMock()
    worker._build_listener = MagicMock(return_value=(mock_listener, mock_hub))
    worker._starting = True

    worker._start_listening_bg("small")

    assert worker._listening is True
    assert worker._starting is False
    assert worker._background_listener is mock_listener
    assert worker._input_hub is mock_hub
    mock_listener.start.assert_called_once()


def test_start_listening_bg_failure_cleans_up(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    worker = _make_worker()
    worker._build_listener = MagicMock(side_effect=RuntimeError("no mic"))
    worker._starting = True

    worker._start_listening_bg("small")

    assert worker._listening is False
    assert worker._starting is False
    assert worker._background_listener is None
    assert worker._input_hub is None


def test_start_listening_bg_aborts_if_cancelled_during_preload(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OSA_HOME", str(tmp_path))
    worker = _make_worker()
    mock_listener = MagicMock()
    worker._build_listener = MagicMock(return_value=(mock_listener, MagicMock()))
    worker._starting = False

    worker._start_listening_bg("small")

    mock_listener.start.assert_not_called()
    assert worker._background_listener is None


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
