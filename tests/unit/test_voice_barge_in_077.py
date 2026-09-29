from __future__ import annotations

import threading
import time

import pytest

from osa.voice.barge_in import (
    BargeInConfig,
    BargeInState,
    VoiceBargeInMonitor,
)
from osa.voice.contracts import VoiceInput


class QueueMicrophone:
    def __init__(
        self,
        inputs: list[VoiceInput],
    ) -> None:
        self._inputs = inputs
        self._index = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        return None

    def read(self) -> VoiceInput:
        with self._lock:
            if self._index >= len(self._inputs):
                time.sleep(0.01)
                return self._inputs[-1]

            result = self._inputs[self._index]
            self._index += 1
            return result

    def stop(self) -> None:
        return None


class FakeVAD:
    def __init__(
        self,
        speech: bool,
    ) -> None:
        self.speech = speech

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        return self.speech

    def reset(self) -> None:
        return None


def make_input() -> VoiceInput:
    return VoiceInput(
        audio=b"\x01\x00" * 100,
        sample_rate_hz=16_000,
        channels=1,
    )


def wait_for(
    predicate,
    timeout: float = 1.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return True

        time.sleep(0.005)

    return predicate()


def test_barge_in_monitor_starts_stopped() -> None:
    monitor = VoiceBargeInMonitor(
        QueueMicrophone([make_input()]),
        FakeVAD(False),
        lambda: None,
    )

    assert monitor.state == BargeInState.STOPPED
    assert monitor.running is False


def test_barge_in_monitor_calls_callback_on_speech() -> None:
    triggered = threading.Event()

    monitor = VoiceBargeInMonitor(
        QueueMicrophone([make_input()]),
        FakeVAD(True),
        triggered.set,
    )

    monitor.start()

    assert wait_for(
        triggered.is_set
    )

    monitor.stop()

    assert monitor.state == BargeInState.STOPPED


def test_barge_in_monitor_does_not_trigger_on_silence() -> None:
    triggered = threading.Event()

    monitor = VoiceBargeInMonitor(
        QueueMicrophone([make_input()]),
        FakeVAD(False),
        triggered.set,
        config=BargeInConfig(
            poll_interval_seconds=0.005
        ),
    )

    monitor.start()

    time.sleep(0.03)
    monitor.stop()

    assert triggered.is_set() is False


def test_barge_in_monitor_start_is_idempotent() -> None:
    microphone = QueueMicrophone(
        [make_input()]
    )

    monitor = VoiceBargeInMonitor(
        microphone,
        FakeVAD(False),
        lambda: None,
    )

    monitor.start()
    monitor.start()
    monitor.stop()

    assert monitor.state == BargeInState.STOPPED


def test_barge_in_monitor_callback_error_is_exposed() -> None:
    def broken_callback() -> None:
        raise RuntimeError("boom")

    monitor = VoiceBargeInMonitor(
        QueueMicrophone([make_input()]),
        FakeVAD(True),
        broken_callback,
    )

    monitor.start()

    assert wait_for(
        lambda: monitor.state == BargeInState.ERROR
    )

    assert isinstance(
        monitor.last_error,
        Exception,
    )

    monitor.reset()

    assert monitor.state == BargeInState.STOPPED
    assert monitor.last_error is None


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        "fast",
    ],
)
def test_invalid_barge_in_config_is_rejected(
    value,
) -> None:
    with pytest.raises(
        ValueError,
        match="poll_interval_seconds",
    ):
        BargeInConfig(
            poll_interval_seconds=value
        )
