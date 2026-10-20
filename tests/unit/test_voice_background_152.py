"""Tests for the background listener core (OSA 1.5.2.2)."""

from __future__ import annotations

import struct
import threading
import time
from collections import deque

import pytest

from osa.voice.background import BackgroundListener, VoiceCommand
from osa.voice.wake import ListenMode, WakeWordGate

SILENCE = struct.pack("<480h", *([0] * 480))
SPEECH = struct.pack("<480h", *([2500, -2500] * 240))
UTTERANCE = [SILENCE] * 15 + [SPEECH] * 25 + [SILENCE] * 35


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _FakeSource:
    def __init__(self, frames=()) -> None:
        self.frames = deque(frames)
        self.started = False
        self.stopped = False
        self.drains = 0
        self.read_error: Exception | None = None
        self.start_error: Exception | None = None

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error

        self.started = True

    def read(self, timeout: float) -> bytes | None:
        if self.read_error is not None:
            raise self.read_error

        if self.frames:
            return self.frames.popleft()

        time.sleep(0.001)
        return None

    def drain(self) -> None:
        self.drains += 1
        self.frames.clear()

    def stop(self) -> None:
        self.stopped = True


class _ScriptedTranscriber:
    def __init__(self, *texts: str, error: Exception | None = None) -> None:
        self.texts = deque(texts)
        self.calls = 0
        self.error = error

    def transcribe(self, audio: bytes, sample_rate_hz: int) -> str:
        self.calls += 1

        if self.error is not None:
            raise self.error

        return self.texts.popleft() if self.texts else ""


def _listener(
    source: _FakeSource,
    transcriber: _ScriptedTranscriber,
    commands: list,
    *,
    mode: ListenMode = ListenMode.WAKE,
    clock: _Clock | None = None,
    **kwargs,
) -> BackgroundListener:
    clock = clock or _Clock()
    gate = WakeWordGate(mode=mode, clock=clock)

    return BackgroundListener(
        source,
        transcriber,
        gate,
        commands.append,
        clock=clock,
        read_timeout_s=0.01,
        **kwargs,
    )


def _run_all(listener: BackgroundListener, source: _FakeSource) -> None:
    while source.frames:
        listener.step()


# --- gating ------------------------------------------------------------------------


def test_addressed_utterance_becomes_a_command() -> None:
    source = _FakeSource(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(source, _ScriptedTranscriber("Оса, открой хром"), commands)

    _run_all(listener, source)

    assert commands == [VoiceCommand("открой хром", "wake", "Оса, открой хром")]


def test_utterance_not_addressed_to_osa_is_dropped() -> None:
    source = _FakeSource(UTTERANCE)
    transcriber = _ScriptedTranscriber("открой хром")
    commands: list[VoiceCommand] = []
    listener = _listener(source, transcriber, commands)

    _run_all(listener, source)

    assert commands == []
    assert transcriber.calls == 1


def test_wake_word_alone_is_delivered_as_wake_only() -> None:
    source = _FakeSource(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(source, _ScriptedTranscriber("Оса."), commands)

    _run_all(listener, source)

    assert [(c.text, c.reason) for c in commands] == [("", "wake_only")]


def test_always_mode_delivers_everything_spoken() -> None:
    source = _FakeSource(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(
        source,
        _ScriptedTranscriber("просто разговор"),
        commands,
        mode=ListenMode.ALWAYS,
    )

    _run_all(listener, source)

    assert [(c.text, c.reason) for c in commands] == [("просто разговор", "always")]


def test_off_mode_never_transcribes() -> None:
    source = _FakeSource(UTTERANCE)
    transcriber = _ScriptedTranscriber("Оса, привет")
    commands: list[VoiceCommand] = []
    listener = _listener(source, transcriber, commands, mode=ListenMode.OFF)

    _run_all(listener, source)

    assert transcriber.calls == 0
    assert commands == []


def test_follow_up_after_answer_needs_no_wake_word() -> None:
    source = _FakeSource(UTTERANCE + UTTERANCE)
    commands: list[VoiceCommand] = []
    transcriber = _ScriptedTranscriber("Оса, открой хром", "а теперь калькулятор")
    listener = _listener(source, transcriber, commands)

    _run_all(listener, source)

    assert [c.reason for c in commands] == ["wake", "follow_up"]


def test_open_follow_up_is_available_to_the_application() -> None:
    source = _FakeSource(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(source, _ScriptedTranscriber("открой хром"), commands)

    listener.open_follow_up()
    _run_all(listener, source)

    assert [c.reason for c in commands] == ["follow_up"]


# --- muting ------------------------------------------------------------------------


def test_muted_audio_is_ignored() -> None:
    source = _FakeSource()
    transcriber = _ScriptedTranscriber("Оса, привет")
    commands: list[VoiceCommand] = []
    listener = _listener(source, transcriber, commands)

    listener.mute()
    listener.step()
    source.frames.extend(UTTERANCE)
    _run_all(listener, source)

    assert transcriber.calls == 0
    assert commands == []
    assert listener.muted is True


def test_unmute_keeps_a_short_silence_tail() -> None:
    clock = _Clock()
    source = _FakeSource()
    listener = _listener(source, _ScriptedTranscriber(), [], clock=clock)

    listener.mute()
    listener.unmute()

    assert listener.muted is True

    clock.now += 0.5

    assert listener.muted is False


def test_mute_levels_nest() -> None:
    clock = _Clock()
    listener = _listener(_FakeSource(), _ScriptedTranscriber(), [], clock=clock)

    listener.mute()
    listener.mute()
    listener.unmute()
    clock.now += 5.0

    assert listener.muted is True

    listener.unmute()
    clock.now += 5.0

    assert listener.muted is False


def test_suppressed_context_manager_mutes_and_unmutes() -> None:
    clock = _Clock()
    listener = _listener(_FakeSource(), _ScriptedTranscriber(), [], clock=clock)

    with listener.suppressed():
        assert listener.muted is True

    clock.now += 5.0

    assert listener.muted is False


def test_mute_and_unmute_discard_captured_audio() -> None:
    source = _FakeSource([SILENCE] * 5)
    listener = _listener(source, _ScriptedTranscriber(), [])

    listener.mute()
    listener.step()
    listener.unmute()
    listener.step()

    assert source.drains == 2
    assert source.frames == deque()


def test_listening_resumes_after_the_tail() -> None:
    clock = _Clock()
    source = _FakeSource()
    commands: list[VoiceCommand] = []
    listener = _listener(source, _ScriptedTranscriber("Оса, привет"), commands, clock=clock)

    listener.mute()
    listener.unmute()
    listener.step()
    clock.now += 1.0
    source.frames.extend(UTTERANCE)
    _run_all(listener, source)

    assert [c.text for c in commands] == ["привет"]


def test_changing_mode_discards_pending_audio() -> None:
    source = _FakeSource()
    listener = _listener(source, _ScriptedTranscriber(), [])

    listener.mode = ListenMode.ALWAYS
    listener.step()

    assert listener.mode is ListenMode.ALWAYS
    assert source.drains == 1


# --- robustness --------------------------------------------------------------------


def test_transcription_error_does_not_stop_listening() -> None:
    source = _FakeSource(UTTERANCE + UTTERANCE)
    commands: list[VoiceCommand] = []
    calls = {"count": 0}

    class _Flaky:
        def transcribe(self, audio: bytes, sample_rate_hz: int) -> str:
            calls["count"] += 1

            if calls["count"] == 1:
                raise RuntimeError("model crashed")

            return "Оса, привет"

    gate = WakeWordGate(clock=_Clock())
    listener = BackgroundListener(source, _Flaky(), gate, commands.append, read_timeout_s=0.01)

    _run_all(listener, source)

    assert [c.text for c in commands] == ["привет"]


def test_failing_callback_is_swallowed() -> None:
    source = _FakeSource(UTTERANCE)

    def boom(command: VoiceCommand) -> None:
        raise RuntimeError("ui failed")

    gate = WakeWordGate(clock=_Clock())
    listener = BackgroundListener(
        source,
        _ScriptedTranscriber("Оса, привет"),
        gate,
        boom,
        read_timeout_s=0.01,
    )

    _run_all(listener, source)

    assert listener.step() is False


def test_step_reports_when_no_frame_is_available() -> None:
    listener = _listener(_FakeSource(), _ScriptedTranscriber(), [])

    assert listener.step() is False


def test_stalled_for_measures_time_since_last_iteration() -> None:
    clock = _Clock()
    listener = _listener(_FakeSource(), _ScriptedTranscriber(), [], clock=clock)

    clock.now += 5.0
    assert listener.stalled_for() == pytest.approx(5.0)

    listener.step()
    assert listener.stalled_for() == pytest.approx(0.0)


# --- privacy of the debug log ------------------------------------------------------


def test_debug_log_omits_transcripts_by_default() -> None:
    messages: list[str] = []
    source = _FakeSource(UTTERANCE)
    listener = _listener(
        source,
        _ScriptedTranscriber("Оса, секретная фраза"),
        [],
        debug=messages.append,
    )

    _run_all(listener, source)

    assert messages
    assert not any("секретная" in message for message in messages)


def test_debug_log_can_include_transcripts_when_enabled() -> None:
    messages: list[str] = []
    source = _FakeSource(UTTERANCE)
    listener = _listener(
        source,
        _ScriptedTranscriber("Оса, секретная фраза"),
        [],
        debug=messages.append,
        log_transcripts=True,
    )

    _run_all(listener, source)

    assert any("секретная" in message for message in messages)


def test_failing_debug_sink_never_breaks_listening() -> None:
    def broken(message: str) -> None:
        raise OSError("disk full")

    source = _FakeSource(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(source, _ScriptedTranscriber("Оса, привет"), commands, debug=broken)

    _run_all(listener, source)

    assert len(commands) == 1


# --- worker thread -----------------------------------------------------------------


def test_worker_thread_delivers_commands_and_stops_cleanly() -> None:
    source = _FakeSource(UTTERANCE)
    commands: list[VoiceCommand] = []
    received = threading.Event()

    def on_command(command: VoiceCommand) -> None:
        commands.append(command)
        received.set()

    gate = WakeWordGate()
    listener = BackgroundListener(
        source,
        _ScriptedTranscriber("Оса, привет"),
        gate,
        on_command,
        read_timeout_s=0.01,
    )

    listener.start()

    try:
        assert received.wait(5.0) is True
        assert listener.running is True
    finally:
        listener.stop()

    assert listener.running is False
    assert source.started is True
    assert source.stopped is True
    assert [c.text for c in commands] == ["привет"]


def test_start_twice_does_not_open_the_source_twice() -> None:
    source = _FakeSource()
    listener = _listener(source, _ScriptedTranscriber(), [])

    listener.start()

    try:
        listener.start()
        assert listener.running is True
    finally:
        listener.stop()


def test_worker_gives_up_after_repeated_source_failures() -> None:
    source = _FakeSource()
    source.read_error = RuntimeError("device lost")
    listener = _listener(
        source,
        _ScriptedTranscriber(),
        [],
        max_consecutive_errors=3,
        error_backoff_s=0.0,
    )

    listener.start()
    listener._thread.join(5.0)

    try:
        assert listener.running is False
        assert isinstance(listener.error, RuntimeError)
    finally:
        listener.stop()


def test_start_failure_propagates_and_leaves_listener_stopped() -> None:
    source = _FakeSource()
    source.start_error = OSError("no microphone")
    listener = _listener(source, _ScriptedTranscriber(), [])

    with pytest.raises(OSError):
        listener.start()

    assert listener.running is False


# --- configuration -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mute_tail_s": -1},
        {"error_backoff_s": -1},
        {"read_timeout_s": 0},
        {"max_consecutive_errors": 0},
    ],
)
def test_constructor_rejects_invalid_values(kwargs: dict) -> None:
    arguments = {
        "source": _FakeSource(),
        "transcriber": _ScriptedTranscriber(),
        "gate": WakeWordGate(),
        "on_command": lambda command: None,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        BackgroundListener(**arguments)
