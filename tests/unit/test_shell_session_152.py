"""Tests for session helpers around background listening (OSA 1.5.2.3)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from osa.shell.session import (
    OutputSpeaker,
    create_confirmation_callback,
    create_debug_sink,
    debug_settings_from_env,
    handle_listen_command,
    listener_problem,
)
from osa.voice.wake import ListenMode


class _Engine:
    def __init__(self, polls_speaking: int = 3, speak_result: bool = True) -> None:
        self.polls_speaking = polls_speaking
        self.speak_result = speak_result
        self.spoken: list[str] = []
        self.stopped = False
        self.error: Exception | None = None

    def speak(self, text: str) -> bool:
        if self.error is not None:
            raise self.error

        self.spoken.append(text)
        return self.speak_result

    def is_speaking(self) -> bool:
        if self.polls_speaking > 0 and not self.stopped:
            self.polls_speaking -= 1
            return True

        return False

    def stop(self) -> None:
        self.stopped = True


class _Listener:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.mode = ListenMode.WAKE
        self.error: Exception | None = None
        self.stalled = 0.0

    def mute(self) -> None:
        self.events.append("mute")

    def unmute(self) -> None:
        self.events.append("unmute")

    def stalled_for(self) -> float:
        return self.stalled


class _Hub:
    def __init__(self, pending: int = 0, answers=()) -> None:
        self._pending = pending
        self.answers = list(answers)
        self.asked_with: list[float | None] = []

    def pending(self) -> int:
        return self._pending

    def ask_text(self, timeout: float | None = None) -> str | None:
        self.asked_with.append(timeout)

        return self.answers.pop(0) if self.answers else None


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _speaker(engine, listener=None, hub=None, clock: _Clock | None = None, **kwargs):
    clock = clock or _Clock()

    return OutputSpeaker(
        engine,
        listener,
        hub,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )


# --- OutputSpeaker -----------------------------------------------------------------


def test_say_mutes_the_microphone_for_the_whole_speech() -> None:
    engine = _Engine(polls_speaking=3)
    listener = _Listener()

    assert _speaker(engine, listener).say("привет") is True

    assert engine.spoken == ["привет"]
    assert listener.events == ["mute", "unmute"]
    assert engine.polls_speaking == 0


def test_say_waits_until_speech_has_finished() -> None:
    clock = _Clock()
    engine = _Engine(polls_speaking=5)

    _speaker(engine, clock=clock).say("долго")

    assert clock.now == pytest.approx(5 * 0.05)


def test_typing_or_speaking_interrupts_the_speech() -> None:
    engine = _Engine(polls_speaking=100)
    hub = _Hub(pending=1)

    assert _speaker(engine, _Listener(), hub).say("перебей меня") is True

    assert engine.stopped is True


def test_speech_that_never_ends_is_stopped_after_the_maximum_wait() -> None:
    clock = _Clock()
    engine = _Engine(polls_speaking=10**9)

    _speaker(engine, clock=clock, max_wait_s=1.0).say("бесконечно")

    assert engine.stopped is True
    assert 1.0 <= clock.now < 1.2


def test_nothing_spoken_still_unmutes_the_microphone() -> None:
    engine = _Engine(speak_result=False)
    listener = _Listener()

    assert _speaker(engine, listener).say("") is False

    assert listener.events == ["mute", "unmute"]


def test_engine_failure_does_not_break_the_session_and_unmutes() -> None:
    engine = _Engine()
    engine.error = RuntimeError("audio device busy")
    listener = _Listener()

    assert _speaker(engine, listener).say("привет") is False

    assert listener.events == ["mute", "unmute"]


def test_say_works_without_listener_and_hub() -> None:
    assert _speaker(_Engine(polls_speaking=1)).say("тихо") is True


def test_speaker_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        OutputSpeaker(_Engine(), max_wait_s=0)

    with pytest.raises(ValueError):
        OutputSpeaker(_Engine(), poll_s=0)


# --- /listen -----------------------------------------------------------------------


def test_listen_command_without_listener_explains_how_to_enable() -> None:
    assert "--listen" in handle_listen_command("/listen", None)


def test_listen_command_reports_current_mode() -> None:
    listener = _Listener()

    reply = handle_listen_command("/listen", listener)

    assert "wake" in reply


@pytest.mark.parametrize("word", ["off", "wake", "always", "ALWAYS"])
def test_listen_command_switches_mode(word: str) -> None:
    listener = _Listener()

    reply = handle_listen_command(f"/listen {word}", listener)

    assert listener.mode is ListenMode.parse(word)
    assert listener.mode.value in reply


def test_listen_command_rejects_unknown_mode_and_keeps_current() -> None:
    listener = _Listener()

    reply = handle_listen_command("/listen loud", listener)

    assert listener.mode is ListenMode.WAKE
    assert "loud" in reply


# --- health ------------------------------------------------------------------------


def test_listener_problem_is_none_for_a_healthy_or_absent_listener() -> None:
    assert listener_problem(None) is None
    assert listener_problem(_Listener()) is None


def test_listener_problem_reports_a_crashed_worker() -> None:
    listener = _Listener()
    listener.error = RuntimeError("device lost")

    assert "device lost" in listener_problem(listener)


def test_listener_problem_reports_a_stalled_worker() -> None:
    listener = _Listener()
    listener.stalled = 120.0

    assert "120" in listener_problem(listener)
    assert listener_problem(listener, stall_after_s=300.0) is None


# --- debug log ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", (False, False)),
        ("0", (False, False)),
        ("1", (True, False)),
        ("TRUE", (True, False)),
        ("text", (True, True)),
        ("2", (True, True)),
        ("nonsense", (False, False)),
    ],
)
def test_debug_settings_from_env(value: str, expected: tuple[bool, bool]) -> None:
    assert debug_settings_from_env({"OSA_VOICE_DEBUG": value}) == expected


def test_debug_settings_default_to_disabled() -> None:
    assert debug_settings_from_env({}) == (False, False)


def test_debug_sink_appends_timestamped_lines_and_creates_directories() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "logs" / "voice-debug.log"
        sink = create_debug_sink(path, clock=lambda: 0.0)

        sink("first")
        sink("second")

        lines = path.read_text(encoding="utf-8").splitlines()

        assert len(lines) == 2
        assert lines[0].endswith(" first")
        assert lines[1].endswith(" second")
        assert lines[0][2] == ":" and lines[0][5] == ":"


def test_debug_sink_never_raises_when_the_file_cannot_be_written() -> None:
    with tempfile.TemporaryDirectory() as folder:
        blocker = Path(folder) / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        sink = create_debug_sink(blocker / "voice.log")

        sink("ignored")


# --- confirmation ------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES ", "д", "да", "Да"])
def test_typed_yes_confirms(answer: str) -> None:
    written: list[str] = []
    callback = create_confirmation_callback(_Hub(answers=[answer]), write=written.append)

    assert callback("Закрыть приложение Chrome") is True
    assert "Закрыть приложение Chrome" in written[0]


@pytest.mark.parametrize("answer", ["n", "no", "", "нет", "yep", "yes please", "ok"])
def test_anything_but_a_clear_yes_denies(answer: str) -> None:
    callback = create_confirmation_callback(_Hub(answers=[answer]), write=lambda text: None)

    assert callback("Удалить файл") is False


def test_no_answer_or_closed_input_denies() -> None:
    callback = create_confirmation_callback(_Hub(answers=[]), write=lambda text: None)

    assert callback("Удалить файл") is False


def test_confirmation_uses_the_configured_timeout_and_tolerates_extra_arguments() -> None:
    hub = _Hub(answers=["y"])
    callback = create_confirmation_callback(hub, timeout_s=7.5, write=lambda text: None)

    assert callback("Действие", SimpleNamespace(), reason="x") is True
    assert hub.asked_with == [7.5]


def test_prompt_states_that_voice_cannot_confirm() -> None:
    written: list[str] = []
    callback = create_confirmation_callback(_Hub(answers=["n"]), write=written.append)

    callback("Действие")

    assert "голосом" in written[0]
