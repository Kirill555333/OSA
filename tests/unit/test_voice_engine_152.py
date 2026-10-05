"""Tests for playback tracking in the voice engine (OSA 1.5.2.2)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

from osa.voice import engine as engine_module
from osa.voice.engine import OSAVoiceEngine


class _FakeProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            if timeout is None:
                self.returncode = 0
                return 0

            raise subprocess.TimeoutExpired("fake", timeout)

        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def finish(self) -> None:
        self.returncode = 0


def _engine() -> tuple[OSAVoiceEngine, list[_FakeProcess]]:
    engine = OSAVoiceEngine(mode="system")
    engine._has_say = True
    processes: list[_FakeProcess] = []

    def fake_popen(command, **kwargs):
        process = _FakeProcess(list(command))
        processes.append(process)

        return process

    engine._popen = fake_popen

    return engine, processes


def _temp_file() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        return Path(handle.name)


# --- speaking state ----------------------------------------------------------------


def test_speak_starts_playback_and_reports_speaking_until_it_ends() -> None:
    engine, processes = _engine()

    assert engine.speak("**Привет**, мир") is True
    assert processes[0].command == ["say", "Привет, мир"]
    assert engine.is_speaking() is True

    processes[0].finish()

    assert engine.is_speaking() is False


def test_new_speech_interrupts_the_previous_one() -> None:
    engine, processes = _engine()

    engine.speak("первая фраза")
    engine.speak("вторая фраза")

    assert processes[0].terminated is True
    assert engine.is_speaking() is True
    assert processes[1].command == ["say", "вторая фраза"]


def test_stop_terminates_playback() -> None:
    engine, processes = _engine()
    engine.speak("долгая фраза")

    engine.stop()

    assert processes[0].terminated is True
    assert engine.is_speaking() is False


def test_stop_without_speech_is_a_noop() -> None:
    engine, _ = _engine()

    engine.stop()

    assert engine.is_speaking() is False


def test_wait_until_done_times_out_while_speaking_and_succeeds_afterwards() -> None:
    engine, processes = _engine()
    engine.speak("фраза")

    assert engine.wait_until_done(timeout=0.01) is False

    processes[0].finish()

    assert engine.wait_until_done(timeout=0.01) is True
    assert engine.is_speaking() is False


def test_wait_until_done_without_speech_returns_immediately() -> None:
    engine, _ = _engine()

    assert engine.wait_until_done() is True


# --- text handling -----------------------------------------------------------------


def test_empty_or_markup_only_text_is_not_spoken() -> None:
    engine, processes = _engine()

    assert engine.speak("") is False
    assert engine.speak("   ") is False
    assert engine.speak("** ```") is False
    assert processes == []


def test_leading_dash_is_not_passed_to_say_as_an_option() -> None:
    engine, processes = _engine()

    assert engine.speak("- пункт списка") is True
    assert processes[0].command == ["say", "пункт списка"]
    assert engine.speak("---") is False


def test_speak_returns_false_when_no_speech_backend_exists() -> None:
    engine, processes = _engine()
    engine._has_say = False

    assert engine.speak("привет") is False
    assert processes == []


def test_failing_process_start_returns_false() -> None:
    engine, _ = _engine()

    def broken(command, **kwargs):
        raise OSError("no say")

    engine._popen = broken

    assert engine.speak("привет") is False
    assert engine.is_speaking() is False


# --- temporary files ---------------------------------------------------------------


def test_temporary_audio_file_is_removed_when_playback_ends() -> None:
    engine, _ = _engine()
    process = _FakeProcess(["afplay", "x.mp3"])
    path = _temp_file()

    engine._track(process, path)

    assert path.exists() is True
    assert engine.is_speaking() is True

    process.finish()

    assert engine.is_speaking() is False
    assert path.exists() is False


def test_temporary_audio_file_is_removed_when_playback_is_stopped() -> None:
    engine, _ = _engine()
    path = _temp_file()
    engine._track(_FakeProcess(["afplay", "x.mp3"]), path)

    engine.stop()

    assert path.exists() is False


# --- neural voice path -------------------------------------------------------------


def _fake_edge_tts(fail: bool) -> ModuleType:
    module = ModuleType("edge_tts")

    class Communicate:
        def __init__(self, text: str, voice: str) -> None:
            self.text = text

        async def save(self, path: str) -> None:
            if fail:
                raise RuntimeError("service unavailable")

            Path(path).write_bytes(b"audio")

    module.Communicate = Communicate

    return module


def _with_platform(platform: str, fake_edge: ModuleType, test) -> None:
    original_edge = sys.modules.get("edge_tts")
    original_sys = engine_module.sys
    sys.modules["edge_tts"] = fake_edge
    engine_module.sys = SimpleNamespace(platform=platform)

    try:
        test()
    finally:
        engine_module.sys = original_sys

        if original_edge is None:
            sys.modules.pop("edge_tts", None)
        else:
            sys.modules["edge_tts"] = original_edge


def test_neural_voice_on_macos_is_tracked_and_its_file_cleaned_up() -> None:
    def check() -> None:
        engine, processes = _engine()
        engine.mode = "jarvis"

        assert engine.speak("привет") is True

        process = processes[0]
        path = Path(process.command[1])
        assert process.command[0] == "/usr/bin/afplay"
        assert path.exists() is True
        assert engine.is_speaking() is True

        process.finish()

        assert engine.is_speaking() is False
        assert path.exists() is False

    _with_platform("darwin", _fake_edge_tts(fail=False), check)


def test_neural_voice_failure_falls_back_to_system_speech_without_leaking_a_file() -> None:
    created: list[str] = []
    real_named_temp = tempfile.NamedTemporaryFile

    def recording_named_temp(*args, **kwargs):
        handle = real_named_temp(*args, **kwargs)
        created.append(handle.name)

        return handle

    def check() -> None:
        engine, processes = _engine()
        engine.mode = "jarvis"
        tempfile.NamedTemporaryFile = recording_named_temp

        try:
            assert engine.speak("привет") is True
        finally:
            tempfile.NamedTemporaryFile = real_named_temp

        assert processes[0].command == ["say", "привет"]
        assert created
        assert all(Path(name).exists() is False for name in created)

    _with_platform("darwin", _fake_edge_tts(fail=True), check)
