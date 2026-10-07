"""Tests for extra wake names and the recent-phrases memory in the listener (OSA 1.5.2.4)."""

from __future__ import annotations

from collections import deque
import struct
import time

from osa.voice.background import BackgroundListener, VoiceCommand
from osa.voice.background_audio import FasterWhisperTranscriber
from osa.voice.wake import WakeWordGate

SILENCE = struct.pack("<480h", *([0] * 480))
SPEECH = struct.pack("<480h", *([2500, -2500] * 240))
UTTERANCE = [SILENCE] * 15 + [SPEECH] * 25 + [SILENCE] * 35


class _Source:
    def __init__(self, frames=()) -> None:
        self.frames = deque(frames)

    def start(self) -> None:
        return None

    def read(self, timeout: float) -> bytes | None:
        if self.frames:
            return self.frames.popleft()

        time.sleep(0.001)
        return None

    def drain(self) -> None:
        self.frames.clear()

    def stop(self) -> None:
        return None


class _Transcriber:
    def __init__(self, *texts: str) -> None:
        self.texts = deque(texts)
        self.names_seen: list[tuple[str, ...]] = []

    def transcribe(self, audio: bytes, sample_rate_hz: int) -> str:
        return self.texts.popleft() if self.texts else ""

    def update_names(self, names) -> None:
        self.names_seen.append(tuple(names))


def _listener(transcriber: _Transcriber, source: _Source, commands: list) -> BackgroundListener:
    return BackgroundListener(
        source,
        transcriber,
        WakeWordGate(),
        commands.append,
        read_timeout_s=0.01,
    )


def _run(listener: BackgroundListener, source: _Source) -> None:
    while source.frames:
        listener.step()


# --- extra names -------------------------------------------------------------------


def test_extra_name_is_heard_after_set_wake_names() -> None:
    source = _Source(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(_Transcriber("Джарвис, открой хром"), source, commands)

    listener.set_wake_names(["джарвис"])
    _run(listener, source)

    assert [(c.text, c.reason) for c in commands] == [("открой хром", "wake")]


def test_builtin_name_still_works_with_an_extra_name() -> None:
    source = _Source(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(_Transcriber("Оса, открой хром"), source, commands)

    listener.set_wake_names(["джарвис"])
    _run(listener, source)

    assert [c.text for c in commands] == ["открой хром"]


def test_clearing_names_stops_the_extra_name() -> None:
    source = _Source(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(_Transcriber("Джарвис, открой хром"), source, commands)

    listener.set_wake_names(["джарвис"])
    listener.set_wake_names([])
    _run(listener, source)

    assert commands == []


def test_transcriber_is_told_the_display_names() -> None:
    transcriber = _Transcriber()
    listener = _listener(transcriber, _Source(), [])

    listener.set_wake_names(["джарвис", "jarvis"])

    assert transcriber.names_seen == [("Оса", "Джарвис", "Jarvis")]


def test_transcriber_without_update_names_is_tolerated() -> None:
    class _Plain:
        def transcribe(self, audio: bytes, sample_rate_hz: int) -> str:
            return ""

    listener = BackgroundListener(
        _Source(),
        _Plain(),
        WakeWordGate(),
        lambda command: None,
        read_timeout_s=0.01,
    )

    listener.set_wake_names(["джарвис"])


def test_failing_transcriber_update_does_not_break_the_listener() -> None:
    class _Broken(_Transcriber):
        def update_names(self, names) -> None:
            raise RuntimeError("prompt failed")

    source = _Source(UTTERANCE)
    commands: list[VoiceCommand] = []
    listener = _listener(_Broken("Джарвис, привет"), source, commands)

    listener.set_wake_names(["джарвис"])
    _run(listener, source)

    assert [c.text for c in commands] == ["привет"]


# --- recent phrases ----------------------------------------------------------------


def test_recent_heard_records_accepted_and_ignored_phrases_with_the_decision() -> None:
    source = _Source(UTTERANCE + UTTERANCE)
    listener = _listener(_Transcriber("просто разговор", "Оса, привет"), source, [])

    _run(listener, source)

    assert listener.recent_heard() == [
        ("просто разговор", "no_wake"),
        ("Оса, привет", "wake"),
    ]


def test_recent_heard_keeps_only_the_latest_twelve() -> None:
    source = _Source(UTTERANCE * 14)
    texts = [f"фраза {number}" for number in range(14)]
    listener = _listener(_Transcriber(*texts), source, [])

    _run(listener, source)

    heard = listener.recent_heard()
    assert len(heard) == 12
    assert heard[0][0] == "фраза 2"
    assert heard[-1][0] == "фраза 13"


def test_recent_heard_returns_a_copy() -> None:
    source = _Source(UTTERANCE)
    listener = _listener(_Transcriber("Оса, привет"), source, [])
    _run(listener, source)

    snapshot = listener.recent_heard()
    snapshot.clear()

    assert len(listener.recent_heard()) == 1


# --- transcriber prompt ------------------------------------------------------------


class _Model:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def transcribe(self, samples, **kwargs):
        self.prompts.append(kwargs.get("initial_prompt"))

        return iter([]), None


def test_update_names_changes_the_recognition_prompt() -> None:
    model = _Model()
    transcriber = FasterWhisperTranscriber(model=model)

    transcriber.transcribe(b"\x01\x00" * 100, 16000)
    transcriber.update_names(["Оса", "Джарвис"])
    transcriber.transcribe(b"\x01\x00" * 100, 16000)

    assert "Оса" in model.prompts[0]
    assert model.prompts[1] == "Разговор с ассистентом по имени Оса или Джарвис."


def test_update_names_ignores_empty_input() -> None:
    model = _Model()
    transcriber = FasterWhisperTranscriber(model=model)

    transcriber.update_names([])
    transcriber.update_names(["", ""])
    transcriber.transcribe(b"\x01\x00" * 100, 16000)

    assert "Оса" in model.prompts[0]
