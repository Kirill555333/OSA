"""Tests for the unified keyboard and voice input hub (OSA 1.5.2.3)."""

from __future__ import annotations

import io
import queue
import threading
import time
from types import SimpleNamespace

from osa.shell.input_hub import InputEvent, InputHub


class _BlockingStream:
    """A stdin stand-in whose readline blocks until a line is fed."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()
        self.reads = 0

    def feed(self, line: str) -> None:
        self._lines.put(line)

    def close(self) -> None:
        self._lines.put("")

    def readline(self) -> str:
        self.reads += 1

        return self._lines.get()


class _BrokenStream:
    def readline(self) -> str:
        raise OSError("stdin lost")


def _drain(hub: InputHub, count: int, timeout: float = 2.0) -> list[InputEvent]:
    events = []

    for _ in range(count):
        event = hub.next_event(timeout)
        assert event is not None
        events.append(event)

    return events


# --- keyboard ----------------------------------------------------------------------


def test_typed_lines_become_text_events_followed_by_eof() -> None:
    hub = InputHub(io.StringIO("привет\r\nвторая строка\n"))

    hub.start()

    assert _drain(hub, 3) == [
        InputEvent("text", "привет"),
        InputEvent("text", "вторая строка"),
        InputEvent("eof"),
    ]


def test_empty_line_is_delivered_as_empty_text() -> None:
    hub = InputHub(io.StringIO("\n"))

    hub.start()

    assert _drain(hub, 2)[0] == InputEvent("text", "")


def test_next_event_times_out_without_blocking_the_caller() -> None:
    stream = _BlockingStream()
    hub = InputHub(stream)
    hub.start()

    started = time.monotonic()
    event = hub.next_event(0.05)

    assert event is None
    assert time.monotonic() - started < 1.0

    stream.close()


def test_line_typed_later_is_delivered_by_the_reader_thread() -> None:
    stream = _BlockingStream()
    hub = InputHub(stream)
    hub.start()

    threading.Timer(0.05, stream.feed, args=("позже\n",)).start()

    assert hub.next_event(2.0) == InputEvent("text", "позже")

    stream.close()


def test_start_twice_runs_a_single_reader() -> None:
    stream = _BlockingStream()
    hub = InputHub(stream)

    hub.start()
    hub.start()
    stream.feed("раз\n")

    assert hub.next_event(2.0) == InputEvent("text", "раз")
    assert hub.next_event(0.05) is None

    stream.close()


def test_broken_input_ends_with_eof() -> None:
    hub = InputHub(_BrokenStream())

    hub.start()

    assert hub.next_event(2.0) == InputEvent("eof")


# --- voice -------------------------------------------------------------------------


def test_voice_commands_are_posted_as_voice_events() -> None:
    hub = InputHub(io.StringIO(""))

    hub.post_voice(SimpleNamespace(text="открой хром", reason="wake"))

    assert hub.pending() == 1
    assert hub.next_event(0.1) == InputEvent("voice", "открой хром", "wake")
    assert hub.pending() == 0


def test_keyboard_and_voice_share_one_ordered_queue() -> None:
    stream = _BlockingStream()
    hub = InputHub(stream)
    hub.start()

    hub.post_voice(SimpleNamespace(text="голосом", reason="wake"))
    stream.feed("текстом\n")

    kinds = [event.kind for event in _drain(hub, 2)]

    assert kinds == ["voice", "text"]

    stream.close()


# --- confirmation answers ----------------------------------------------------------


def test_ask_text_returns_the_next_typed_line_and_drops_voice() -> None:
    hub = InputHub(io.StringIO(""))
    hub.post_voice(SimpleNamespace(text="да", reason="follow_up"))
    hub.post_voice(SimpleNamespace(text="разрешаю", reason="always"))
    hub._events.put(InputEvent("text", "y"))

    assert hub.ask_text(timeout=1.0) == "y"
    assert hub.pending() == 0


def test_ask_text_times_out_with_none() -> None:
    hub = InputHub(io.StringIO(""))

    assert hub.ask_text(timeout=0.05) is None


def test_ask_text_with_closed_input_returns_none_and_keeps_eof_visible() -> None:
    hub = InputHub(io.StringIO(""))
    hub._events.put(InputEvent("eof"))

    assert hub.ask_text(timeout=1.0) is None
    assert hub.next_event(0.1) == InputEvent("eof")


def test_voice_events_alone_never_satisfy_ask_text() -> None:
    hub = InputHub(io.StringIO(""))

    for _ in range(3):
        hub.post_voice(SimpleNamespace(text="да", reason="wake"))

    assert hub.ask_text(timeout=0.1) is None
