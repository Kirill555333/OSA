"""Unified input for the interactive session (OSA 1.5.2.3).

Typed lines and background voice commands arrive on one queue, so the main loop never
blocks on ``input()`` and can react to either source. A daemon thread reads the
keyboard (it works the same on macOS, Linux and Windows). Confirmation prompts read
their answer from the same queue through ``ask_text``: only typed answers count, so
spoken words can never confirm a dangerous action.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True, slots=True)
class InputEvent:
    """One thing the user did: typed a line, spoke a command, or closed the input.

    ``kind`` is ``text``, ``voice`` or ``eof``. For voice events ``reason`` is the
    wake-word gate decision (wake, wake_only, follow_up, always).
    """

    kind: str
    text: str = ""
    reason: str = ""


class InputHub:
    """Merges keyboard lines and voice commands into a single event stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream
        self._events: queue.Queue[InputEvent] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start reading the keyboard on a daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._read_lines,
            name="osa-stdin-reader",
            daemon=True,
        )
        self._thread.start()

    def post_voice(self, command: Any) -> None:
        """Callback for the background listener (any object with ``text`` and ``reason``)."""
        self._events.put(InputEvent("voice", str(command.text), str(command.reason)))

    def next_event(self, timeout: float | None = None) -> InputEvent | None:
        """Return the next event, or None when nothing arrived within the timeout."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def pending(self) -> int:
        """Return how many events are waiting to be processed."""
        return self._events.qsize()

    def ask_text(self, timeout: float | None = None) -> str | None:
        """Return the next typed line; voice events are dropped, never used as answers.

        Returns None when the input is closed or the timeout expires. A closed input
        stays visible to the main loop afterwards.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            remaining = None if deadline is None else deadline - time.monotonic()

            if remaining is not None and remaining <= 0:
                return None

            event = self.next_event(remaining)

            if event is None:
                return None

            if event.kind == "text":
                return event.text

            if event.kind == "eof":
                self._events.put(event)
                return None

    def _read_lines(self) -> None:
        stream = self._stream if self._stream is not None else sys.stdin

        try:
            while True:
                line = stream.readline()

                if not line:
                    break

                self._events.put(InputEvent("text", line.rstrip("\r\n")))
        except Exception:
            pass
        finally:
            self._events.put(InputEvent("eof"))
