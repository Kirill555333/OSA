"""Helpers for the interactive session with background listening (OSA 1.5.2.3).

Small, dependency-light pieces that the main loop wires together: speaking with the
microphone muted, the ``/listen`` command, the optional voice debug log, health
reporting and keyboard-only confirmation of dangerous actions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import sys
import time
from typing import Any

from osa.voice.wake import ListenMode

_YES = frozenset({"y", "yes", "д", "да"})


class OutputSpeaker:
    """Speaks text while the microphone is muted, and lets the user interrupt it."""

    def __init__(
        self,
        engine: Any,
        listener: Any | None = None,
        hub: Any | None = None,
        *,
        max_wait_s: float = 120.0,
        poll_s: float = 0.05,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if max_wait_s <= 0 or poll_s <= 0:
            raise ValueError("max_wait_s and poll_s must be greater than zero.")

        self._engine = engine
        self._listener = listener
        self._hub = hub
        self._max_wait_s = max_wait_s
        self._poll_s = poll_s
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic

    def say(self, text: str) -> bool:
        """Speak and wait for the end; return False when nothing was spoken.

        Typing or speaking while OSA talks interrupts the speech. The microphone is
        muted for the whole time, so OSA never hears its own voice.
        """
        if self._listener is not None:
            self._listener.mute()

        try:
            try:
                if not self._engine.speak(text):
                    return False
            except Exception:
                return False

            deadline = self._monotonic() + self._max_wait_s

            while self._engine.is_speaking():
                interrupted = self._hub is not None and self._hub.pending() > 0

                if interrupted or self._monotonic() >= deadline:
                    self._engine.stop()
                    break

                self._sleep(self._poll_s)

            return True
        finally:
            if self._listener is not None:
                self._listener.unmute()


def handle_listen_command(line: str, listener: Any | None) -> str:
    """Handle ``/listen`` and ``/listen <off|wake|always>``; return the reply text."""
    if listener is None:
        return "Фоновое прослушивание не запущено. Запустите OSA с флагом --listen."

    parts = line.split()

    if len(parts) < 2:
        return f"Режим прослушивания: {listener.mode.value}. Сменить: /listen off|wake|always"

    try:
        mode = ListenMode.parse(parts[1])
    except ValueError as exc:
        return str(exc)

    listener.mode = mode

    return f"Режим прослушивания: {mode.value}"


def listener_problem(listener: Any | None, *, stall_after_s: float = 90.0) -> str | None:
    """Describe a listener failure, or return None when it is healthy."""
    if listener is None:
        return None

    error = listener.error

    if error is not None:
        return f"Микрофон перестал работать ({error!r}). Продолжаю только с клавиатурой."

    stalled = listener.stalled_for()

    if stalled > stall_after_s:
        return (
            f"Слушатель не отвечает уже {stalled:.0f} с. "
            "Перезапустите OSA, если голос не работает."
        )

    return None


def debug_settings_from_env(env: Mapping[str, str]) -> tuple[bool, bool]:
    """Read OSA_VOICE_DEBUG: returns (enabled, include_transcripts).

    ``1`` enables a log with sizes and decisions only; ``text`` also logs what was
    recognized (it stays on this computer, in logs/voice-debug.log).
    """
    value = str(env.get("OSA_VOICE_DEBUG", "")).strip().lower()

    if value in ("1", "true", "yes", "on"):
        return True, False

    if value in ("text", "transcripts", "2"):
        return True, True

    return False, False


def create_debug_sink(
    path: Path,
    *,
    clock: Callable[[], float] | None = None,
) -> Callable[[str], None]:
    """Return a function that appends timestamped lines to a log file."""
    now = clock or time.time

    def sink(message: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H:%M:%S", time.localtime(now()))

            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{stamp} {message}\n")
        except OSError:
            pass

    return sink


def create_confirmation_callback(
    hub: Any,
    *,
    timeout_s: float = 120.0,
    write: Callable[[str], None] | None = None,
) -> Callable[..., bool]:
    """Build a confirmation callback that accepts only a typed yes.

    Voice is deliberately ignored: a misheard word must never approve closing an
    application or writing a file. No answer within the timeout means "no".
    """

    def default_write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    out = write or default_write

    def callback(description: str, *args: Any, **kwargs: Any) -> bool:
        out(
            "\n[ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ]\n"
            f"{description}\n"
            "Разрешить? [y/N] (ответьте с клавиатуры, голосом подтвердить нельзя): "
        )
        answer = hub.ask_text(timeout=timeout_s)

        return answer is not None and answer.strip().lower() in _YES

    return callback
