"""Always-on background listening (OSA 1.5.2.2).

One persistent audio stream feeds a segmenter; completed utterances are transcribed,
gated by the wake word and delivered to a callback. Everything that touches hardware
or heavy libraries is injected (``FrameSource``, ``Transcriber``), so the logic is
testable with fakes. Nothing in the audio path blocks on user input.

Privacy: audio is never stored, utterances that are not addressed to OSA are dropped,
and the optional debug log records only sizes and decisions unless transcript logging
is enabled explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Protocol

from osa.voice.segmenter import AdaptiveEnergyVad, UtteranceSegmenter
from osa.voice.wake import ListenMode, WakeWordGate


class FrameSource(Protocol):
    """Supplies fixed-size PCM16 mono frames from a persistent stream."""

    def start(self) -> None: ...

    def read(self, timeout: float) -> bytes | None: ...

    def drain(self) -> None: ...

    def stop(self) -> None: ...


class Transcriber(Protocol):
    """Turns one utterance of PCM16 mono audio into text."""

    def transcribe(self, audio: bytes, sample_rate_hz: int) -> str: ...


@dataclass(frozen=True, slots=True)
class VoiceCommand:
    """A recognized utterance that was addressed to OSA.

    ``reason`` is the gate decision (wake, wake_only, follow_up, always). For
    ``wake_only`` the text is empty: the user only called the assistant by name.
    """

    text: str
    reason: str
    heard: str = ""


class BackgroundListener:
    """Continuously listens, transcribes and gates speech on a worker thread."""

    def __init__(
        self,
        source: FrameSource,
        transcriber: Transcriber,
        gate: WakeWordGate,
        on_command: Callable[[VoiceCommand], None],
        *,
        segmenter: UtteranceSegmenter | None = None,
        clock: Callable[[], float] | None = None,
        debug: Callable[[str], None] | None = None,
        log_transcripts: bool = False,
        mute_tail_s: float = 0.4,
        read_timeout_s: float = 0.1,
        max_consecutive_errors: int = 5,
        error_backoff_s: float = 0.2,
    ) -> None:
        if mute_tail_s < 0 or error_backoff_s < 0:
            raise ValueError("mute_tail_s and error_backoff_s cannot be negative.")

        if read_timeout_s <= 0:
            raise ValueError("read_timeout_s must be greater than zero.")

        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1.")

        self._source = source
        self._transcriber = transcriber
        self._gate = gate
        self._on_command = on_command
        self._segmenter = segmenter or UtteranceSegmenter(AdaptiveEnergyVad())
        self._sample_rate_hz = self._segmenter.config.sample_rate_hz
        self._clock = clock or time.monotonic
        self._debug = debug
        self._log_transcripts = log_transcripts
        self._mute_tail_s = float(mute_tail_s)
        self._read_timeout_s = float(read_timeout_s)
        self._max_consecutive_errors = max_consecutive_errors
        self._error_backoff_s = float(error_backoff_s)
        self._lock = threading.RLock()
        self._mute_depth = 0
        self._quiet_until = 0.0
        self._drain_pending = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._last_activity = self._clock()

    # --- state ---------------------------------------------------------------------

    @property
    def mode(self) -> ListenMode:
        with self._lock:
            return self._gate.mode

    @mode.setter
    def mode(self, value: ListenMode) -> None:
        with self._lock:
            self._gate.mode = value
            self._drain_pending = True

        self._log(f"mode -> {ListenMode(value).value}")

    @property
    def running(self) -> bool:
        thread = self._thread

        return thread is not None and thread.is_alive()

    @property
    def error(self) -> Exception | None:
        """The error that stopped the worker, if any."""
        return self._error

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._mute_depth > 0 or self._clock() < self._quiet_until

    def stalled_for(self) -> float:
        """Seconds since the worker last started a loop iteration."""
        return self._clock() - self._last_activity

    # --- control -------------------------------------------------------------------

    def mute(self) -> None:
        """Ignore the microphone (call before OSA speaks)."""
        with self._lock:
            self._mute_depth += 1
            self._drain_pending = True

        self._log("mute")

    def unmute(self) -> None:
        """Resume listening after a short tail, so the end of speech is not heard."""
        with self._lock:
            if self._mute_depth > 0:
                self._mute_depth -= 1

            if self._mute_depth == 0:
                self._quiet_until = self._clock() + self._mute_tail_s

            self._drain_pending = True

        self._log("unmute")

    @contextmanager
    def suppressed(self) -> Iterator[None]:
        """Mute the microphone for the duration of a ``with`` block."""
        self.mute()

        try:
            yield
        finally:
            self.unmute()

    def open_follow_up(self) -> None:
        """Let the next utterances through without the wake word for a short window."""
        with self._lock:
            self._gate.open_follow_up()

    def start(self) -> None:
        """Open the audio stream once and start the worker thread."""
        if self.running:
            return

        self._stop_event.clear()
        self._error = None
        self._last_activity = self._clock()
        self._source.start()
        self._log("source started")
        self._thread = threading.Thread(
            target=self._run,
            name="osa-background-listener",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the worker and close the audio stream."""
        self._stop_event.set()
        thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

        try:
            self._source.stop()
        except Exception as exc:
            self._log(f"source stop failed: {exc!r}")

        self._log("stopped")

    # --- processing ----------------------------------------------------------------

    def step(self) -> bool:
        """Read and process at most one frame; return False when none was available."""
        self._last_activity = self._clock()

        with self._lock:
            drain = self._drain_pending
            self._drain_pending = False

        if drain:
            self._source.drain()
            self._segmenter.reset()

        frame = self._source.read(self._read_timeout_s)

        if frame is None:
            return False

        if self.muted or self.mode is ListenMode.OFF:
            self._segmenter.reset()
            return True

        audio = self._segmenter.feed(frame)

        if audio is not None:
            self._handle_utterance(audio)

        return True

    def _handle_utterance(self, audio: bytes) -> None:
        seconds = len(audio) / (2 * self._sample_rate_hz)
        self._log(f"utterance {seconds:.1f}s")
        started = self._clock()

        try:
            heard = (self._transcriber.transcribe(audio, self._sample_rate_hz) or "").strip()
        except Exception as exc:
            self._log(f"transcribe failed: {exc!r}")
            return

        elapsed = self._clock() - started
        self._log(f"transcribed in {elapsed:.1f}s, {len(heard)} chars{self._shown(heard)}")

        with self._lock:
            result = self._gate.process(heard)

        if not result.accepted:
            self._log(f"ignored ({result.reason})")
            return

        self._log(f"accepted ({result.reason}){self._shown(result.command)}")

        try:
            self._on_command(VoiceCommand(result.command, result.reason, heard))
        except Exception as exc:
            self._log(f"on_command failed: {exc!r}")

    def _run(self) -> None:
        failures = 0

        while not self._stop_event.is_set():
            try:
                self.step()
                failures = 0
            except Exception as exc:
                failures += 1
                self._log(f"step failed ({failures}): {exc!r}")

                if failures >= self._max_consecutive_errors:
                    self._error = exc
                    self._log("worker gave up")
                    return

                self._stop_event.wait(self._error_backoff_s)

    def _shown(self, text: str) -> str:
        return f": {text!r}" if self._log_transcripts and text else ""

    def _log(self, message: str) -> None:
        if self._debug is None:
            return

        try:
            self._debug(message)
        except Exception:
            pass
