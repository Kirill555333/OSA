"""Hardware adapters for background listening (OSA 1.5.2.2).

``SoundDeviceFrameSource`` keeps one input stream open for the whole session instead of
opening and closing it per phrase. ``FasterWhisperTranscriber`` runs Whisper locally
and drops segments that look like silence or hallucination. Heavy libraries are
imported lazily, so importing this module is cheap and safe without them.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from typing import Any

_DEFAULT_PROMPT = "Разговор с ассистентом по имени Оса."


class SoundDeviceFrameSource:
    """Persistent microphone stream delivering fixed-size PCM16 mono frames."""

    def __init__(
        self,
        *,
        sample_rate_hz: int = 16000,
        frame_ms: int = 30,
        device: int | str | None = None,
        max_queue_frames: int = 400,
        stop_timeout_s: float = 2.0,
    ) -> None:
        if sample_rate_hz <= 0 or frame_ms <= 0 or max_queue_frames < 1:
            raise ValueError("sample_rate_hz, frame_ms and max_queue_frames must be positive.")

        if (sample_rate_hz * frame_ms) % 1000 != 0:
            raise ValueError("frame_ms must correspond to a whole number of samples.")

        self._sample_rate_hz = sample_rate_hz
        self._frame_samples = sample_rate_hz * frame_ms // 1000
        self._device = device
        self._stop_timeout_s = stop_timeout_s
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue_frames)
        self._stream: Any = None

    @property
    def frame_bytes(self) -> int:
        return self._frame_samples * 2

    def start(self) -> None:
        """Open and start the input stream (once)."""
        if self._stream is not None:
            return

        import sounddevice

        stream = sounddevice.RawInputStream(
            samplerate=self._sample_rate_hz,
            blocksize=self._frame_samples,
            channels=1,
            dtype="int16",
            device=self._device,
            callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def read(self, timeout: float) -> bytes | None:
        """Return the next frame, or None when none arrived within the timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard everything that has been captured but not yet read."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        """Close the stream without risking a permanent hang in the audio driver."""
        stream = self._stream
        self._stream = None

        if stream is None:
            return

        def close() -> None:
            try:
                stream.abort()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

        closer = threading.Thread(target=close, name="osa-audio-close", daemon=True)
        closer.start()
        closer.join(self._stop_timeout_s)

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        frame = bytes(indata)

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass


def keep_segment(
    no_speech_prob: float | None,
    avg_logprob: float | None,
    *,
    max_no_speech_prob: float = 0.6,
    min_avg_logprob: float = -1.0,
) -> bool:
    """Decide whether a Whisper segment is trustworthy speech."""
    if no_speech_prob is not None and no_speech_prob > max_no_speech_prob:
        return False

    if avg_logprob is not None and avg_logprob < min_avg_logprob:
        return False

    return True


class FasterWhisperTranscriber:
    """Local speech recognition with faster-whisper (CPU, int8)."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        language: str = "ru",
        beam_size: int = 3,
        initial_prompt: str | None = _DEFAULT_PROMPT,
        model: Any = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if beam_size < 1:
            raise ValueError("beam_size must be at least 1.")

        self._model_size = model_size
        self._language = language
        self._beam_size = beam_size
        self._initial_prompt = initial_prompt
        self._model = model
        self._model_factory = model_factory

    @property
    def model_size(self) -> str:
        return self._model_size

    def update_names(self, names: Iterable[str]) -> None:
        """Bias recognition towards the assistant's names (used as the prompt)."""
        unique = [name for name in dict.fromkeys(names) if name]

        if unique:
            self._initial_prompt = "Разговор с ассистентом по имени " + " или ".join(unique) + "."

    def preload(self) -> None:
        """Load the model now so the first utterance is not delayed."""
        self._ensure_model()

    def transcribe(self, audio: bytes, sample_rate_hz: int) -> str:
        if not audio:
            return ""

        import numpy as np

        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self._ensure_model().transcribe(
            samples,
            language=self._language,
            beam_size=self._beam_size,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        parts = [
            segment.text.strip()
            for segment in segments
            if keep_segment(
                getattr(segment, "no_speech_prob", None),
                getattr(segment, "avg_logprob", None),
            )
        ]

        return " ".join(part for part in parts if part).strip()

    def _ensure_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self._model_size)
            else:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")

        return self._model
