"""Utterance segmentation for always-on listening (OSA 1.5.2.1).

Pure logic over PCM16 mono frames: no audio drivers, no threads, no clocks and no
third-party packages, so every behavior is testable with synthetic audio.

The adaptive detector structurally satisfies ``osa.voice.vad.VoiceActivityDetector``.
"""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
import math
import sys
from typing import Protocol


def pcm16_rms(frame: bytes) -> float:
    """Return the RMS level of little-endian PCM16 audio."""
    if not frame:
        return 0.0

    if len(frame) % 2 != 0:
        raise ValueError("PCM16 audio must contain an even number of bytes.")

    samples = array("h")
    samples.frombytes(frame)

    if sys.byteorder == "big":
        samples.byteswap()

    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class AdaptiveEnergyVad:
    """Energy detector whose threshold follows the ambient noise floor.

    The first ``calibration_frames`` frames are treated as ambient noise. Afterwards
    a frame is speech when its RMS reaches ``max(min_threshold, floor * ratio)``. The
    floor adapts slowly on non-speech frames. If "speech" lasts implausibly long
    (a fan, a hum), the level is adopted as the new noise floor.
    """

    def __init__(
        self,
        *,
        min_threshold: float = 300.0,
        ratio: float = 3.0,
        calibration_frames: int = 10,
        smoothing: float = 0.05,
        max_floor: float = 3000.0,
        stuck_frames: int = 700,
    ) -> None:
        if min_threshold <= 0 or ratio <= 1 or max_floor <= 0:
            raise ValueError("min_threshold and max_floor must be > 0, ratio must be > 1.")

        if calibration_frames < 1 or stuck_frames < 1:
            raise ValueError("calibration_frames and stuck_frames must be >= 1.")

        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1].")

        self._min_threshold = float(min_threshold)
        self._ratio = float(ratio)
        self._calibration_frames = calibration_frames
        self._smoothing = float(smoothing)
        self._max_floor = float(max_floor)
        self._stuck_frames = stuck_frames
        self._floor = 0.0
        self._calibrated = 0
        self._speech_run = 0

    @property
    def threshold(self) -> float:
        """Return the current speech threshold."""
        return max(self._min_threshold, self._floor * self._ratio)

    def detect(self, audio: bytes, *, sample_rate_hz: int = 16000, channels: int = 1) -> bool:
        if sample_rate_hz <= 0 or channels <= 0:
            raise ValueError("sample_rate_hz and channels must be greater than zero.")

        rms = pcm16_rms(audio)

        if self._calibrated < self._calibration_frames:
            self._calibrated += 1
            self._floor += (min(rms, self._max_floor) - self._floor) / self._calibrated
            return False

        if rms < self.threshold:
            self._speech_run = 0
            self._floor += self._smoothing * (min(rms, self._max_floor) - self._floor)
            return False

        self._speech_run += 1

        if self._speech_run > self._stuck_frames:
            self._floor = min(rms, self._max_floor)
            self._speech_run = 0
            return False

        return True

    def reset(self) -> None:
        self._floor = 0.0
        self._calibrated = 0
        self._speech_run = 0


class FrameVad(Protocol):
    """Minimal detector interface used by the segmenter."""

    def detect(self, audio: bytes, *, sample_rate_hz: int, channels: int = 1) -> bool: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    """Timing parameters, all in milliseconds."""

    sample_rate_hz: int = 16000
    frame_ms: int = 30
    preroll_ms: int = 300
    start_speech_ms: int = 90
    end_silence_ms: int = 900
    tail_ms: int = 210
    min_speech_ms: int = 400
    max_utterance_ms: int = 15000

    def __post_init__(self) -> None:
        values = (
            self.sample_rate_hz,
            self.frame_ms,
            self.preroll_ms,
            self.start_speech_ms,
            self.end_silence_ms,
            self.tail_ms,
            self.min_speech_ms,
            self.max_utterance_ms,
        )

        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("All segmenter settings must be integers.")

        if any(value <= 0 for value in values):
            raise ValueError("All segmenter settings must be greater than zero.")

        if (self.sample_rate_hz * self.frame_ms) % 1000 != 0:
            raise ValueError("frame_ms must correspond to a whole number of samples.")

        if self.tail_ms > self.end_silence_ms:
            raise ValueError("tail_ms cannot exceed end_silence_ms.")

        if self.min_speech_ms > self.max_utterance_ms:
            raise ValueError("min_speech_ms cannot exceed max_utterance_ms.")

    @property
    def frame_samples(self) -> int:
        return self.sample_rate_hz * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2

    def frames(self, milliseconds: int) -> int:
        """Return the number of whole frames covering the given duration."""
        return max(1, -(-milliseconds // self.frame_ms))


class UtteranceSegmenter:
    """Turn a stream of fixed-size frames into complete utterances."""

    def __init__(self, vad: FrameVad, config: SegmenterConfig | None = None) -> None:
        self._vad = vad
        self._config = config or SegmenterConfig()
        cfg = self._config
        self._start_frames = cfg.frames(cfg.start_speech_ms)
        self._end_silence_frames = cfg.frames(cfg.end_silence_ms)
        self._tail_frames = cfg.frames(cfg.tail_ms)
        self._min_speech_frames = cfg.frames(cfg.min_speech_ms)
        self._max_frames = cfg.frames(cfg.max_utterance_ms)
        self._preroll: deque[bytes] = deque(maxlen=cfg.frames(cfg.preroll_ms))
        self._candidate: list[bytes] = []
        self._frames: list[bytes] = []
        self._active = False
        self._speech_frames = 0
        self._silence_run = 0
        self._last_speech_index = -1

    @property
    def config(self) -> SegmenterConfig:
        return self._config

    @property
    def active(self) -> bool:
        """Return True while an utterance is being collected."""
        return self._active

    def reset(self, *, reset_detector: bool = False) -> None:
        """Drop any partial audio; optionally restart noise calibration."""
        self._preroll.clear()
        self._candidate.clear()
        self._frames = []
        self._active = False
        self._speech_frames = 0
        self._silence_run = 0
        self._last_speech_index = -1

        if reset_detector:
            self._vad.reset()

    def feed(self, frame: bytes) -> bytes | None:
        """Consume one frame; return PCM16 audio when an utterance has just ended."""
        if len(frame) != self._config.frame_bytes:
            raise ValueError(
                f"Expected a frame of {self._config.frame_bytes} bytes, got {len(frame)}."
            )

        is_speech = self._vad.detect(
            frame,
            sample_rate_hz=self._config.sample_rate_hz,
            channels=1,
        )

        if not self._active:
            self._feed_idle(frame, is_speech)
            return None

        self._frames.append(frame)

        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
            self._last_speech_index = len(self._frames) - 1
        else:
            self._silence_run += 1

        if self._silence_run >= self._end_silence_frames or len(self._frames) >= self._max_frames:
            return self._finish()

        return None

    def flush(self) -> bytes | None:
        """Finish a pending utterance (for example at shutdown)."""
        if not self._active:
            self._candidate.clear()
            return None

        return self._finish()

    def _feed_idle(self, frame: bytes, is_speech: bool) -> None:
        if is_speech:
            self._candidate.append(frame)

            if len(self._candidate) >= self._start_frames:
                self._begin()

            return

        if self._candidate:
            self._preroll.extend(self._candidate)
            self._candidate.clear()

        self._preroll.append(frame)

    def _begin(self) -> None:
        self._frames = [*self._preroll, *self._candidate]
        self._speech_frames = len(self._candidate)
        self._last_speech_index = len(self._frames) - 1
        self._silence_run = 0
        self._preroll.clear()
        self._candidate.clear()
        self._active = True

    def _finish(self) -> bytes | None:
        keep_until = min(len(self._frames), self._last_speech_index + 1 + self._tail_frames)
        audio = b"".join(self._frames[:keep_until])
        enough_speech = self._speech_frames >= self._min_speech_frames
        self.reset()

        return audio if enough_speech else None
