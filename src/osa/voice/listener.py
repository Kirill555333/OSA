"""Speech-to-Text (STT) listener module using Whisper medium for OSA.
Explicit stream lifecycle management (start/stop/close) to prevent macOS CoreAudio deadlocks.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np
import sounddevice as sd

INITIAL_PROMPT = "Разговор с персональным AI-ассистентом OSA (Оса, Джарвис). Команды: открой тг, напиши другу, закрой, запомни, звук, wolfcut, булат."


class OSAVoiceListener:
    """Captures microphone audio and transcribes it using local Whisper medium."""

    def __init__(self, model_size: str = "medium", sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.model_size = model_size
        self._model: Any = None

    def preload_model(self) -> None:
        """Pre-load Whisper model into memory at startup to avoid runtime freezes."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

    def _ensure_model(self) -> Any:
        if self._model is None:
            self.preload_model()
        return self._model

    def record_until_enter(self) -> np.ndarray:
        """Record audio continuously with clean stream lifecycle (no deadlocks)."""
        buffer: list[np.ndarray] = []

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            buffer.append(indata.copy())

        # Explicit stream lifecycle: opened and closed per phrase
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        stream.start()

        try:
            # Wait for user to finish speaking
            input()
        except (KeyboardInterrupt, SystemExit):
            stream.stop()
            stream.close()
            raise
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        if not buffer:
            return np.array([], dtype=np.float32)

        raw_data = np.concatenate(buffer, axis=0).flatten()

        # Clean NaN/Inf and clip to safe range [-1.0, 1.0] to prevent STFT overflow
        raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=1.0, neginf=-1.0)
        max_val = np.max(np.abs(raw_data))
        if max_val > 0.001:
            raw_data = raw_data / max(max_val, 1.0)
        raw_data = np.clip(raw_data, -1.0, 1.0)

        return raw_data.astype(np.float32)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to Russian text using preloaded Whisper model."""
        if audio_data.size == 0 or len(audio_data) < int(self.sample_rate * 0.3):
            return ""

        model = self._ensure_model()
        segments, _ = model.transcribe(
            audio_data,
            language="ru",
            beam_size=3,
            initial_prompt=INITIAL_PROMPT,
        )
        return " ".join([seg.text for seg in segments]).strip()

    def listen_phrase(self) -> str:
        """Record audio until Enter and transcribe."""
        audio = self.record_until_enter()
        print(" [⏳ Расшифровываю речь через Whisper medium...]", flush=True)
        return self.transcribe(audio)
