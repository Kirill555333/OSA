"""Local faster-whisper speech-to-text adapter for OSA 0.7.2."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from osa.voice.contracts import VoiceInput, VoiceTranscript
from osa.voice.stt import SpeechToText, SpeechToTextError


AudioConverter = Callable[[bytes], Any]


@dataclass(frozen=True)
class FasterWhisperConfig:
    """Configuration for the lazy-loaded faster-whisper model."""

    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Faster-whisper model cannot be empty.")
        if not self.device.strip():
            raise ValueError("Faster-whisper device cannot be empty.")
        if not self.compute_type.strip():
            raise ValueError("Faster-whisper compute_type cannot be empty.")
        if self.beam_size < 1:
            raise ValueError("Faster-whisper beam_size must be positive.")

        for field_name in ("model", "device", "compute_type", "language"):
            value = getattr(self, field_name)
            if value is None:
                continue
            object.__setattr__(self, field_name, value.strip() or None)


class FasterWhisperSpeechToText(SpeechToText):
    """Speech-to-text implementation backed by faster-whisper."""

    REQUIRED_SAMPLE_RATE_HZ = 16_000
    SUPPORTED_ENCODING = "pcm_s16le"

    def __init__(
        self,
        config: FasterWhisperConfig | None = None,
        *,
        model: Any | None = None,
        audio_converter: AudioConverter | None = None,
    ) -> None:
        self._config = config or FasterWhisperConfig()
        self._model = model
        self._model_lock = Lock()
        self._audio_converter = (
            audio_converter or self._pcm16_to_float32
        )

    @property
    def config(self) -> FasterWhisperConfig:
        """Return the immutable adapter configuration."""
        return self._config

    @property
    def model_loaded(self) -> bool:
        """Return whether the Whisper model has already been constructed."""
        return self._model is not None

    def transcribe(self, audio: VoiceInput) -> VoiceTranscript:
        """Transcribe one normalized PCM16 mono utterance."""
        self._validate_audio(audio)

        try:
            samples = self._audio_converter(audio.audio)
            model = self._get_model()

            kwargs: dict[str, Any] = {
                "beam_size": self._config.beam_size,
                "vad_filter": self._config.vad_filter,
                "condition_on_previous_text": (
                    self._config.condition_on_previous_text
                ),
            }

            if self._config.language is not None:
                kwargs["language"] = self._config.language

            segments, info = model.transcribe(samples, **kwargs)

            text = "".join(
                segment.text
                for segment in segments
                if getattr(segment, "text", None)
            ).strip()

            detected_language = getattr(info, "language", None)

            return VoiceTranscript(
                text=text,
                language=detected_language or self._config.language,
                confidence=None,
                is_final=True,
            )
        except SpeechToTextError:
            raise
        except Exception as exc:
            raise SpeechToTextError(
                f"Faster-whisper transcription failed: {exc}"
            ) from exc

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SpeechToTextError(
                    "faster-whisper is not installed. "
                    "Install the OSA voice extra before using this adapter."
                ) from exc

            try:
                self._model = WhisperModel(
                    self._config.model,
                    device=self._config.device,
                    compute_type=self._config.compute_type,
                )
            except Exception as exc:
                raise SpeechToTextError(
                    f"Could not load faster-whisper model "
                    f"'{self._config.model}': {exc}"
                ) from exc

            return self._model

    def _validate_audio(self, audio: VoiceInput) -> None:
        if not isinstance(audio, VoiceInput):
            raise SpeechToTextError(
                "Faster-whisper expects a VoiceInput instance."
            )

        if audio.encoding.lower() != self.SUPPORTED_ENCODING:
            raise SpeechToTextError(
                "Faster-whisper adapter expects pcm_s16le audio."
            )

        if audio.channels != 1:
            raise SpeechToTextError(
                "Faster-whisper adapter expects mono audio. "
                "Channel conversion belongs to the audio normalization stage."
            )

        if audio.sample_rate_hz != self.REQUIRED_SAMPLE_RATE_HZ:
            raise SpeechToTextError(
                "Faster-whisper adapter expects 16000 Hz audio. "
                "Resampling belongs to the audio normalization stage."
            )

        if not audio.audio:
            raise SpeechToTextError(
                "Faster-whisper cannot transcribe empty audio."
            )

        if len(audio.audio) % 2:
            raise SpeechToTextError(
                "PCM16 audio must contain an even number of bytes."
            )

    @staticmethod
    def _pcm16_to_float32(audio: bytes) -> Any:
        """Convert little-endian PCM16 bytes to a float32 NumPy waveform."""
        samples = array("h")
        samples.frombytes(audio)

        if _needs_byteswap():
            samples.byteswap()

        try:
            import numpy as np
        except ImportError as exc:
            raise SpeechToTextError(
                "NumPy is required by the faster-whisper adapter. "
                "Install the OSA voice extra before using this adapter."
            ) from exc

        return np.asarray(samples, dtype=np.float32) / 32768.0


def create_faster_whisper_stt(
    config: FasterWhisperConfig | None = None,
    *,
    model: Any | None = None,
    audio_converter: AudioConverter | None = None,
) -> FasterWhisperSpeechToText:
    """Create a faster-whisper STT adapter."""
    return FasterWhisperSpeechToText(
        config=config,
        model=model,
        audio_converter=audio_converter,
    )


def _needs_byteswap() -> bool:
    """Return whether array native byte order differs from little endian."""
    import sys

    return sys.byteorder != "little"
