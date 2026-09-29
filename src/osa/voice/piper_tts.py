"""Local Piper text-to-speech adapter for OSA 0.7.3."""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from osa.voice.contracts import VoiceOutput
from osa.voice.tts import TextToSpeech, TextToSpeechError


SynthesisConfigFactory = Callable[..., Any]


@dataclass(frozen=True)
class PiperTTSConfig:
    """Configuration for a local Piper voice model."""

    model_path: str | None = None
    use_cuda: bool = False
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    speaker_id: int | None = None

    def __post_init__(self) -> None:
        if self.model_path is not None:
            normalized = self.model_path.strip()
            object.__setattr__(self, "model_path", normalized or None)

        if self.length_scale is not None and self.length_scale <= 0:
            raise ValueError("Piper length_scale must be positive.")

        if self.noise_scale is not None and self.noise_scale < 0:
            raise ValueError("Piper noise_scale cannot be negative.")

        if self.noise_w_scale is not None and self.noise_w_scale < 0:
            raise ValueError("Piper noise_w_scale cannot be negative.")

        if self.speaker_id is not None and self.speaker_id < 0:
            raise ValueError("Piper speaker_id cannot be negative.")


class PiperTextToSpeech(TextToSpeech):
    """Text-to-speech implementation backed by Piper."""

    def __init__(
        self,
        config: PiperTTSConfig | None = None,
        *,
        voice: Any | None = None,
        synthesis_config_factory: SynthesisConfigFactory | None = None,
    ) -> None:
        self._config = config or PiperTTSConfig()
        self._voice = voice
        self._voice_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._synthesis_config_factory = (
            synthesis_config_factory or self._create_synthesis_config
        )

    @property
    def config(self) -> PiperTTSConfig:
        """Return the immutable Piper configuration."""
        return self._config

    @property
    def model_loaded(self) -> bool:
        """Return whether the Piper voice model has been loaded."""
        return self._voice is not None

    def synthesize(self, output: VoiceOutput) -> bytes:
        """Synthesize one response into WAV bytes."""
        self._validate_output(output)
        self._stop_requested.clear()

        try:
            voice = self._get_voice()

            synthesis_config = self._synthesis_config_factory(
                speaker_id=self._config.speaker_id,
                length_scale=self._config.length_scale,
                noise_scale=self._config.noise_scale,
                noise_w_scale=self._config.noise_w_scale,
            )

            chunks = voice.synthesize(
                output.text,
                syn_config=synthesis_config,
            )

            try:
                first_chunk = next(chunks)
            except StopIteration as exc:
                raise TextToSpeechError(
                    "Piper produced no audio for the supplied text."
                ) from exc

            if self._stop_requested.is_set():
                raise TextToSpeechError(
                    "Piper synthesis was stopped."
                )

            wav_buffer = io.BytesIO()

            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setframerate(first_chunk.sample_rate)
                wav_file.setsampwidth(first_chunk.sample_width)
                wav_file.setnchannels(first_chunk.sample_channels)
                wav_file.writeframes(first_chunk.audio_int16_bytes)

                for chunk in chunks:
                    if self._stop_requested.is_set():
                        raise TextToSpeechError(
                            "Piper synthesis was stopped."
                        )

                    wav_file.writeframes(chunk.audio_int16_bytes)

            return wav_buffer.getvalue()

        except TextToSpeechError:
            raise
        except ImportError as exc:
            raise TextToSpeechError(
                "piper-tts is not installed. "
                "Install the OSA voice extra before using this adapter."
            ) from exc
        except Exception as exc:
            raise TextToSpeechError(
                f"Piper synthesis failed: {exc}"
            ) from exc

    def stop(self) -> None:
        """Request cancellation of the current or next synthesis loop."""
        self._stop_requested.set()

    def _get_voice(self) -> Any:
        if self._voice is not None:
            return self._voice

        model_path = self._config.model_path
        if model_path is None:
            raise TextToSpeechError(
                "Piper model_path is required when no injected voice is provided."
            )

        path = Path(model_path)
        if not path.is_file():
            raise TextToSpeechError(
                f"Piper voice model was not found: {path}"
            )

        with self._voice_lock:
            if self._voice is not None:
                return self._voice

            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise TextToSpeechError(
                    "piper-tts is not installed. "
                    "Install the OSA voice extra before using this adapter."
                ) from exc

            try:
                self._voice = PiperVoice.load(
                    path,
                    use_cuda=self._config.use_cuda,
                )
            except Exception as exc:
                raise TextToSpeechError(
                    f"Could not load Piper voice model '{path}': {exc}"
                ) from exc

            return self._voice

    def _create_synthesis_config(self, **kwargs: Any) -> Any:
        """Create the real Piper SynthesisConfig lazily."""
        try:
            from piper import SynthesisConfig
        except ImportError as exc:
            raise TextToSpeechError(
                "piper-tts is not installed. "
                "Install the OSA voice extra before using this adapter."
            ) from exc

        return SynthesisConfig(**kwargs)

    @staticmethod
    def _validate_output(output: VoiceOutput) -> None:
        if not isinstance(output, VoiceOutput):
            raise TextToSpeechError(
                "Piper expects a VoiceOutput instance."
            )

        if not output.text.strip():
            raise TextToSpeechError(
                "Piper cannot synthesize empty text."
            )


def create_piper_tts(
    config: PiperTTSConfig | None = None,
    *,
    voice: Any | None = None,
    synthesis_config_factory: SynthesisConfigFactory | None = None,
) -> PiperTextToSpeech:
    """Create a local Piper TTS adapter."""
    return PiperTextToSpeech(
        config=config,
        voice=voice,
        synthesis_config_factory=synthesis_config_factory,
    )
