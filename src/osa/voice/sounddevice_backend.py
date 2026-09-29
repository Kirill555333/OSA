"""Cross-platform audio backend using python-sounddevice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osa.voice.audio import (
    AudioInputError,
    AudioOutputError,
    MicrophoneInput,
    SpeakerOutput,
)
from osa.voice.contracts import VoiceInput
from osa.voice.platform import (
    AudioBackend,
    AudioPlatform,
)


class SoundDeviceAudioError(
    RuntimeError
):
    """Raised when the sounddevice backend cannot operate."""


@dataclass(frozen=True)
class SoundDeviceAudioConfig:
    """Configuration for the sounddevice audio backend."""

    sample_rate_hz: int = 16_000
    channels: int = 1
    frames_per_read: int = 1_600
    dtype: str = "int16"
    latency: str = "low"
    input_device: int | str | None = None
    output_device: int | str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(
                self.sample_rate_hz,
                int,
            )
        ):
            raise ValueError(
                "sample_rate_hz must be an integer."
            )

        if self.sample_rate_hz <= 0:
            raise ValueError(
                "sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(self.channels, bool)
            or not isinstance(
                self.channels,
                int,
            )
        ):
            raise ValueError(
                "channels must be an integer."
            )

        if self.channels <= 0:
            raise ValueError(
                "channels must be greater than zero."
            )

        if (
            isinstance(self.frames_per_read, bool)
            or not isinstance(
                self.frames_per_read,
                int,
            )
        ):
            raise ValueError(
                "frames_per_read must be an integer."
            )

        if self.frames_per_read <= 0:
            raise ValueError(
                "frames_per_read must be greater than zero."
            )

        if not isinstance(
            self.dtype,
            str,
        ):
            raise ValueError(
                "dtype must be a string."
            )

        if not self.dtype.strip():
            raise ValueError(
                "dtype cannot be empty."
            )

        if self.dtype.strip().casefold() != "int16":
            raise ValueError(
                "dtype must be 'int16' for the OSA PCM audio contract."
            )

        if not isinstance(
            self.latency,
            str,
        ):
            raise ValueError(
                "latency must be a string."
            )

        if not self.latency.strip():
            raise ValueError(
                "latency cannot be empty."
            )

        object.__setattr__(
            self,
            "dtype",
            self.dtype.strip().lower(),
        )
        object.__setattr__(
            self,
            "latency",
            self.latency.strip().lower(),
        )


def _load_sounddevice(
    module: Any | None = None,
) -> Any:
    """Return an injected or lazily imported sounddevice module."""
    if module is not None:
        return module

    try:
        import sounddevice
    except ImportError as exc:
        raise SoundDeviceAudioError(
            "sounddevice is not installed. "
            "Install it with "
            "'python -m pip install sounddevice==0.5.6'."
        ) from exc

    return sounddevice


class SoundDeviceMicrophone(
    MicrophoneInput,
):
    """Microphone input backed by sounddevice.RawInputStream."""

    def __init__(
        self,
        *,
        config: SoundDeviceAudioConfig | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else SoundDeviceAudioConfig()
        )
        self._sounddevice_module = sounddevice_module
        self._stream: Any | None = None
        self._started = False

    @property
    def config(self) -> SoundDeviceAudioConfig:
        return self._config

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise AudioInputError(
                "Microphone is already started."
            )

        sounddevice = _load_sounddevice(
            self._sounddevice_module
        )

        try:
            stream = sounddevice.RawInputStream(
                samplerate=self._config.sample_rate_hz,
                blocksize=self._config.frames_per_read,
                device=self._config.input_device,
                channels=self._config.channels,
                dtype=self._config.dtype,
                latency=self._config.latency,
            )

            stream.start()

        except Exception as exc:
            raise AudioInputError(
                f"Failed to start microphone: {exc}"
            ) from exc

        self._stream = stream
        self._started = True

    def read(self) -> VoiceInput:
        if not self._started or self._stream is None:
            raise AudioInputError(
                "Microphone is not started."
            )

        try:
            data, overflowed = self._stream.read(
                self._config.frames_per_read
            )

        except Exception as exc:
            raise AudioInputError(
                f"Failed to read microphone audio: {exc}"
            ) from exc

        if overflowed:
            raise AudioInputError(
                "Microphone input overflowed."
            )

        payload = bytes(data)

        if not payload:
            raise AudioInputError(
                "Microphone returned empty audio."
            )

        return VoiceInput(
            audio=payload,
            sample_rate_hz=self._config.sample_rate_hz,
            channels=self._config.channels,
            encoding="pcm_s16le",
        )

    def stop(self) -> None:
        if self._stream is None:
            return

        stream = self._stream
        self._stream = None
        was_started = self._started
        self._started = False

        try:
            if was_started:
                stream.stop()

            stream.close()

        except Exception as exc:
            raise AudioInputError(
                f"Failed to stop microphone: {exc}"
            ) from exc


class SoundDeviceSpeaker(
    SpeakerOutput,
):
    """Speaker output backed by sounddevice.RawOutputStream."""

    def __init__(
        self,
        *,
        config: SoundDeviceAudioConfig | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else SoundDeviceAudioConfig()
        )
        self._sounddevice_module = sounddevice_module
        self._stream: Any | None = None

    @property
    def config(self) -> SoundDeviceAudioConfig:
        return self._config

    @property
    def playing(self) -> bool:
        return self._stream is not None

    def play(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> None:
        if not isinstance(
            audio,
            bytes,
        ):
            raise AudioOutputError(
                "audio must be bytes."
            )

        if not audio:
            raise AudioOutputError(
                "audio cannot be empty."
            )

        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(
                sample_rate_hz,
                int,
            )
        ):
            raise AudioOutputError(
                "sample_rate_hz must be an integer."
            )

        if sample_rate_hz <= 0:
            raise AudioOutputError(
                "sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(channels, bool)
            or not isinstance(
                channels,
                int,
            )
        ):
            raise AudioOutputError(
                "channels must be an integer."
            )

        if channels <= 0:
            raise AudioOutputError(
                "channels must be greater than zero."
            )

        if self._stream is not None:
            raise AudioOutputError(
                "Speaker is already playing audio."
            )

        sounddevice = _load_sounddevice(
            self._sounddevice_module
        )

        try:
            stream = sounddevice.RawOutputStream(
                samplerate=sample_rate_hz,
                blocksize=0,
                device=self._config.output_device,
                channels=channels,
                dtype=self._config.dtype,
                latency=self._config.latency,
            )

            stream.start()
            self._stream = stream

            stream.write(
                audio
            )

        except Exception as exc:
            if self._stream is stream if "stream" in locals() else False:
                self._stream = None

            try:
                stream.stop()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

            raise AudioOutputError(
                f"Failed to play audio: {exc}"
            ) from exc

        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            raise AudioOutputError(
                f"Failed to finish audio playback: {exc}"
            ) from exc
        finally:
            self._stream = None

    def stop(self) -> None:
        stream = self._stream

        if stream is None:
            return

        self._stream = None

        try:
            try:
                stream.abort()
            except AttributeError:
                stream.stop()
        except Exception as exc:
            raise AudioOutputError(
                f"Failed to stop speaker playback: {exc}"
            ) from exc
        finally:
            try:
                stream.close()
            except Exception:
                pass


class SoundDeviceAudioBackend(
    AudioBackend,
):
    """Cross-platform OSA audio backend using sounddevice."""

    def __init__(
        self,
        *,
        config: SoundDeviceAudioConfig | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else SoundDeviceAudioConfig()
        )
        self._sounddevice_module = sounddevice_module

    @property
    def platform(self) -> AudioPlatform:
        return AudioPlatform.current()

    @property
    def config(self) -> SoundDeviceAudioConfig:
        return self._config

    def available(self) -> bool:
        try:
            sounddevice = _load_sounddevice(
                self._sounddevice_module
            )

            devices = sounddevice.query_devices()

        except Exception:
            return False

        try:
            return any(
                (
                    int(
                        device.get(
                            "max_input_channels",
                            0,
                        )
                    ) > 0
                    or int(
                        device.get(
                            "max_output_channels",
                            0,
                        )
                    ) > 0
                )
                for device in devices
            )

        except Exception:
            return False

    def microphone(self) -> SoundDeviceMicrophone:
        return SoundDeviceMicrophone(
            config=self._config,
            sounddevice_module=self._sounddevice_module,
        )

    def speaker(self) -> SoundDeviceSpeaker:
        return SoundDeviceSpeaker(
            config=self._config,
            sounddevice_module=self._sounddevice_module,
        )

    def query_devices(self) -> Any:
        """Return the provider's device list for diagnostics."""
        sounddevice = _load_sounddevice(
            self._sounddevice_module
        )

        try:
            return sounddevice.query_devices()
        except Exception as exc:
            raise SoundDeviceAudioError(
                f"Failed to query audio devices: {exc}"
            ) from exc
