"""Platform-neutral audio backend selection for OSA 0.6.x."""

from __future__ import annotations

import sys
from enum import Enum
from typing import Protocol, runtime_checkable

from osa.voice.audio import (
    MicrophoneInput,
    SpeakerOutput,
)


class AudioPlatformError(
    RuntimeError
):
    """Base error for platform audio integration."""


class AudioPlatformUnavailableError(
    AudioPlatformError
):
    """Raised when no usable audio backend is available."""


class AudioPlatform(str, Enum):
    """Supported operating-system families for OSA voice audio."""

    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    UNKNOWN = "unknown"

    @classmethod
    def current(cls) -> AudioPlatform:
        """Return the normalized platform for the current interpreter."""
        if sys.platform == "darwin":
            return cls.MACOS

        if sys.platform == "win32":
            return cls.WINDOWS

        if sys.platform.startswith("linux"):
            return cls.LINUX

        return cls.UNKNOWN


@runtime_checkable
class AudioBackend(Protocol):
    """Provider-neutral platform audio backend."""

    @property
    def platform(self) -> AudioPlatform:
        ...

    def available(self) -> bool:
        ...

    def microphone(self) -> MicrophoneInput:
        ...

    def speaker(self) -> SpeakerOutput:
        ...


class UnavailableAudioBackend:
    """
    Explicit unavailable backend.

    It is used when no concrete provider is configured or when an optional
    provider is not installed.
    """

    def __init__(
        self,
        platform: AudioPlatform,
        *,
        reason: str = "No audio backend is configured.",
    ) -> None:
        self._platform = platform
        self._reason = reason

    @property
    def platform(self) -> AudioPlatform:
        return self._platform

    @property
    def reason(self) -> str:
        return self._reason

    def available(self) -> bool:
        return False

    def microphone(self) -> MicrophoneInput:
        raise AudioPlatformUnavailableError(
            self._reason
        )

    def speaker(self) -> SpeakerOutput:
        raise AudioPlatformUnavailableError(
            self._reason
        )


class DesktopAudioBackendFactory:
    """
    Select the platform audio backend.

    Explicitly injected backends always take precedence. When no backend is
    supplied for a platform, the default sounddevice backend is attempted
    lazily for macOS, Windows, and Linux.
    """

    def __init__(
        self,
        *,
        macos: AudioBackend | None = None,
        windows: AudioBackend | None = None,
        linux: AudioBackend | None = None,
        unknown: AudioBackend | None = None,
        enable_default_provider: bool = True,
    ) -> None:
        self._backends = {
            AudioPlatform.MACOS: (
                macos
                if macos is not None
                else self._default_backend(
                    AudioPlatform.MACOS,
                    enabled=enable_default_provider,
                )
            ),
            AudioPlatform.WINDOWS: (
                windows
                if windows is not None
                else self._default_backend(
                    AudioPlatform.WINDOWS,
                    enabled=enable_default_provider,
                )
            ),
            AudioPlatform.LINUX: (
                linux
                if linux is not None
                else self._default_backend(
                    AudioPlatform.LINUX,
                    enabled=enable_default_provider,
                )
            ),
            AudioPlatform.UNKNOWN: (
                unknown
                if unknown is not None
                else UnavailableAudioBackend(
                    AudioPlatform.UNKNOWN,
                    reason=(
                        "No default audio backend is defined "
                        "for this platform."
                    ),
                )
            ),
        }

    @staticmethod
    def _default_backend(
        platform: AudioPlatform,
        *,
        enabled: bool,
    ) -> AudioBackend:
        if not enabled:
            return UnavailableAudioBackend(
                platform,
                reason=(
                    "Default audio provider is disabled."
                ),
            )

        try:
            from osa.voice.sounddevice_backend import (
                SoundDeviceAudioBackend,
            )

            backend = SoundDeviceAudioBackend()

        except Exception as exc:
            return UnavailableAudioBackend(
                platform,
                reason=(
                    "sounddevice audio backend is unavailable: "
                    f"{exc}"
                ),
            )

        if not isinstance(
            backend,
            AudioBackend,
        ):
            return UnavailableAudioBackend(
                platform,
                reason=(
                    "sounddevice backend does not satisfy "
                    "the OSA audio backend contract."
                ),
            )

        return backend

    @property
    def backends(
        self,
    ) -> dict[AudioPlatform, AudioBackend]:
        """
        Return a shallow copy of configured backends.

        The factory's internal mapping cannot be mutated through this result.
        """
        return dict(self._backends)

    def current(self) -> AudioBackend:
        """Return the backend for the current operating system."""
        return self.for_platform(
            AudioPlatform.current()
        )

    def for_platform(
        self,
        platform: AudioPlatform,
    ) -> AudioBackend:
        """Return the configured backend for a platform."""
        if not isinstance(
            platform,
            AudioPlatform,
        ):
            raise TypeError(
                "platform must be an AudioPlatform."
            )

        backend = self._backends[platform]

        if not isinstance(
            backend,
            AudioBackend,
        ):
            raise TypeError(
                "Configured audio backend does not satisfy AudioBackend."
            )

        return backend

    def available(
        self,
        platform: AudioPlatform | None = None,
    ) -> bool:
        """Return whether a selected platform backend is available."""
        backend = (
            self.current()
            if platform is None
            else self.for_platform(platform)
        )

        try:
            return bool(
                backend.available()
            )
        except Exception:
            return False


def create_audio_backend_factory(
    *,
    macos: AudioBackend | None = None,
    windows: AudioBackend | None = None,
    linux: AudioBackend | None = None,
    unknown: AudioBackend | None = None,
    enable_default_provider: bool = True,
) -> DesktopAudioBackendFactory:
    """Create a platform-aware audio backend factory."""
    return DesktopAudioBackendFactory(
        macos=macos,
        windows=windows,
        linux=linux,
        unknown=unknown,
        enable_default_provider=enable_default_provider,
    )


def create_default_audio_backend() -> AudioBackend:
    """
    Create the default backend for the current platform.

    The returned object may be an UnavailableAudioBackend when the optional
    sounddevice provider is not installed or no provider is defined.
    """
    return create_audio_backend_factory().current()
