"""Composition helpers for the OSA voice runtime."""

from __future__ import annotations

from osa.voice.audio import (
    MicrophoneInput,
    SpeakerOutput,
)
from osa.voice.platform import (
    AudioBackend,
    AudioPlatformUnavailableError,
    DesktopAudioBackendFactory,
    create_audio_backend_factory,
)
from osa.voice.runtime import (
    VoiceRuntime,
    VoiceRuntimeConfig,
)
from osa.voice.session import VoiceSession


class VoiceCompositionError(
    RuntimeError
):
    """Raised when the default voice runtime cannot be composed."""


def create_voice_runtime_from_backend(
    session: VoiceSession,
    backend: AudioBackend,
    *,
    config: VoiceRuntimeConfig | None = None,
) -> VoiceRuntime:
    """Create a VoiceRuntime from an already selected audio backend."""
    if session is None:
        raise ValueError(
            "session is required."
        )

    if backend is None:
        raise ValueError(
            "backend is required."
        )

    if not isinstance(
        backend,
        AudioBackend,
    ):
        raise TypeError(
            "backend must provide the OSA AudioBackend interface."
        )

    try:
        available = bool(
            backend.available()
        )
    except Exception as exc:
        raise VoiceCompositionError(
            "Failed to determine audio backend availability."
        ) from exc

    if not available:
        raise AudioPlatformUnavailableError(
            (
                "The selected audio backend is unavailable "
                f"for platform '{backend.platform.value}'."
            )
        )

    try:
        microphone = backend.microphone()
        speaker = backend.speaker()
    except Exception as exc:
        raise VoiceCompositionError(
            "Failed to create microphone and speaker."
        ) from exc

    if not isinstance(
        microphone,
        MicrophoneInput,
    ):
        raise VoiceCompositionError(
            "Audio backend returned an invalid microphone."
        )

    if not isinstance(
        speaker,
        SpeakerOutput,
    ):
        raise VoiceCompositionError(
            "Audio backend returned an invalid speaker."
        )

    return VoiceRuntime(
        session,
        microphone,
        speaker,
        config=config,
    )


def create_default_voice_runtime(
    session: VoiceSession,
    *,
    factory: DesktopAudioBackendFactory | None = None,
    config: VoiceRuntimeConfig | None = None,
) -> VoiceRuntime:
    """
    Create a voice runtime using the current platform's default backend.

    No platform-specific implementation is exposed to the caller.
    """
    backend_factory = (
        factory
        if factory is not None
        else create_audio_backend_factory()
    )

    if not isinstance(
        backend_factory,
        DesktopAudioBackendFactory,
    ):
        raise TypeError(
            "factory must be a DesktopAudioBackendFactory."
        )

    try:
        backend = backend_factory.current()
    except Exception as exc:
        raise VoiceCompositionError(
            "Failed to select the current platform audio backend."
        ) from exc

    return create_voice_runtime_from_backend(
        session,
        backend,
        config=config,
    )
