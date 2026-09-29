from __future__ import annotations

import sys

import pytest

from osa.voice.audio import (
    FakeMicrophone,
    FakeSpeaker,
)
from osa.voice.platform import (
    AudioBackend,
    AudioPlatform,
    DesktopAudioBackendFactory,
    UnavailableAudioBackend,
    create_audio_backend_factory,
    create_default_audio_backend,
)


class FakeBackend:
    def __init__(
        self,
        platform: AudioPlatform,
        *,
        available: bool = True,
    ) -> None:
        self._platform = platform
        self._available = available
        self._microphone = FakeMicrophone()
        self._speaker = FakeSpeaker()

    @property
    def platform(self) -> AudioPlatform:
        return self._platform

    def available(self) -> bool:
        return self._available

    def microphone(self):
        return self._microphone

    def speaker(self):
        return self._speaker


def test_explicit_backend_has_priority_over_default_provider():
    backend = FakeBackend(
        AudioPlatform.MACOS,
    )

    factory = DesktopAudioBackendFactory(
        macos=backend,
    )

    assert factory.for_platform(
        AudioPlatform.MACOS
    ) is backend


def test_default_provider_can_be_disabled():
    factory = DesktopAudioBackendFactory(
        enable_default_provider=False,
    )

    for platform in (
        AudioPlatform.MACOS,
        AudioPlatform.WINDOWS,
        AudioPlatform.LINUX,
    ):
        backend = factory.for_platform(
            platform
        )

        assert isinstance(
            backend,
            UnavailableAudioBackend,
        )
        assert backend.available() is False


def test_factory_exposes_copy_of_backend_mapping():
    backend = FakeBackend(
        AudioPlatform.MACOS,
    )

    factory = DesktopAudioBackendFactory(
        macos=backend,
    )

    mapping = factory.backends

    assert mapping[
        AudioPlatform.MACOS
    ] is backend

    mapping.pop(
        AudioPlatform.MACOS
    )

    assert factory.for_platform(
        AudioPlatform.MACOS
    ) is backend


def test_default_factory_current_backend_has_protocol_shape():
    backend = create_default_audio_backend()

    assert isinstance(
        backend,
        AudioBackend,
    )


def test_default_factory_is_safe_when_provider_is_disabled():
    factory = create_audio_backend_factory(
        enable_default_provider=False,
    )

    backend = factory.current()

    assert isinstance(
        backend,
        UnavailableAudioBackend,
    )


def test_default_factory_accepts_all_supported_platform_injections():
    macos = FakeBackend(
        AudioPlatform.MACOS,
    )
    windows = FakeBackend(
        AudioPlatform.WINDOWS,
    )
    linux = FakeBackend(
        AudioPlatform.LINUX,
    )
    unknown = FakeBackend(
        AudioPlatform.UNKNOWN,
    )

    factory = create_audio_backend_factory(
        macos=macos,
        windows=windows,
        linux=linux,
        unknown=unknown,
    )

    assert factory.for_platform(
        AudioPlatform.MACOS
    ) is macos

    assert factory.for_platform(
        AudioPlatform.WINDOWS
    ) is windows

    assert factory.for_platform(
        AudioPlatform.LINUX
    ) is linux

    assert factory.for_platform(
        AudioPlatform.UNKNOWN
    ) is unknown


def test_current_platform_is_one_of_known_values():
    assert AudioPlatform.current() in {
        AudioPlatform.MACOS,
        AudioPlatform.WINDOWS,
        AudioPlatform.LINUX,
        AudioPlatform.UNKNOWN,
    }


def test_platform_specific_environment_does_not_change_factory_contract():
    original = sys.platform

    try:
        sys.platform = "darwin"

        factory = DesktopAudioBackendFactory(
            enable_default_provider=False,
        )

        assert factory.current().platform == (
            AudioPlatform.MACOS
        )

        sys.platform = "win32"

        assert factory.current().platform == (
            AudioPlatform.WINDOWS
        )

        sys.platform = "linux"

        assert factory.current().platform == (
            AudioPlatform.LINUX
        )

    finally:
        sys.platform = original


def test_unavailable_backend_remains_fail_closed():
    backend = UnavailableAudioBackend(
        AudioPlatform.WINDOWS,
        reason="provider unavailable",
    )

    assert backend.available() is False

    with pytest.raises(
        RuntimeError,
        match="provider unavailable",
    ):
        backend.microphone()

    with pytest.raises(
        RuntimeError,
        match="provider unavailable",
    ):
        backend.speaker()
