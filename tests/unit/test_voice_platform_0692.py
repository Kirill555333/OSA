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
    AudioPlatformUnavailableError,
    DesktopAudioBackendFactory,
    UnavailableAudioBackend,
)


class FakeAudioBackend:
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


def test_known_current_platform_is_supported():
    platform = AudioPlatform.current()

    assert platform in (
        AudioPlatform.MACOS,
        AudioPlatform.WINDOWS,
        AudioPlatform.LINUX,
        AudioPlatform.UNKNOWN,
    )


@pytest.mark.parametrize(
    ("platform_value", "expected"),
    [
        ("darwin", AudioPlatform.MACOS),
        ("win32", AudioPlatform.WINDOWS),
    ],
)
def test_platform_values_are_stable(
    platform_value: str,
    expected: AudioPlatform,
):
    original = sys.platform

    try:
        sys.platform = platform_value

        assert AudioPlatform.current() == expected
    finally:
        sys.platform = original


def test_unavailable_backend_is_not_available():
    backend = UnavailableAudioBackend(
        AudioPlatform.MACOS,
    )

    assert backend.available() is False
    assert backend.platform == AudioPlatform.MACOS


def test_unavailable_microphone_fails_closed():
    backend = UnavailableAudioBackend(
        AudioPlatform.WINDOWS,
        reason="Windows audio provider missing.",
    )

    with pytest.raises(
        AudioPlatformUnavailableError,
        match="Windows audio provider missing",
    ):
        backend.microphone()


def test_unavailable_speaker_fails_closed():
    backend = UnavailableAudioBackend(
        AudioPlatform.WINDOWS,
    )

    with pytest.raises(
        AudioPlatformUnavailableError,
        match="No audio backend",
    ):
        backend.speaker()


def test_unavailable_backend_satisfies_protocol():
    backend = UnavailableAudioBackend(
        AudioPlatform.MACOS,
    )

    assert isinstance(
        backend,
        AudioBackend,
    )


def test_factory_returns_platform_specific_backend():
    macos = FakeAudioBackend(
        AudioPlatform.MACOS,
    )
    windows = FakeAudioBackend(
        AudioPlatform.WINDOWS,
    )

    factory = DesktopAudioBackendFactory(
        macos=macos,
        windows=windows,
        enable_default_provider=False,
    )

    assert factory.for_platform(
        AudioPlatform.MACOS
    ) is macos

    assert factory.for_platform(
        AudioPlatform.WINDOWS
    ) is windows


def test_factory_reports_backend_availability():
    windows = FakeAudioBackend(
        AudioPlatform.WINDOWS,
        available=True,
    )
    linux = FakeAudioBackend(
        AudioPlatform.LINUX,
        available=False,
    )

    factory = DesktopAudioBackendFactory(
        windows=windows,
        linux=linux,
        enable_default_provider=False,
    )

    assert factory.available(
        AudioPlatform.WINDOWS
    ) is True

    assert factory.available(
        AudioPlatform.LINUX
    ) is False


def test_factory_defaults_to_unavailable_when_provider_is_disabled():
    factory = DesktopAudioBackendFactory(
        enable_default_provider=False,
    )

    backend = factory.for_platform(
        AudioPlatform.WINDOWS
    )

    assert isinstance(
        backend,
        UnavailableAudioBackend,
    )
    assert backend.available() is False


def test_factory_rejects_invalid_platform():
    factory = DesktopAudioBackendFactory(
        enable_default_provider=False,
    )

    with pytest.raises(
        TypeError,
        match="AudioPlatform",
    ):
        factory.for_platform(
            "windows"
        )


def test_factory_current_backend_is_provider_neutral():
    factory = DesktopAudioBackendFactory(
        enable_default_provider=False,
    )

    backend = factory.current()

    assert isinstance(
        backend.platform,
        AudioPlatform,
    )


def test_factory_available_fails_closed_on_backend_exception():
    class RaisingBackend(FakeAudioBackend):
        def available(self) -> bool:
            raise RuntimeError(
                "device discovery failed"
            )

    factory = DesktopAudioBackendFactory(
        macos=RaisingBackend(
            AudioPlatform.MACOS,
        ),
        enable_default_provider=False,
    )

    assert factory.available(
        AudioPlatform.MACOS
    ) is False


def test_backend_microphone_and_speaker_are_exposed():
    backend = FakeAudioBackend(
        AudioPlatform.MACOS,
    )

    assert isinstance(
        backend.microphone(),
        FakeMicrophone,
    )

    assert isinstance(
        backend.speaker(),
        FakeSpeaker,
    )
