from __future__ import annotations

import pytest

from osa.voice.audio import (
    FakeMicrophone,
    FakeSpeaker,
)
from osa.voice.composition import (
    VoiceCompositionError,
    create_default_voice_runtime,
    create_voice_runtime_from_backend,
)
from osa.voice.contracts import (
    VoiceInput,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.platform import (
    AudioPlatform,
    DesktopAudioBackendFactory,
    UnavailableAudioBackend,
)
from osa.voice.runtime import (
    VoiceRuntime,
    VoiceRuntimeConfig,
)
from osa.voice.session import VoiceSession


class FakeAgent:
    def chat(
        self,
        user_input: str,
    ):
        class Response:
            content = "Hello from OSA."

        return Response()


class FakeVAD:
    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        return True

    def reset(self) -> None:
        return None


class FakeSTT:
    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        return VoiceTranscript(
            text="hello",
            language="en",
        )


class FakeTTS:
    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        return None


class FakeBackend:
    def __init__(
        self,
        platform: AudioPlatform = AudioPlatform.MACOS,
        *,
        available: bool = True,
        microphone=None,
        speaker=None,
    ) -> None:
        self._platform = platform
        self._available = available
        self._microphone = (
            microphone
            if microphone is not None
            else FakeMicrophone(
                (
                    VoiceInput(
                        audio=b"\x01\x02",
                        sample_rate_hz=16_000,
                        channels=1,
                    ),
                )
            )
        )
        self._speaker = (
            speaker
            if speaker is not None
            else FakeSpeaker()
        )

    @property
    def platform(self) -> AudioPlatform:
        return self._platform

    def available(self) -> bool:
        return self._available

    def microphone(self):
        return self._microphone

    def speaker(self):
        return self._speaker


def _session() -> VoiceSession:
    return VoiceSession(
        agent=FakeAgent(),
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=FakeTTS(),
    )


def test_create_runtime_from_backend_returns_runtime():
    backend = FakeBackend()

    runtime = create_voice_runtime_from_backend(
        _session(),
        backend,
    )

    assert isinstance(
        runtime,
        VoiceRuntime,
    )


def test_runtime_from_backend_preserves_config():
    backend = FakeBackend()

    config = VoiceRuntimeConfig(
        output_sample_rate_hz=24_000,
        output_channels=2,
    )

    runtime = create_voice_runtime_from_backend(
        _session(),
        backend,
        config=config,
    )

    assert runtime.config is config


def test_runtime_from_backend_uses_backend_devices():
    microphone = FakeMicrophone()
    speaker = FakeSpeaker()

    backend = FakeBackend(
        microphone=microphone,
        speaker=speaker,
    )

    runtime = create_voice_runtime_from_backend(
        _session(),
        backend,
    )

    assert runtime.microphone is microphone
    assert runtime.speaker is speaker


def test_unavailable_backend_fails_closed():
    backend = FakeBackend(
        available=False,
        platform=AudioPlatform.WINDOWS,
    )

    with pytest.raises(
        RuntimeError,
        match="unavailable",
    ):
        create_voice_runtime_from_backend(
            _session(),
            backend,
        )


def test_backend_availability_exception_is_wrapped():
    class RaisingBackend(FakeBackend):
        def available(self) -> bool:
            raise RuntimeError(
                "device query failure"
            )

    with pytest.raises(
        VoiceCompositionError,
        match="availability",
    ):
        create_voice_runtime_from_backend(
            _session(),
            RaisingBackend(),
        )


def test_invalid_microphone_from_backend_is_rejected():
    backend = FakeBackend(
        microphone=object(),
    )

    with pytest.raises(
        VoiceCompositionError,
        match="invalid microphone",
    ):
        create_voice_runtime_from_backend(
            _session(),
            backend,
        )


def test_invalid_speaker_from_backend_is_rejected():
    backend = FakeBackend(
        speaker=object(),
    )

    with pytest.raises(
        VoiceCompositionError,
        match="invalid speaker",
    ):
        create_voice_runtime_from_backend(
            _session(),
            backend,
        )


def test_backend_factory_current_backend_can_build_runtime():
    backend = FakeBackend(
        platform=AudioPlatform.MACOS,
    )

    factory = DesktopAudioBackendFactory(
        macos=backend,
        enable_default_provider=False,
    )

    runtime = create_default_voice_runtime(
        _session(),
        factory=factory,
    )

    assert isinstance(
        runtime,
        VoiceRuntime,
    )
    assert runtime.microphone is backend.microphone()
    assert runtime.speaker is backend.speaker()


def test_default_runtime_uses_current_platform_selection():
    backend = FakeBackend(
        platform=AudioPlatform.MACOS,
    )

    factory = DesktopAudioBackendFactory(
        macos=backend,
        enable_default_provider=False,
    )

    runtime = create_default_voice_runtime(
        _session(),
        factory=factory,
    )

    runtime.start()

    result = runtime.run_once()

    assert result is not None
    assert result.response.text == (
        "Hello from OSA."
    )
    assert runtime.state.value == "ready"


def test_default_runtime_rejects_invalid_factory():
    with pytest.raises(
        TypeError,
        match="DesktopAudioBackendFactory",
    ):
        create_default_voice_runtime(
            _session(),
            factory=object(),
        )


def test_default_runtime_factory_can_fail_closed():
    factory = DesktopAudioBackendFactory(
        windows=UnavailableAudioBackend(
            AudioPlatform.WINDOWS,
        ),
        enable_default_provider=False,
    )

    class WindowsOnlyFactory(
        DesktopAudioBackendFactory
    ):
        def current(self):
            return self.for_platform(
                AudioPlatform.WINDOWS
            )

    with pytest.raises(
        RuntimeError,
        match="unavailable",
    ):
        create_default_voice_runtime(
            _session(),
            factory=WindowsOnlyFactory(
                windows=UnavailableAudioBackend(
                    AudioPlatform.WINDOWS,
                ),
                enable_default_provider=False,
            ),
        )


def test_composed_runtime_starts_and_stops_devices():
    microphone = FakeMicrophone(
        (
            VoiceInput(
                audio=b"\x01\x02",
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
    )
    speaker = FakeSpeaker()

    backend = FakeBackend(
        microphone=microphone,
        speaker=speaker,
    )

    runtime = create_voice_runtime_from_backend(
        _session(),
        backend,
    )

    runtime.start()
    assert microphone.started is True

    result = runtime.run_once()

    assert result is not None
    assert runtime.state.value == "ready"

    runtime.stop()

    assert microphone.started is False
    assert runtime.state.value == "stopped"


def test_session_must_be_present():
    backend = FakeBackend()

    with pytest.raises(
        ValueError,
        match="session",
    ):
        create_voice_runtime_from_backend(
            None,
            backend,
        )


def test_backend_must_be_present():
    with pytest.raises(
        ValueError,
        match="backend",
    ):
        create_voice_runtime_from_backend(
            _session(),
            None,
        )


def test_backend_must_satisfy_protocol():
    with pytest.raises(
        TypeError,
        match="AudioBackend",
    ):
        create_voice_runtime_from_backend(
            _session(),
            object(),
        )
