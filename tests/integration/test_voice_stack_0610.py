from __future__ import annotations

from dataclasses import dataclass

from osa.core.agent_voice import AgentVoiceIntegration
from osa.voice import (
    FakeMicrophone,
    FakeSpeaker,
    VoiceInput,
    VoiceSessionState,
    VoiceTranscript,
    VoiceRuntimeState,
    create_voice_runtime_from_backend,
)
from osa.voice.platform import (
    AudioPlatform,
)


@dataclass(frozen=True)
class FakeResponse:
    content: str


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(
        self,
        user_input: str,
    ) -> FakeResponse:
        self.calls.append(
            user_input
        )

        return FakeResponse(
            content="Browser request received.",
        )


class FakeVAD:
    def __init__(
        self,
        speech: bool = True,
    ) -> None:
        self.speech = speech

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        return self.speech

    def reset(self) -> None:
        return None


class FakeSTT:
    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        return VoiceTranscript(
            text="open browser",
            language="en",
        )


class FakeTTS:
    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"tts-audio"

    def stop(self) -> None:
        return None


class FakeAudioBackend:
    def __init__(self) -> None:
        self._microphone = FakeMicrophone(
            (
                VoiceInput(
                    audio=b"\x01\x02",
                    sample_rate_hz=16_000,
                    channels=1,
                ),
            )
        )
        self._speaker = FakeSpeaker()

    @property
    def platform(self) -> AudioPlatform:
        return AudioPlatform.MACOS

    def available(self) -> bool:
        return True

    def microphone(self):
        return self._microphone

    def speaker(self):
        return self._speaker


def test_full_voice_stack_routes_audio_to_same_agent():
    agent = FakeAgent()

    integration = AgentVoiceIntegration(
        agent,
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )

    backend = FakeAudioBackend()

    runtime = create_voice_runtime_from_backend(
        integration.session,
        backend,
    )

    runtime.start()

    result = runtime.run_once()

    assert result is not None
    assert result.transcript.text == (
        "open browser"
    )
    assert result.response.text == (
        "Browser request received."
    )
    assert result.audio == b"tts-audio"

    assert agent.calls == [
        "open browser",
    ]

    assert backend._speaker.play_calls == [
        (
            b"tts-audio",
            16_000,
            1,
        )
    ]

    assert integration.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_voice_stack_uses_single_agent_instance():
    agent = FakeAgent()

    integration = AgentVoiceIntegration(
        agent,
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )

    assert integration.agent is agent
    assert integration.session.agent is agent


def test_voice_stack_interrupt_does_not_replace_agent_result():
    agent = FakeAgent()

    integration = AgentVoiceIntegration(
        agent,
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )

    backend = FakeAudioBackend()

    runtime = create_voice_runtime_from_backend(
        integration.session,
        backend,
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None
    assert integration.state == (
        VoiceSessionState.SPEAKING
    )

    runtime.interrupt()

    assert result.response.text == (
        "Browser request received."
    )
    assert agent.calls == [
        "open browser",
    ]
    assert integration.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_voice_stack_shutdown_is_clean():
    agent = FakeAgent()

    integration = AgentVoiceIntegration(
        agent,
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )

    backend = FakeAudioBackend()

    runtime = create_voice_runtime_from_backend(
        integration.session,
        backend,
    )

    runtime.start()

    result = runtime.run_once()

    assert result is not None

    runtime.stop()

    assert backend._microphone.started is False
    assert runtime.state == (
        VoiceRuntimeState.STOPPED
    )
    assert integration.state == (
        VoiceSessionState.STOPPED
    )


def test_voice_stack_preserves_captured_audio_identity():
    agent = FakeAgent()

    microphone = FakeMicrophone(
        (
            VoiceInput(
                audio=b"captured-audio",
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
    )

    speaker = FakeSpeaker()

    class Backend:
        platform = AudioPlatform.MACOS

        def available(self) -> bool:
            return True

        def microphone(self):
            return microphone

        def speaker(self):
            return speaker

    integration = AgentVoiceIntegration(
        agent,
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )

    runtime = create_voice_runtime_from_backend(
        integration.session,
        Backend(),
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None
    assert microphone.remaining == 0


def test_public_voice_api_exposes_core_contracts():
    import osa.voice as voice

    expected = {
        "VoiceInput",
        "VoiceTranscript",
        "VoiceOutput",
        "VoiceSession",
        "VoiceRuntime",
        "AudioBackend",
        "AudioPlatform",
        "SpeechToText",
        "TextToSpeech",
        "VoiceActivityDetector",
        "MicrophoneInput",
        "SpeakerOutput",
    }

    assert expected.issubset(
        set(voice.__all__)
    )
