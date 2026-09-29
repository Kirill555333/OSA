from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.core.agent_voice import (
    AgentVoiceIntegration,
    AgentVoiceIntegrationError,
    VoiceCapableAgent,
)
from osa.voice.contracts import (
    VoiceInput,
    VoiceSessionConfig,
    VoiceSessionState,
    VoiceTranscript,
)


@dataclass(frozen=True)
class FakeResponse:
    content: str


class FakeAgent:
    def __init__(
        self,
        response: str = "Hello from OSA.",
    ) -> None:
        self.response = response
        self.calls: list[str] = []

    def chat(
        self,
        user_input: str,
    ) -> FakeResponse:
        self.calls.append(user_input)

        return FakeResponse(
            content=self.response,
        )


class FakeVAD:
    def __init__(
        self,
        speech: bool = True,
    ) -> None:
        self.speech = speech
        self.reset_calls = 0

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        return self.speech

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSTT:
    def __init__(
        self,
        text: str = "open browser",
    ) -> None:
        self.text = text
        self.calls: list[VoiceInput] = []

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        self.calls.append(
            voice_input
        )

        return VoiceTranscript(
            text=self.text,
            language="en",
        )


class FakeTTS:
    def __init__(
        self,
        audio: bytes = b"speech",
    ) -> None:
        self.audio = audio
        self.synthesized = []
        self.stop_calls = 0

    def synthesize(
        self,
        output,
    ) -> bytes:
        self.synthesized.append(
            output
        )

        return self.audio

    def stop(self) -> None:
        self.stop_calls += 1


class InvalidAgent:
    pass


def _integration(
    *,
    agent: FakeAgent | None = None,
    speech: bool = True,
):
    agent_value = (
        agent
        if agent is not None
        else FakeAgent()
    )

    vad = FakeVAD(
        speech=speech,
    )
    stt = FakeSTT()
    tts = FakeTTS()

    integration = AgentVoiceIntegration(
        agent_value,
        vad,
        stt,
        tts,
        config=VoiceSessionConfig(
            language="en",
        ),
    )

    return (
        integration,
        agent_value,
        vad,
        stt,
        tts,
    )


def test_agent_protocol_is_runtime_checkable():
    agent = FakeAgent()

    assert isinstance(
        agent,
        VoiceCapableAgent,
    )


def test_integration_preserves_agent_identity():
    integration, agent, _, _, _ = _integration()

    assert integration.agent is agent
    assert integration.session.agent is agent


def test_integration_exposes_session():
    integration, _, _, _, _ = _integration()

    assert integration.session is not None
    assert integration.state == (
        VoiceSessionState.IDLE
    )


def test_process_audio_delegates_to_same_agent():
    integration, agent, _, stt, tts = (
        _integration()
    )

    result = integration.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    assert result is not None
    assert agent.calls == [
        "open browser",
    ]
    assert len(stt.calls) == 1
    assert len(tts.synthesized) == 1
    assert result.response.text == (
        "Hello from OSA."
    )
    assert result.audio == b"speech"
    assert integration.state == (
        VoiceSessionState.SPEAKING
    )


def test_non_speech_does_not_call_agent():
    integration, agent, _, _, _ = _integration(
        speech=False,
    )

    result = integration.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert result is None
    assert agent.calls == []
    assert integration.state == (
        VoiceSessionState.IDLE
    )


def test_complete_speaking_delegates_to_session():
    integration, _, _, _, _ = _integration()

    integration.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    integration.complete_speaking()

    assert integration.state == (
        VoiceSessionState.IDLE
    )


def test_interrupt_delegates_to_session():
    integration, _, _, _, tts = _integration()

    integration.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    integration.interrupt()

    assert tts.stop_calls == 1
    assert integration.state == (
        VoiceSessionState.IDLE
    )


def test_reset_delegates_to_session():
    integration, _, vad, _, _ = _integration()

    integration.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    integration.reset()

    assert vad.reset_calls == 1
    assert integration.state == (
        VoiceSessionState.IDLE
    )


def test_stop_delegates_to_session():
    integration, _, _, _, tts = _integration()

    integration.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    integration.stop()

    assert tts.stop_calls == 1
    assert integration.state == (
        VoiceSessionState.STOPPED
    )


def test_integration_requires_agent():
    with pytest.raises(
        ValueError,
        match="agent",
    ):
        AgentVoiceIntegration(
            None,
            FakeVAD(),
            FakeSTT(),
            FakeTTS(),
        )


def test_integration_rejects_invalid_agent():
    with pytest.raises(
        TypeError,
        match="chat",
    ):
        AgentVoiceIntegration(
            InvalidAgent(),
            FakeVAD(),
            FakeSTT(),
            FakeTTS(),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("vad", None),
        ("stt", None),
        ("tts", None),
    ],
)
def test_integration_requires_voice_dependencies(
    name: str,
    value,
):
    dependencies = {
        "vad": FakeVAD(),
        "stt": FakeSTT(),
        "tts": FakeTTS(),
    }
    dependencies[name] = value

    with pytest.raises(
        ValueError,
        match=name,
    ):
        AgentVoiceIntegration(
            FakeAgent(),
            dependencies["vad"],
            dependencies["stt"],
            dependencies["tts"],
        )


def test_config_is_forwarded_to_session():
    config = VoiceSessionConfig(
        language="ru-ru",
        max_utterance_seconds=10,
        interrupt_enabled=False,
    )

    integration = AgentVoiceIntegration(
        FakeAgent(),
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
        config=config,
    )

    assert integration.config is config
    assert (
        integration.config.language
        == "ru-ru"
    )
    assert (
        integration.config.max_utterance_seconds
        == 10.0
    )
    assert (
        integration.config.interrupt_enabled
        is False
    )


def test_voice_processing_preserves_voice_session_errors():
    integration, _, _, _, _ = _integration()

    with pytest.raises(
        Exception,
    ):
        integration.complete_speaking()


def test_agent_response_flows_to_tts():
    agent = FakeAgent(
        response="The browser is ready.",
    )

    integration, _, _, _, tts = _integration(
        agent=agent,
    )

    result = integration.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    assert result is not None
    assert tts.synthesized[0].text == (
        "The browser is ready."
    )


def test_invalid_voice_session_creation_is_wrapped():
    class BrokenVAD:
        pass

    with pytest.raises(
        AgentVoiceIntegrationError,
        match="voice session",
    ):
        AgentVoiceIntegration(
            FakeAgent(),
            BrokenVAD(),
            FakeSTT(),
            FakeTTS(),
        )
