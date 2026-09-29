from __future__ import annotations

from dataclasses import dataclass

from osa.voice.activation import (
    create_keyword_activation_detector,
)
from osa.voice.contracts import (
    VoiceInput,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.session import VoiceSession


@dataclass(frozen=True)
class Response:
    content: str


class Agent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, user_input: str) -> Response:
        self.calls.append(user_input)
        return Response(
            content="Done."
        )


class VAD:
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


class STT:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        return VoiceTranscript(
            text=self.text,
            language="en",
        )


class TTS:
    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        return None


def make_session(
    text: str,
) -> tuple[VoiceSession, Agent]:
    agent = Agent()

    session = VoiceSession(
        agent=agent,
        vad=VAD(),
        stt=STT(text),
        tts=TTS(),
        activation_detector=create_keyword_activation_detector(
            "OSA"
        ),
    )

    return session, agent


def test_non_activated_transcript_never_reaches_agent() -> None:
    session, agent = make_session(
        "open browser"
    )

    result = session.process_audio(
        b"\x01\x00" * 1600,
        sample_rate_hz=16_000,
        channels=1,
    )

    assert result is None
    assert agent.calls == []
    assert session.state == (
        VoiceSessionState.IDLE
    )


def test_activation_strips_wake_phrase_before_agent() -> None:
    session, agent = make_session(
        "OSA open browser"
    )

    result = session.process_audio(
        b"\x01\x00" * 1600,
        sample_rate_hz=16_000,
        channels=1,
    )

    assert result is not None
    assert agent.calls == [
        "open browser"
    ]
    assert result.transcript.text == (
        "OSA open browser"
    )
    assert session.state == (
        VoiceSessionState.SPEAKING
    )


def test_activation_preserves_transcript_in_result() -> None:
    session, _ = make_session(
        "OSA what time is it"
    )

    result = session.process_audio(
        b"\x01\x00" * 1600,
        sample_rate_hz=16_000,
        channels=1,
    )

    assert result is not None
    assert result.transcript.text == (
        "OSA what time is it"
    )
    assert result.response.text == "Done."
