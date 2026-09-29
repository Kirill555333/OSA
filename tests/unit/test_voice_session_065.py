from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.voice.contracts import (
    VoiceInput,
    VoiceSessionConfig,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.session import (
    VoiceSession,
    VoiceSessionError,
    VoiceSessionResult,
    VoiceSessionStateError,
)


@dataclass(frozen=True)
class FakeModelResponse:
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
    ) -> FakeModelResponse:
        self.calls.append(user_input)

        return FakeModelResponse(
            content=self.response,
        )


class FakeVAD:
    def __init__(
        self,
        *,
        speech: bool = True,
    ) -> None:
        self.speech = speech
        self.calls: list[
            tuple[
                bytes,
                int,
                int,
            ]
        ] = []
        self.reset_calls = 0

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        self.calls.append(
            (
                audio,
                sample_rate_hz,
                channels,
            )
        )

        return self.speech

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSTT:
    def __init__(
        self,
        transcript: VoiceTranscript | None = None,
    ) -> None:
        self.transcript = (
            transcript
            if transcript is not None
            else VoiceTranscript(
                text="hello OSA",
                language="en",
                confidence=0.95,
            )
        )
        self.calls: list[VoiceInput] = []

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        self.calls.append(
            voice_input
        )

        return self.transcript


class FakeTTS:
    def __init__(
        self,
        audio: bytes = b"speech-audio",
        *,
        exception: Exception | None = None,
    ) -> None:
        self.audio = audio
        self.exception = exception
        self.synthesize_calls = []
        self.stop_calls = 0

    def synthesize(
        self,
        output,
    ) -> bytes:
        self.synthesize_calls.append(
            output
        )

        if self.exception is not None:
            raise self.exception

        return self.audio

    def stop(self) -> None:
        self.stop_calls += 1


def _session(
    *,
    speech: bool = True,
    response: str = "Hello from OSA.",
    language: str = "en",
    tts_audio: bytes = b"speech-audio",
    tts_exception: Exception | None = None,
):
    agent = FakeAgent(
        response=response,
    )
    vad = FakeVAD(
        speech=speech,
    )
    stt = FakeSTT(
        VoiceTranscript(
            text="open browser",
            language=language,
        )
    )
    tts = FakeTTS(
        audio=tts_audio,
        exception=tts_exception,
    )

    session = VoiceSession(
        agent=agent,
        vad=vad,
        stt=stt,
        tts=tts,
        config=VoiceSessionConfig(
            language="en",
        ),
    )

    return session, agent, vad, stt, tts


def test_initial_state_is_idle():
    session, _, _, _, _ = _session()

    assert session.state == VoiceSessionState.IDLE


def test_process_audio_ignores_non_speech():
    session, agent, vad, stt, tts = _session(
        speech=False,
    )

    result = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert result is None
    assert session.state == VoiceSessionState.IDLE
    assert len(vad.calls) == 1
    assert stt.calls == []
    assert agent.calls == []
    assert tts.synthesize_calls == []


def test_process_audio_runs_vad_stt_agent_tts_in_order():
    session, agent, vad, stt, tts = _session()

    result = session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
        channels=1,
    )

    assert isinstance(
        result,
        VoiceSessionResult,
    )

    assert vad.calls == [
        (
            b"\x01\x02",
            16000,
            1,
        )
    ]

    assert len(stt.calls) == 1
    assert stt.calls[0].audio == b"\x01\x02"
    assert stt.calls[0].sample_rate_hz == 16000

    assert agent.calls == [
        "open browser"
    ]

    assert len(tts.synthesize_calls) == 1
    assert tts.synthesize_calls[0].text == (
        "Hello from OSA."
    )

    assert result.transcript.text == "open browser"
    assert result.response.text == "Hello from OSA."
    assert result.response.language == "en"
    assert result.audio == b"speech-audio"
    assert session.state == VoiceSessionState.SPEAKING


def test_language_falls_back_to_session_config():
    session, _, _, stt, tts = _session(
        language="auto",
    )

    stt.transcript = VoiceTranscript(
        text="hello",
        language=None,
    )

    result = session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    assert result is not None
    assert result.response.language == "en"
    assert tts.synthesize_calls[0].language == "en"


def test_complete_speaking_returns_session_to_idle():
    session, _, _, _, _ = _session()

    session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    session.complete_speaking()

    assert session.state == VoiceSessionState.IDLE


def test_interrupt_stops_tts_and_returns_to_idle():
    session, _, _, _, tts = _session()

    session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    session.interrupt()

    assert tts.stop_calls == 1
    assert session.state == VoiceSessionState.IDLE


def test_stop_from_speaking_stops_tts_and_enters_stopped():
    session, _, _, _, tts = _session()

    session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    session.stop()

    assert tts.stop_calls == 1
    assert session.state == VoiceSessionState.STOPPED


def test_stop_from_idle_enters_stopped():
    session, _, _, _, _ = _session()

    session.stop()

    assert session.state == VoiceSessionState.STOPPED


def test_stop_is_idempotent():
    session, _, _, _, tts = _session()

    session.stop()
    session.stop()

    assert session.state == VoiceSessionState.STOPPED
    assert tts.stop_calls == 0


def test_reset_returns_to_idle_and_resets_vad():
    session, _, vad, _, _ = _session()

    session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    session.reset()

    assert session.state == VoiceSessionState.IDLE
    assert vad.reset_calls == 1


def test_process_audio_is_rejected_while_speaking():
    session, _, _, _, _ = _session()

    session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        VoiceSessionStateError,
        match="requires voice session state",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )


def test_complete_speaking_requires_speaking_state():
    session, _, _, _, _ = _session()

    with pytest.raises(
        VoiceSessionStateError,
        match="SPEAKING".lower(),
    ):
        session.complete_speaking()


def test_interrupt_requires_speaking_state():
    session, _, _, _, _ = _session()

    with pytest.raises(
        VoiceSessionStateError,
        match="SPEAKING".lower(),
    ):
        session.interrupt()


def test_operations_are_rejected_after_stop():
    session, _, _, _, _ = _session()

    session.stop()

    with pytest.raises(
        VoiceSessionStateError,
        match="idle",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )


def test_vad_exception_moves_session_to_error():
    class RaisingVAD(FakeVAD):
        def detect(
            self,
            audio: bytes,
            *,
            sample_rate_hz: int,
            channels: int = 1,
        ) -> bool:
            raise RuntimeError(
                "vad failure"
            )

    agent = FakeAgent()
    vad = RaisingVAD()
    stt = FakeSTT()
    tts = FakeTTS()

    session = VoiceSession(
        agent=agent,
        vad=vad,
        stt=stt,
        tts=tts,
    )

    with pytest.raises(
        RuntimeError,
        match="vad failure",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_stt_exception_moves_session_to_error():
    class RaisingSTT(FakeSTT):
        def transcribe(
            self,
            voice_input: VoiceInput,
        ):
            raise RuntimeError(
                "stt failure"
            )

    session = VoiceSession(
        agent=FakeAgent(),
        vad=FakeVAD(),
        stt=RaisingSTT(),
        tts=FakeTTS(),
    )

    with pytest.raises(
        RuntimeError,
        match="stt failure",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_agent_exception_moves_session_to_error():
    class RaisingAgent:
        def chat(
            self,
            user_input: str,
        ):
            raise RuntimeError(
                "agent failure"
            )

    session = VoiceSession(
        agent=RaisingAgent(),
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=FakeTTS(),
    )

    with pytest.raises(
        RuntimeError,
        match="agent failure",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_tts_exception_moves_session_to_error():
    session, _, _, _, _ = _session(
        tts_exception=RuntimeError(
            "tts failure"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="tts failure",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_empty_tts_audio_is_rejected():
    session, _, _, _, _ = _session(
        tts_audio=b"",
    )

    with pytest.raises(
        VoiceSessionError,
        match="empty audio",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_invalid_agent_response_is_rejected():
    class InvalidAgent:
        def chat(
            self,
            user_input: str,
        ):
            return object()

    session = VoiceSession(
        agent=InvalidAgent(),
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=FakeTTS(),
    )

    with pytest.raises(
        VoiceSessionError,
        match="string content",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_empty_agent_response_is_rejected():
    session, _, _, _, _ = _session(
        response="   ",
    )

    with pytest.raises(
        VoiceSessionError,
        match="cannot be empty",
    ):
        session.process_audio(
            b"\x01\x02",
            sample_rate_hz=16000,
        )

    assert session.state == VoiceSessionState.ERROR


def test_interrupt_tts_failure_moves_session_to_error():
    class RaisingTTS(FakeTTS):
        def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError(
                "stop failure"
            )

    tts = RaisingTTS()

    session = VoiceSession(
        agent=FakeAgent(),
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=tts,
    )

    session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        RuntimeError,
        match="stop failure",
    ):
        session.interrupt()

    assert session.state == VoiceSessionState.ERROR


def test_constructor_requires_dependencies():
    with pytest.raises(
        ValueError,
        match="agent",
    ):
        VoiceSession(
            None,
            FakeVAD(),
            FakeSTT(),
            FakeTTS(),
        )

    with pytest.raises(
        ValueError,
        match="vad",
    ):
        VoiceSession(
            FakeAgent(),
            None,
            FakeSTT(),
            FakeTTS(),
        )

    with pytest.raises(
        ValueError,
        match="stt",
    ):
        VoiceSession(
            FakeAgent(),
            FakeVAD(),
            None,
            FakeTTS(),
        )

    with pytest.raises(
        ValueError,
        match="tts",
    ):
        VoiceSession(
            FakeAgent(),
            FakeVAD(),
            FakeSTT(),
            None,
        )


def test_config_is_preserved():
    config = VoiceSessionConfig(
        language="en-us",
        max_utterance_seconds=15,
        interrupt_enabled=False,
    )

    session = VoiceSession(
        agent=FakeAgent(),
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=FakeTTS(),
        config=config,
    )

    assert session.config is config


def test_voice_session_result_is_immutable():
    session, _, _, _, _ = _session()

    result = session.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    assert result is not None

    with pytest.raises(
        AttributeError,
    ):
        result.audio = b"changed"
