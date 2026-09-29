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
    VoiceSessionStateError,
)


@dataclass(frozen=True)
class FakeResponse:
    content: str


class FakeAgent:
    def chat(self, user_input: str) -> FakeResponse:
        return FakeResponse(
            content="Hello from OSA.",
        )


class FakeVAD:
    def __init__(self, speech: bool = True) -> None:
        self.speech = speech
        self.stop_reset_calls = 0

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        return self.speech

    def reset(self) -> None:
        self.stop_reset_calls += 1


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
    def __init__(
        self,
        *,
        stop_exception: Exception | None = None,
    ) -> None:
        self.stop_calls = 0
        self.stop_exception = stop_exception

    def synthesize(self, output) -> bytes:
        return b"audio"

    def stop(self) -> None:
        self.stop_calls += 1

        if self.stop_exception is not None:
            raise self.stop_exception


def _session(
    *,
    config: VoiceSessionConfig | None = None,
    tts: FakeTTS | None = None,
) -> tuple[
    VoiceSession,
    FakeTTS,
]:
    tts_value = (
        tts
        if tts is not None
        else FakeTTS()
    )

    session = VoiceSession(
        agent=FakeAgent(),
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=tts_value,
        config=config,
    )

    return session, tts_value


def test_interrupt_is_available_only_while_speaking():
    session, _ = _session()

    with pytest.raises(
        VoiceSessionStateError,
        match="requires voice session state",
    ):
        session.interrupt()


def test_interrupt_returns_session_to_idle():
    session, tts = _session()

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert session.state == VoiceSessionState.SPEAKING

    session.interrupt()

    assert tts.stop_calls == 1
    assert session.state == VoiceSessionState.IDLE


def test_interrupt_is_idempotently_forbidden_after_first_interrupt():
    session, tts = _session()

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    session.interrupt()

    with pytest.raises(
        VoiceSessionStateError,
        match="requires voice session state",
    ):
        session.interrupt()

    assert tts.stop_calls == 1


def test_interrupt_does_not_change_completed_agent_result():
    session, tts = _session()

    result = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert result is not None
    response_before = result.response

    session.interrupt()

    assert result.response is response_before
    assert result.response.text == "Hello from OSA."
    assert tts.stop_calls == 1


def test_interrupt_can_be_disabled():
    session, tts = _session(
        config=VoiceSessionConfig(
            interrupt_enabled=False,
        )
    )

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        VoiceSessionStateError,
        match="disabled",
    ):
        session.interrupt()

    assert tts.stop_calls == 0
    assert session.state == VoiceSessionState.SPEAKING


def test_interrupt_tts_failure_fails_closed():
    session, tts = _session(
        tts=FakeTTS(
            stop_exception=RuntimeError(
                "speaker failure"
            )
        )
    )

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        RuntimeError,
        match="speaker failure",
    ):
        session.interrupt()

    assert tts.stop_calls == 1
    assert session.state == VoiceSessionState.ERROR


def test_reset_stops_active_speech():
    session, tts = _session()

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    session.reset()

    assert tts.stop_calls == 1
    assert session.state == VoiceSessionState.IDLE


def test_reset_from_idle_does_not_stop_tts():
    session, tts = _session()

    session.reset()

    assert tts.stop_calls == 0
    assert session.state == VoiceSessionState.IDLE


def test_stop_stops_active_speech_once():
    session, tts = _session()

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    session.stop()

    assert tts.stop_calls == 1
    assert session.state == VoiceSessionState.STOPPED

    session.stop()

    assert tts.stop_calls == 1


def test_stop_from_idle_does_not_call_tts():
    session, tts = _session()

    session.stop()

    assert tts.stop_calls == 0
    assert session.state == VoiceSessionState.STOPPED


def test_interrupted_session_can_process_new_audio():
    session, tts = _session()

    first = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert first is not None

    session.interrupt()

    second = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert second is not None
    assert session.state == VoiceSessionState.SPEAKING
    assert tts.stop_calls == 1


def test_reset_tts_failure_moves_to_error():
    session, _ = _session(
        tts=FakeTTS(
            stop_exception=RuntimeError(
                "reset speaker failure"
            )
        )
    )

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        RuntimeError,
        match="reset speaker failure",
    ):
        session.reset()

    assert session.state == VoiceSessionState.ERROR


def test_stop_tts_failure_moves_to_error():
    session, _ = _session(
        tts=FakeTTS(
            stop_exception=RuntimeError(
                "stop speaker failure"
            )
        )
    )

    session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        RuntimeError,
        match="stop speaker failure",
    ):
        session.stop()

    assert session.state == VoiceSessionState.ERROR
