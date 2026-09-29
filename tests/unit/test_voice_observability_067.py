from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.observability.logger import (
    InMemoryObservabilityLogger,
)
from osa.observability.voice import (
    ObservableVoiceSession,
)
from osa.voice.contracts import (
    VoiceInput,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.session import VoiceSession


@dataclass(frozen=True)
class FakeResponse:
    content: str


class FakeAgent:
    def chat(
        self,
        user_input: str,
    ) -> FakeResponse:
        return FakeResponse(
            content="Hello from OSA.",
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
    def __init__(
        self,
        *,
        stop_exception: Exception | None = None,
    ) -> None:
        self.stop_exception = stop_exception
        self.stop_calls = 0

    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        self.stop_calls += 1

        if self.stop_exception is not None:
            raise self.stop_exception


class FailingLogger:
    def log(
        self,
        event,
        **data,
    ) -> None:
        raise RuntimeError(
            "logger failure"
        )


def _session(
    *,
    speech: bool = True,
    tts: FakeTTS | None = None,
) -> VoiceSession:
    return VoiceSession(
        agent=FakeAgent(),
        vad=FakeVAD(
            speech=speech,
        ),
        stt=FakeSTT(),
        tts=(
            tts
            if tts is not None
            else FakeTTS()
        ),
    )


def _events(
    logger: InMemoryObservabilityLogger,
) -> tuple[str, ...]:
    return tuple(
        event.event
        for event in logger.events()
    )


def test_wrapper_preserves_session_identity():
    session = _session()
    logger = InMemoryObservabilityLogger()

    observable = ObservableVoiceSession(
        session,
        logger,
    )

    assert observable.session is session
    assert observable.logger is logger


def test_wrapper_preserves_correlation_identifiers():
    observable = ObservableVoiceSession(
        _session(),
        InMemoryObservabilityLogger(),
        session_id="voice-1",
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
    )

    assert observable.session_id == "voice-1"


def test_speech_cycle_emits_observability_events():
    logger = InMemoryObservabilityLogger()

    observable = ObservableVoiceSession(
        _session(),
        logger,
        session_id="voice-1",
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
    )

    result = observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    assert result is not None

    assert _events(logger) == (
        "voice.session.started",
        "voice.listening.started",
        "voice.listening.completed",
        "voice.transcription.completed",
        "voice.response.started",
        "voice.response.completed",
        "voice.tts.completed",
        "voice.session.completed",
    )


def test_speech_cycle_does_not_log_raw_transcript():
    logger = InMemoryObservabilityLogger()

    observable = ObservableVoiceSession(
        _session(),
        logger,
    )

    observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    rendered = repr(
        logger.events()
    )

    assert "open browser" not in rendered
    assert "Hello from OSA." not in rendered


def test_speech_event_contains_safe_lengths():
    logger = InMemoryObservabilityLogger()

    observable = ObservableVoiceSession(
        _session(),
        logger,
    )

    observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    events = logger.events()

    transcription_event = next(
        event
        for event in events
        if event.event == (
            "voice.transcription.completed"
        )
    )

    assert transcription_event.metadata[
        "transcript_present"
    ] is True
    assert transcription_event.metadata[
        "transcript_length"
    ] == len("open browser")

    response_event = next(
        event
        for event in events
        if event.event == (
            "voice.response.completed"
        )
    )

    assert response_event.metadata[
        "response_present"
    ] is True
    assert response_event.metadata[
        "response_length"
    ] == len("Hello from OSA.")


def test_silence_emits_completed_session_without_transcript():
    logger = InMemoryObservabilityLogger()

    observable = ObservableVoiceSession(
        _session(
            speech=False,
        ),
        logger,
    )

    result = observable.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
    )

    assert result is None

    assert _events(logger) == (
        "voice.session.started",
        "voice.listening.started",
        "voice.listening.completed",
        "voice.session.completed",
    )


def test_interrupt_emits_interrupted_event():
    logger = InMemoryObservabilityLogger()
    observable = ObservableVoiceSession(
        _session(),
        logger,
    )

    observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )
    observable.interrupt()

    assert _events(logger)[-1] == (
        "voice.response.interrupted"
    )


def test_playback_completion_is_observed():
    logger = InMemoryObservabilityLogger()
    observable = ObservableVoiceSession(
        _session(),
        logger,
    )

    observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )
    observable.complete_speaking()

    assert _events(logger)[-1] == (
        "voice.response.playback.completed"
    )
    assert observable.state == VoiceSessionState.IDLE


def test_tts_failure_is_observed_and_original_exception_preserved():
    logger = InMemoryObservabilityLogger()

    observable = ObservableVoiceSession(
        _session(
            tts=FakeTTS(
                stop_exception=RuntimeError(
                    "speaker failure"
                )
            )
        ),
        logger,
    )

    observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    with pytest.raises(
        RuntimeError,
        match="speaker failure",
    ):
        observable.interrupt()

    events = _events(logger)

    assert events[-2:] == (
        "voice.tts.failed",
        "voice.session.failed",
    )


def test_logger_failure_never_changes_session_result():
    observable = ObservableVoiceSession(
        _session(),
        FailingLogger(),
    )

    result = observable.process_audio(
        b"\x01\x02",
        sample_rate_hz=16000,
    )

    assert result is not None
    assert result.response.text == (
        "Hello from OSA."
    )
    assert observable.state == VoiceSessionState.SPEAKING


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", " "),
        ("run_id", " "),
        ("task_id", " "),
        ("request_id", " "),
    ],
)
def test_empty_correlation_identifier_is_rejected(
    field: str,
    value: str,
):
    logger = InMemoryObservabilityLogger()

    with pytest.raises(
        ValueError,
        match=field,
    ):
        ObservableVoiceSession(
            _session(),
            logger,
            **{field: value},
        )


def test_invalid_correlation_identifier_type_is_rejected():
    with pytest.raises(
        TypeError,
        match="session_id",
    ):
        ObservableVoiceSession(
            _session(),
            InMemoryObservabilityLogger(),
            session_id=123,
        )


def test_logger_errors_are_not_rethrown():
    observable = ObservableVoiceSession(
        _session(),
        FailingLogger(),
    )

    observable.stop()

    assert observable.state == VoiceSessionState.STOPPED


def test_missing_dependencies_are_rejected():
    logger = InMemoryObservabilityLogger()

    with pytest.raises(
        ValueError,
        match="session",
    ):
        ObservableVoiceSession(
            None,
            logger,
        )

    with pytest.raises(
        ValueError,
        match="logger",
    ):
        ObservableVoiceSession(
            _session(),
            None,
        )
