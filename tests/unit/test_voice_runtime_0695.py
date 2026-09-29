from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.voice.audio import (
    FakeMicrophone,
    FakeSpeaker,
)
from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.runtime import (
    VoiceRuntime,
    VoiceRuntimeConfig,
    VoiceRuntimeError,
    VoiceRuntimeState,
    create_voice_runtime,
)
from osa.voice.session import (
    VoiceSession,
    VoiceSessionResult,
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
    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        return None


class RaisingMicrophone:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True
        raise RuntimeError(
            "microphone unavailable"
        )

    def read(self):
        raise RuntimeError(
            "should not read"
        )

    def stop(self) -> None:
        return None


class RaisingSpeaker:
    def play(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> None:
        raise RuntimeError(
            "speaker unavailable"
        )

    def stop(self) -> None:
        return None


def _session(
    *,
    speech: bool = True,
) -> tuple[
    VoiceSession,
    FakeAgent,
]:
    agent = FakeAgent()

    session = VoiceSession(
        agent=agent,
        vad=FakeVAD(
            speech=speech,
        ),
        stt=FakeSTT(),
        tts=FakeTTS(),
    )

    return session, agent


def _microphone(
    *,
    audio: bytes = b"\x01\x02",
) -> FakeMicrophone:
    return FakeMicrophone(
        (
            VoiceInput(
                audio=audio,
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
    )


def _result() -> VoiceSessionResult:
    return VoiceSessionResult(
        transcript=VoiceTranscript(
            text="hello",
            language="en",
        ),
        response=VoiceOutput(
            text="Hello from OSA.",
            language="en",
        ),
        audio=b"speech",
    )


def test_runtime_starts_stopped():
    session, _ = _session()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        FakeSpeaker(),
    )

    assert runtime.state == VoiceRuntimeState.STOPPED


def test_runtime_start_starts_microphone():
    session, _ = _session()
    microphone = _microphone()

    runtime = VoiceRuntime(
        session,
        microphone,
        FakeSpeaker(),
    )

    runtime.start()

    assert runtime.state == VoiceRuntimeState.READY
    assert microphone.started is True
    assert microphone.start_calls == 1


def test_runtime_start_is_idempotent():
    session, _ = _session()
    microphone = _microphone()

    runtime = VoiceRuntime(
        session,
        microphone,
        FakeSpeaker(),
    )

    runtime.start()
    runtime.start()

    assert runtime.state == VoiceRuntimeState.READY
    assert microphone.start_calls == 1


def test_listen_once_requires_started_runtime():
    session, _ = _session()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        FakeSpeaker(),
    )

    with pytest.raises(
        VoiceRuntimeError,
        match="READY",
    ):
        runtime.listen_once()


def test_listen_once_processes_microphone_input():
    session, agent = _session()
    microphone = _microphone()

    runtime = VoiceRuntime(
        session,
        microphone,
        FakeSpeaker(),
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None
    assert result.transcript.text == (
        "open browser"
    )
    assert result.response.text == (
        "Hello from OSA."
    )
    assert agent.calls == [
        "open browser",
    ]
    assert runtime.state == VoiceRuntimeState.READY
    assert session.state == VoiceSessionState.SPEAKING


def test_listen_once_returns_none_for_silence():
    session, agent = _session(
        speech=False,
    )

    runtime = VoiceRuntime(
        session,
        _microphone(),
        FakeSpeaker(),
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is None
    assert agent.calls == []
    assert runtime.state == VoiceRuntimeState.READY
    assert session.state == VoiceSessionState.IDLE


def test_speak_plays_audio_and_completes_session():
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None

    runtime.speak(result)

    assert speaker.play_calls == [
        (
            b"speech",
            16_000,
            1,
        )
    ]
    assert session.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_speak_requires_session_speaking_state():
    session, _ = _session()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        FakeSpeaker(),
    )

    runtime.start()

    with pytest.raises(
        VoiceRuntimeError,
        match="SPEAKING",
    ):
        runtime.speak(
            _result()
        )


def test_speak_rejects_invalid_result_type():
    session, _ = _session()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        FakeSpeaker(),
    )

    with pytest.raises(
        TypeError,
        match="VoiceSessionResult",
    ):
        runtime.speak(
            object()
        )


def test_run_once_performs_complete_voice_cycle():
    session, agent = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    result = runtime.run_once()

    assert result is not None
    assert agent.calls == [
        "open browser",
    ]
    assert speaker.play_calls == [
        (
            b"speech",
            16_000,
            1,
        )
    ]
    assert session.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_run_once_returns_none_for_non_speech():
    session, agent = _session(
        speech=False,
    )
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        speaker,
    )

    runtime.start()

    result = runtime.run_once()

    assert result is None
    assert agent.calls == []
    assert speaker.play_calls == []
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_microphone_failure_moves_runtime_to_error():
    session, _ = _session()

    runtime = VoiceRuntime(
        session,
        RaisingMicrophone(),
        FakeSpeaker(),
    )

    with pytest.raises(
        RuntimeError,
        match="microphone unavailable",
    ):
        runtime.start()

    assert runtime.state == (
        VoiceRuntimeState.ERROR
    )


def test_speaker_failure_moves_runtime_to_error():
    session, _ = _session()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        RaisingSpeaker(),
    )

    runtime.start()
    result = runtime.listen_once()

    assert result is not None

    with pytest.raises(
        RuntimeError,
        match="speaker unavailable",
    ):
        runtime.speak(result)

    assert runtime.state == (
        VoiceRuntimeState.ERROR
    )


def test_interrupt_stops_speaker_and_session():
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None
    assert session.state == (
        VoiceSessionState.SPEAKING
    )

    runtime.interrupt()

    assert speaker.stop_calls == 1
    assert session.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_reset_stops_speaker_and_resets_session():
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None

    runtime.reset()

    assert speaker.stop_calls == 1
    assert session.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )


def test_stop_stops_all_resources():
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    runtime.stop()

    assert microphone.started is False
    assert microphone.stop_calls == 1
    assert speaker.stop_calls == 1
    assert runtime.state == (
        VoiceRuntimeState.STOPPED
    )
    assert session.state == (
        VoiceSessionState.STOPPED
    )


def test_stop_is_idempotent_for_resources():
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    runtime.stop()
    runtime.stop()

    assert microphone.stop_calls == 1
    assert speaker.stop_calls == 2
    assert runtime.state == (
        VoiceRuntimeState.STOPPED
    )


def test_runtime_config_defaults():
    config = VoiceRuntimeConfig()

    assert config.output_sample_rate_hz == 16_000
    assert config.output_channels == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_sample_rate_hz", 0),
        ("output_sample_rate_hz", -1),
        ("output_channels", 0),
        ("output_channels", -1),
    ],
)
def test_invalid_runtime_config_is_rejected(
    field: str,
    value: int,
):
    with pytest.raises(
        ValueError,
        match=field,
    ):
        VoiceRuntimeConfig(
            **{field: value}
        )


def test_custom_runtime_config_is_used_for_playback():
    session, _ = _session()
    speaker = FakeSpeaker()

    runtime = VoiceRuntime(
        session,
        _microphone(),
        speaker,
        config=VoiceRuntimeConfig(
            output_sample_rate_hz=24_000,
            output_channels=2,
        ),
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None

    runtime.speak(result)

    assert speaker.play_calls == [
        (
            b"speech",
            24_000,
            2,
        )
    ]


def test_runtime_factory_creates_runtime():
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    runtime = create_voice_runtime(
        session,
        microphone,
        speaker,
    )

    assert isinstance(
        runtime,
        VoiceRuntime,
    )


@pytest.mark.parametrize(
    ("dependency", "value", "message"),
    [
        ("session", None, "session"),
        ("microphone", None, "microphone"),
        ("speaker", None, "speaker"),
    ],
)
def test_runtime_rejects_missing_dependencies(
    dependency: str,
    value,
    message: str,
):
    session, _ = _session()
    microphone = _microphone()
    speaker = FakeSpeaker()

    dependencies = {
        "session": session,
        "microphone": microphone,
        "speaker": speaker,
    }
    dependencies[dependency] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        VoiceRuntime(
            dependencies["session"],
            dependencies["microphone"],
            dependencies["speaker"],
        )
