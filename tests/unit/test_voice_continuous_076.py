from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from osa.voice.audio import FakeMicrophone, FakeSpeaker
from osa.voice.continuous import (
    ContinuousVoiceRuntime,
    ContinuousVoiceRuntimeConfig,
    ContinuousVoiceRuntimeState,
    create_continuous_voice_runtime,
)
from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.runtime import (
    VoiceRuntime,
    VoiceRuntimeState,
)
from osa.voice.session import VoiceSession


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
        self.calls.append(user_input)
        return FakeResponse(
            content="Hello from OSA."
        )


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
    def __init__(self) -> None:
        self.stop_calls = 0

    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        self.stop_calls += 1


class BlockingMicrophone:
    """Microphone used by lifecycle tests; it never exhausts."""

    def __init__(self) -> None:
        self._started = False
        self._stop_event = threading.Event()
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True
        self._stop_event.clear()
        self.start_calls += 1

    def read(self) -> VoiceInput:
        self._stop_event.wait(0.01)

        if not self._started:
            raise RuntimeError(
                "microphone stopped"
            )

        return VoiceInput(
            audio=b"\x01\x00" * 1600,
            sample_rate_hz=16_000,
            channels=1,
        )

    def stop(self) -> None:
        if not self._started:
            return

        self._started = False
        self._stop_event.set()
        self.stop_calls += 1


def make_runtime(
    *,
    input_count: int = 3,
) -> tuple[
    VoiceRuntime,
    FakeMicrophone,
    FakeSpeaker,
    FakeAgent,
]:
    agent = FakeAgent()

    inputs = tuple(
        VoiceInput(
            audio=b"\x01\x00" * 1600,
            sample_rate_hz=16_000,
            channels=1,
        )
        for _ in range(input_count)
    )

    microphone = FakeMicrophone(
        inputs
    )
    speaker = FakeSpeaker()

    session = VoiceSession(
        agent=agent,
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=FakeTTS(),
    )

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    return (
        runtime,
        microphone,
        speaker,
        agent,
    )


def make_blocking_runtime() -> tuple[
    VoiceRuntime,
    BlockingMicrophone,
    FakeSpeaker,
    FakeAgent,
]:
    agent = FakeAgent()
    microphone = BlockingMicrophone()
    speaker = FakeSpeaker()

    session = VoiceSession(
        agent=agent,
        vad=FakeVAD(),
        stt=FakeSTT(),
        tts=FakeTTS(),
    )

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    return (
        runtime,
        microphone,
        speaker,
        agent,
    )


def wait_for(
    predicate,
    *,
    timeout: float = 1.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)

    return predicate()


def test_continuous_runtime_starts_stopped() -> None:
    runtime, _, _, _ = make_runtime()

    continuous = ContinuousVoiceRuntime(
        runtime
    )

    assert continuous.state == (
        ContinuousVoiceRuntimeState.STOPPED
    )
    assert continuous.running is False


def test_continuous_runtime_processes_repeated_inputs() -> None:
    runtime, microphone, speaker, agent = make_runtime(
        input_count=20
    )

    results: list[str] = []

    continuous = ContinuousVoiceRuntime(
        runtime,
        on_result=lambda result: results.append(
            result.response.text
        ),
    )

    continuous.start()

    assert wait_for(
        lambda: len(agent.calls) >= 3
    )

    continuous.stop()

    assert len(agent.calls) >= 3
    assert results[:3] == [
        "Hello from OSA."
    ] * 3
    assert speaker.play_calls[:3] == [
        (
            b"speech",
            16_000,
            1,
        )
    ] * 3
    assert microphone.started is False
    assert continuous.state == (
        ContinuousVoiceRuntimeState.STOPPED
    )


def test_continuous_start_is_idempotent() -> None:
    runtime, microphone, _, _ = make_blocking_runtime()

    continuous = ContinuousVoiceRuntime(
        runtime
    )

    continuous.start()
    first_thread = continuous.thread

    assert first_thread is not None
    assert continuous.state == (
        ContinuousVoiceRuntimeState.RUNNING
    )

    continuous.start()

    assert continuous.thread is first_thread
    assert microphone.start_calls == 1
    assert continuous.state == (
        ContinuousVoiceRuntimeState.RUNNING
    )

    continuous.stop()


def test_continuous_runtime_is_restartable_after_stop() -> None:
    runtime, microphone, _, agent = make_runtime(
        input_count=20
    )

    continuous = create_continuous_voice_runtime(
        runtime
    )

    continuous.start()

    assert wait_for(
        lambda: len(agent.calls) >= 1
    )

    continuous.stop()

    assert continuous.state == (
        ContinuousVoiceRuntimeState.STOPPED
    )
    assert runtime.state == VoiceRuntimeState.STOPPED
    assert microphone.started is False

    continuous.start()

    assert wait_for(
        lambda: len(agent.calls) >= 2
    )

    continuous.stop()

    assert len(agent.calls) >= 2


def test_continuous_reset_clears_previous_result() -> None:
    runtime, _, _, agent = make_runtime(
        input_count=20
    )

    continuous = ContinuousVoiceRuntime(
        runtime
    )

    continuous.start()

    assert wait_for(
        lambda: bool(agent.calls)
    )

    continuous.stop()

    assert continuous.last_result is not None

    continuous.reset()

    assert continuous.state == (
        ContinuousVoiceRuntimeState.STOPPED
    )
    assert continuous.last_result is None
    assert continuous.last_error is None
    assert runtime.session.state == (
        VoiceSessionState.IDLE
    )


def test_continuous_interrupt_delegates_to_runtime() -> None:
    runtime, _, speaker, _ = make_blocking_runtime()

    continuous = ContinuousVoiceRuntime(
        runtime
    )

    continuous.start()

    assert wait_for(
        lambda: bool(speaker.play_calls)
    )

    continuous.interrupt()

    continuous.stop()

    assert speaker.stop_calls >= 1


def test_continuous_rejects_invalid_runtime() -> None:
    with pytest.raises(
        ValueError,
        match="runtime",
    ):
        ContinuousVoiceRuntime(
            None
        )


def test_continuous_rejects_invalid_callback() -> None:
    runtime, _, _, _ = make_runtime()

    with pytest.raises(
        TypeError,
        match="callable",
    ):
        ContinuousVoiceRuntime(
            runtime,
            on_result="not-callable",
        )


def test_continuous_config_validates_thread_name() -> None:
    with pytest.raises(
        ValueError,
        match="thread_name",
    ):
        ContinuousVoiceRuntimeConfig(
            thread_name="  "
        )


def test_continuous_config_validates_join_timeout() -> None:
    with pytest.raises(
        ValueError,
        match="join_timeout_seconds",
    ):
        ContinuousVoiceRuntimeConfig(
            join_timeout_seconds=-1
        )


def test_worker_error_is_exposed_and_runtime_stops() -> None:
    runtime, _, _, _ = make_runtime()

    class BrokenRuntime(VoiceRuntime):
        def run_once(self):
            raise RuntimeError(
                "worker boom"
            )

        def stop_capture(self):
            return None

    broken = BrokenRuntime(
        runtime.session,
        runtime.microphone,
        runtime.speaker,
    )

    continuous = ContinuousVoiceRuntime(
        broken
    )

    continuous.start()

    assert wait_for(
        lambda: (
            continuous.state
            == ContinuousVoiceRuntimeState.ERROR
        )
    )

    assert isinstance(
        continuous.last_error,
        Exception,
    )


def test_stop_before_start_is_safe() -> None:
    runtime, _, _, _ = make_runtime()

    continuous = ContinuousVoiceRuntime(
        runtime
    )

    continuous.stop()

    assert continuous.state == (
        ContinuousVoiceRuntimeState.STOPPED
    )


def test_runtime_stop_capture_is_reversible() -> None:
    runtime, microphone, _, _ = make_blocking_runtime()

    runtime.start()

    runtime.stop_capture()

    assert runtime.state == (
        VoiceRuntimeState.STOPPED
    )
    assert microphone.started is False

    runtime.start()

    assert runtime.state == (
        VoiceRuntimeState.READY
    )
    assert microphone.started is True

    runtime.stop_capture()
