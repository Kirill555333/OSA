from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from osa.voice.audio import FakeMicrophone
from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceTranscript,
    VoiceSessionState,
)
from osa.voice.runtime import (
    VoiceRuntime,
    VoiceRuntimeConfig,
    VoiceRuntimeState,
)
from osa.voice.session import (
    VoiceSession,
    VoiceSessionResult,
)


@dataclass(frozen=True)
class Response:
    content: str


class Agent:
    def chat(self, user_input: str) -> Response:
        return Response(
            content="Hello from OSA."
        )


class VAD:
    def __init__(self) -> None:
        self.calls = 0

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        self.calls += 1
        return True

    def reset(self) -> None:
        return None


class STT:
    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        return VoiceTranscript(
            text="hello",
            language="en",
        )


class TTS:
    def __init__(self) -> None:
        self.stop_calls = 0

    def synthesize(
        self,
        output,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        self.stop_calls += 1


class BlockingSpeaker:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.play_calls = 0
        self.stop_calls = 0

    def play(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> None:
        self.play_calls += 1
        self.started.set()
        self.stopped.wait(1.0)

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


def make_runtime() -> tuple[
    VoiceRuntime,
    BlockingSpeaker,
    TTS,
]:
    tts = TTS()
    session = VoiceSession(
        agent=Agent(),
        vad=VAD(),
        stt=STT(),
        tts=tts,
    )

    microphone = FakeMicrophone(
        (
            VoiceInput(
                audio=b"\x01\x00" * 1600,
                sample_rate_hz=16_000,
                channels=1,
            ),
            VoiceInput(
                audio=b"\x01\x00" * 1600,
                sample_rate_hz=16_000,
                channels=1,
            ),
            VoiceInput(
                audio=b"\x01\x00" * 1600,
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
    )

    speaker = BlockingSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
        config=VoiceRuntimeConfig(
            barge_in_enabled=True,
            barge_in_poll_interval_seconds=0.005,
        ),
    )

    return runtime, speaker, tts


def test_barge_in_monitor_is_created_when_enabled() -> None:
    runtime, _, _ = make_runtime()

    assert runtime.barge_in_monitor is not None


def test_speak_completes_normally_without_barge_in() -> None:
    tts = TTS()

    class SilentVAD:
        def detect(
            self,
            audio: bytes,
            *,
            sample_rate_hz: int,
            channels: int = 1,
        ) -> bool:
            return False

        def reset(self) -> None:
            return None

    session = VoiceSession(
        agent=Agent(),
        vad=SilentVAD(),
        stt=STT(),
        tts=tts,
    )

    microphone = FakeMicrophone(
        (
            VoiceInput(
                audio=b"\x01\x00" * 1600,
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
    )

    speaker = BlockingSpeaker()

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
        config=VoiceRuntimeConfig(
            barge_in_enabled=True,
            barge_in_poll_interval_seconds=0.005,
        ),
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is None


def test_barge_in_interrupts_speaking() -> None:
    runtime, speaker, tts = make_runtime()

    runtime.start()

    result = runtime.listen_once()

    assert result is not None
    assert runtime.session.state == (
        VoiceSessionState.SPEAKING
    )

    playback_thread = threading.Thread(
        target=runtime.speak,
        args=(result,),
        daemon=True,
    )

    playback_thread.start()

    assert speaker.started.wait(1.0)

    deadline = time.monotonic() + 1.0

    while (
        tts.stop_calls == 0
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)

    assert tts.stop_calls >= 1
    assert speaker.stop_calls >= 1

    playback_thread.join(1.0)

    assert playback_thread.is_alive() is False
    assert runtime.session.state == (
        VoiceSessionState.IDLE
    )
    assert runtime.state == (
        VoiceRuntimeState.READY
    )
