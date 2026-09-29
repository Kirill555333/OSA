from __future__ import annotations

import io
import wave

from osa.voice.audio import FakeMicrophone, FakeSpeaker
from osa.voice.contracts import VoiceInput, VoiceOutput, VoiceTranscript
from osa.voice.runtime import VoiceRuntime
from osa.voice.session import VoiceSession


class Response:
    def __init__(self, content: str) -> None:
        self.content = content


class Agent:
    def chat(self, user_input: str) -> Response:
        return Response("Hello from OSA.")


class CapturingVAD:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bytes]] = []

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        self.calls.append(
            (
                sample_rate_hz,
                channels,
                audio,
            )
        )
        return True

    def reset(self) -> None:
        return None


class CapturingSTT:
    def __init__(self) -> None:
        self.inputs: list[VoiceInput] = []

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        self.inputs.append(voice_input)
        return VoiceTranscript(
            text="hello",
            language="en",
        )


class WavTTS:
    def synthesize(
        self,
        output: VoiceOutput,
    ) -> bytes:
        buffer = io.BytesIO()

        with wave.open(
            buffer,
            "wb",
        ) as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(
                b"\x64\x00\x9c\xff"
            )

        return buffer.getvalue()

    def stop(self) -> None:
        return None


def test_runtime_normalizes_microphone_before_vad_and_stt() -> None:
    microphone = FakeMicrophone(
        (
            VoiceInput(
                audio=(
                    b"\xe8\x03"
                    b"\xb8\x0b"
                    b"\x18\xfc"
                    b"\x48\xf4"
                ),
                sample_rate_hz=8_000,
                channels=2,
            ),
        )
    )

    vad = CapturingVAD()
    stt = CapturingSTT()

    session = VoiceSession(
        agent=Agent(),
        vad=vad,
        stt=stt,
        tts=WavTTS(),
    )

    runtime = VoiceRuntime(
        session,
        microphone,
        FakeSpeaker(),
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None

    assert vad.calls[0][0] == 16_000
    assert vad.calls[0][1] == 1

    assert stt.inputs[0].sample_rate_hz == 16_000
    assert stt.inputs[0].channels == 1
    assert stt.inputs[0].encoding == "pcm_s16le"


def test_runtime_unwraps_and_normalizes_piper_wav_output() -> None:
    microphone = FakeMicrophone(
        (
            VoiceInput(
                audio=b"\x01\x00" * 1600,
                sample_rate_hz=16_000,
                channels=1,
            ),
        )
    )

    speaker = FakeSpeaker()

    session = VoiceSession(
        agent=Agent(),
        vad=CapturingVAD(),
        stt=CapturingSTT(),
        tts=WavTTS(),
    )

    runtime = VoiceRuntime(
        session,
        microphone,
        speaker,
    )

    runtime.start()

    result = runtime.listen_once()

    assert result is not None

    runtime.speak(result)

    assert speaker.play_calls[0][1] == 16_000
    assert speaker.play_calls[0][2] == 1
    assert not speaker.play_calls[0][0].startswith(b"RIFF")
