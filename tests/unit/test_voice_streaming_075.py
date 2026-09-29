from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceTranscript,
)
from osa.voice.streaming import (
    BufferedSpeechToTextStream,
    ChunkedTextToSpeechStream,
    VoiceStreamChunk,
    VoiceStreamingError,
    VoiceStreamingState,
    create_buffered_stt_stream,
    create_chunked_tts_stream,
)


@dataclass(frozen=True)
class FakeResponse:
    content: str


class FakeSTT:
    def __init__(self) -> None:
        self.calls: list[VoiceInput] = []

    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        self.calls.append(voice_input)

        return VoiceTranscript(
            text="hello OSA",
            language="en",
        )


class FakeTTS:
    def __init__(
        self,
        audio: bytes = b"0123456789",
    ) -> None:
        self.audio = audio
        self.calls: list[VoiceOutput] = []
        self.stop_calls = 0

    def synthesize(
        self,
        output: VoiceOutput,
    ) -> bytes:
        self.calls.append(output)
        return self.audio

    def stop(self) -> None:
        self.stop_calls += 1


def make_input(
    payload: bytes,
    *,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
) -> VoiceInput:
    return VoiceInput(
        audio=payload,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def make_chunk(
    sequence: int,
    payload: bytes,
    *,
    is_final: bool = False,
) -> VoiceStreamChunk:
    return VoiceStreamChunk(
        voice_input=make_input(payload),
        sequence=sequence,
        is_final=is_final,
    )


def test_voice_stream_chunk_validates_sequence() -> None:
    with pytest.raises(
        VoiceStreamingError,
        match="sequence cannot be negative",
    ):
        VoiceStreamChunk(
            voice_input=make_input(b"12"),
            sequence=-1,
        )


def test_buffered_stt_starts_idle() -> None:
    stream = BufferedSpeechToTextStream(
        FakeSTT()
    )

    assert stream.state == VoiceStreamingState.IDLE
    assert stream.chunk_count == 0


def test_buffered_stt_accumulates_and_finishes() -> None:
    stt = FakeSTT()
    stream = create_buffered_stt_stream(stt)

    stream.start()

    assert stream.state == VoiceStreamingState.ACTIVE

    assert stream.push(
        make_chunk(
            0,
            b"ab",
        )
    ) is None

    assert stream.push(
        make_chunk(
            1,
            b"cd",
        )
    ) is None

    assert stream.chunk_count == 2

    transcript = stream.finish()

    assert transcript.text == "hello OSA"
    assert transcript.language == "en"
    assert stream.state == VoiceStreamingState.STOPPED

    assert stt.calls == [
        make_input(b"abcd")
    ]


def test_buffered_stt_final_chunk_finishes_stream() -> None:
    stt = FakeSTT()
    stream = create_buffered_stt_stream(stt)

    stream.start()

    transcript = stream.push(
        make_chunk(
            0,
            b"final",
            is_final=True,
        )
    )

    assert transcript is not None
    assert transcript.text == "hello OSA"
    assert stream.state == VoiceStreamingState.STOPPED


def test_buffered_stt_rejects_wrong_sequence() -> None:
    stream = create_buffered_stt_stream(
        FakeSTT()
    )

    stream.start()

    with pytest.raises(
        VoiceStreamingError,
        match="Unexpected chunk sequence",
    ):
        stream.push(
            make_chunk(
                1,
                b"ab",
            )
        )


def test_buffered_stt_rejects_mixed_formats() -> None:
    stream = create_buffered_stt_stream(
        FakeSTT()
    )

    stream.start()
    stream.push(
        make_chunk(
            0,
            b"ab",
        )
    )

    with pytest.raises(
        VoiceStreamingError,
        match="same sample rate",
    ):
        stream.push(
            VoiceStreamChunk(
                voice_input=make_input(
                    b"cd",
                    sample_rate_hz=8_000,
                ),
                sequence=1,
            )
        )


def test_buffered_stt_rejects_push_before_start() -> None:
    stream = create_buffered_stt_stream(
        FakeSTT()
    )

    with pytest.raises(
        VoiceStreamingError,
        match="must be ACTIVE",
    ):
        stream.push(
            make_chunk(
                0,
                b"ab",
            )
        )


def test_buffered_stt_rejects_empty_finish() -> None:
    stream = create_buffered_stt_stream(
        FakeSTT()
    )

    stream.start()

    with pytest.raises(
        VoiceStreamingError,
        match="empty STT stream",
    ):
        stream.finish()


def test_buffered_stt_reset_clears_stream() -> None:
    stream = create_buffered_stt_stream(
        FakeSTT()
    )

    stream.start()
    stream.push(
        make_chunk(
            0,
            b"ab",
        )
    )

    stream.reset()

    assert stream.state == VoiceStreamingState.IDLE
    assert stream.chunk_count == 0


def test_buffered_stt_stop_discards_buffer() -> None:
    stream = create_buffered_stt_stream(
        FakeSTT()
    )

    stream.start()
    stream.push(
        make_chunk(
            0,
            b"ab",
        )
    )

    stream.stop()

    assert stream.state == VoiceStreamingState.STOPPED
    assert stream.chunk_count == 0


def test_buffered_stt_wraps_provider_errors() -> None:
    class BrokenSTT:
        def transcribe(
            self,
            voice_input: VoiceInput,
        ) -> VoiceTranscript:
            raise RuntimeError("boom")

    stream = create_buffered_stt_stream(
        BrokenSTT()
    )

    stream.start()
    stream.push(
        make_chunk(
            0,
            b"ab",
        )
    )

    with pytest.raises(
        VoiceStreamingError,
        match="transcription failed",
    ):
        stream.finish()

    assert stream.state == VoiceStreamingState.STOPPED


def test_chunked_tts_streams_fixed_size_chunks() -> None:
    tts = FakeTTS(
        audio=b"0123456789"
    )

    stream = create_chunked_tts_stream(
        tts,
        chunk_size_bytes=4,
    )

    chunks = list(
        stream.stream(
            VoiceOutput(
                text="Hello",
            )
        )
    )

    assert chunks == [
        b"0123",
        b"4567",
        b"89",
    ]
    assert stream.state == VoiceStreamingState.STOPPED
    assert tts.calls == [
        VoiceOutput(
            text="Hello"
        )
    ]


def test_chunked_tts_stream_validates_chunk_size() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_size_bytes",
    ):
        ChunkedTextToSpeechStream(
            FakeTTS(),
            chunk_size_bytes=0,
        )


def test_chunked_tts_stream_rejects_concurrent_stream() -> None:
    tts = FakeTTS()
    stream = create_chunked_tts_stream(
        tts,
        chunk_size_bytes=2,
    )

    iterator = stream.stream(
        VoiceOutput(
            text="Hello"
        )
    )

    next(iterator)

    with pytest.raises(
        VoiceStreamingError,
        match="already active",
    ):
        list(
            stream.stream(
                VoiceOutput(
                    text="Second"
                )
            )
        )

    list(iterator)


def test_chunked_tts_stop_calls_wrapped_provider() -> None:
    tts = FakeTTS(
        audio=b"0123456789"
    )

    stream = create_chunked_tts_stream(
        tts,
        chunk_size_bytes=2,
    )

    iterator = stream.stream(
        VoiceOutput(
            text="Hello"
        )
    )

    next(iterator)
    stream.stop()

    assert tts.stop_calls == 1
    assert stream.state == VoiceStreamingState.STOPPED
    assert list(iterator) == []


def test_chunked_tts_reset_returns_to_idle() -> None:
    stream = create_chunked_tts_stream(
        FakeTTS()
    )

    list(
        stream.stream(
            VoiceOutput(
                text="Hello"
            )
        )
    )

    stream.reset()

    assert stream.state == VoiceStreamingState.IDLE


def test_chunked_tts_rejects_invalid_output() -> None:
    stream = create_chunked_tts_stream(
        FakeTTS()
    )

    with pytest.raises(
        TypeError,
        match="VoiceOutput",
    ):
        list(
            stream.stream(
                object()
            )
        )


def test_chunked_tts_wraps_provider_errors() -> None:
    class BrokenTTS:
        def synthesize(
            self,
            output: VoiceOutput,
        ) -> bytes:
            raise RuntimeError("boom")

        def stop(self) -> None:
            return None

    stream = create_chunked_tts_stream(
        BrokenTTS()
    )

    with pytest.raises(
        VoiceStreamingError,
        match="synthesis failed",
    ):
        list(
            stream.stream(
                VoiceOutput(
                    text="Hello"
                )
            )
        )


def test_streaming_protocols_accept_adapters() -> None:
    stt_stream = BufferedSpeechToTextStream(
        FakeSTT()
    )
    tts_stream = ChunkedTextToSpeechStream(
        FakeTTS()
    )

    from osa.voice.streaming import (
        StreamingSpeechToText,
        StreamingTextToSpeech,
    )

    assert isinstance(
        stt_stream,
        StreamingSpeechToText,
    )
    assert isinstance(
        tts_stream,
        StreamingTextToSpeech,
    )
