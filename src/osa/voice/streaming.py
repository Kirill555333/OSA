"""Provider-neutral streaming contracts for OSA 0.7.5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Event
from typing import Iterator, Protocol, runtime_checkable

from osa.voice.contracts import VoiceInput, VoiceOutput, VoiceTranscript
from osa.voice.stt import SpeechToText
from osa.voice.tts import TextToSpeech


class VoiceStreamingError(
    RuntimeError
):
    """Raised when a streaming voice operation fails."""


class VoiceStreamingState(str, Enum):
    """Lifecycle state for a streaming adapter."""

    IDLE = "idle"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True)
class VoiceStreamChunk:
    """One immutable input chunk in a streaming STT session."""

    voice_input: VoiceInput
    sequence: int
    is_final: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.voice_input,
            VoiceInput,
        ):
            raise VoiceStreamingError(
                "voice_input must be a VoiceInput."
            )

        if (
            isinstance(self.sequence, bool)
            or not isinstance(
                self.sequence,
                int,
            )
        ):
            raise VoiceStreamingError(
                "sequence must be an integer."
            )

        if self.sequence < 0:
            raise VoiceStreamingError(
                "sequence cannot be negative."
            )

        if not isinstance(
            self.is_final,
            bool,
        ):
            raise VoiceStreamingError(
                "is_final must be boolean."
            )


@runtime_checkable
class StreamingSpeechToText(Protocol):
    """Provider-neutral streaming speech-to-text interface."""

    @property
    def state(self) -> VoiceStreamingState:
        ...

    def start(self) -> None:
        ...

    def push(
        self,
        chunk: VoiceStreamChunk,
    ) -> VoiceTranscript | None:
        ...

    def finish(self) -> VoiceTranscript:
        ...

    def stop(self) -> None:
        ...

    def reset(self) -> None:
        ...


@runtime_checkable
class StreamingTextToSpeech(Protocol):
    """Provider-neutral streaming text-to-speech interface."""

    @property
    def state(self) -> VoiceStreamingState:
        ...

    def stream(
        self,
        output: VoiceOutput,
    ) -> Iterator[bytes]:
        ...

    def stop(self) -> None:
        ...

    def reset(self) -> None:
        ...


class BufferedSpeechToTextStream:
    """
    Streaming-compatible STT adapter backed by an existing SpeechToText.

    Audio chunks are collected during the stream and submitted once at
    finish(). This is deliberately not real-time transcription; it provides
    the stable streaming contract needed by later runtime work.
    """

    def __init__(
        self,
        stt: SpeechToText,
    ) -> None:
        if stt is None:
            raise ValueError(
                "stt is required."
            )

        if not isinstance(
            stt,
            SpeechToText,
        ):
            raise TypeError(
                "stt must provide transcribe()."
            )

        self._stt = stt
        self._state = VoiceStreamingState.IDLE
        self._chunks: list[VoiceInput] = []
        self._next_sequence = 0
        self._sample_rate_hz: int | None = None
        self._channels: int | None = None
        self._encoding: str | None = None

    @property
    def state(self) -> VoiceStreamingState:
        """Return the current stream state."""
        return self._state

    @property
    def stt(self) -> SpeechToText:
        """Return the wrapped speech-to-text provider."""
        return self._stt

    @property
    def chunk_count(self) -> int:
        """Return the number of accepted audio chunks."""
        return len(self._chunks)

    def start(self) -> None:
        """Begin a new buffered transcription stream."""
        if self._state == VoiceStreamingState.ACTIVE:
            return

        if self._state == VoiceStreamingState.STOPPING:
            raise VoiceStreamingError(
                "Cannot start while the STT stream is stopping."
            )

        self._clear_buffer()
        self._state = VoiceStreamingState.ACTIVE

    def push(
        self,
        chunk: VoiceStreamChunk,
    ) -> VoiceTranscript | None:
        """
        Accept one audio chunk.

        Buffered STT returns None until finish(). A final chunk automatically
        completes the stream and returns its transcript.
        """
        self._require_active()

        if not isinstance(
            chunk,
            VoiceStreamChunk,
        ):
            raise TypeError(
                "chunk must be a VoiceStreamChunk."
            )

        if chunk.sequence != self._next_sequence:
            raise VoiceStreamingError(
                (
                    "Unexpected chunk sequence "
                    f"{chunk.sequence}; expected "
                    f"{self._next_sequence}."
                )
            )

        self._validate_format(
            chunk.voice_input
        )

        self._chunks.append(
            chunk.voice_input
        )
        self._next_sequence += 1

        if chunk.is_final:
            return self.finish()

        return None

    def finish(self) -> VoiceTranscript:
        """Submit all buffered audio to the wrapped STT provider."""
        self._require_active()

        if not self._chunks:
            raise VoiceStreamingError(
                "Cannot finish an empty STT stream."
            )

        combined = b"".join(
            item.audio
            for item in self._chunks
        )

        voice_input = VoiceInput(
            audio=combined,
            sample_rate_hz=self._sample_rate_hz or 16_000,
            channels=self._channels or 1,
            encoding=self._encoding or "pcm_s16le",
        )

        try:
            transcript = self._stt.transcribe(
                voice_input
            )
        except Exception as exc:
            self._state = VoiceStreamingState.STOPPED
            raise VoiceStreamingError(
                f"Buffered STT transcription failed: {exc}"
            ) from exc

        if not isinstance(
            transcript,
            VoiceTranscript,
        ):
            self._state = VoiceStreamingState.STOPPED
            raise VoiceStreamingError(
                "STT provider returned an invalid VoiceTranscript."
            )

        self._state = VoiceStreamingState.STOPPED

        return transcript

    def stop(self) -> None:
        """Stop the current stream without transcribing it."""
        if self._state == VoiceStreamingState.STOPPED:
            return

        self._state = VoiceStreamingState.STOPPING
        self._clear_buffer()
        self._state = VoiceStreamingState.STOPPED

    def reset(self) -> None:
        """Reset the stream to IDLE."""
        self._clear_buffer()
        self._state = VoiceStreamingState.IDLE

    def _validate_format(
        self,
        voice_input: VoiceInput,
    ) -> None:
        if self._sample_rate_hz is None:
            self._sample_rate_hz = voice_input.sample_rate_hz
            self._channels = voice_input.channels
            self._encoding = voice_input.encoding
            return

        if (
            voice_input.sample_rate_hz
            != self._sample_rate_hz
        ):
            raise VoiceStreamingError(
                "All STT chunks must use the same sample rate."
            )

        if (
            voice_input.channels
            != self._channels
        ):
            raise VoiceStreamingError(
                "All STT chunks must use the same channel count."
            )

        if (
            voice_input.encoding
            != self._encoding
        ):
            raise VoiceStreamingError(
                "All STT chunks must use the same encoding."
            )

    def _require_active(self) -> None:
        if self._state != VoiceStreamingState.ACTIVE:
            raise VoiceStreamingError(
                (
                    "STT stream must be ACTIVE; current state is "
                    f"'{self._state.value}'."
                )
            )

    def _clear_buffer(self) -> None:
        self._chunks.clear()
        self._next_sequence = 0
        self._sample_rate_hz = None
        self._channels = None
        self._encoding = None


class ChunkedTextToSpeechStream:
    """
    Streaming-compatible TTS adapter backed by an existing TextToSpeech.

    The wrapped TTS still synthesizes one complete response. The resulting
    bytes are then exposed incrementally in fixed-size chunks. Later provider
    implementations can replace this without changing the stream contract.
    """

    def __init__(
        self,
        tts: TextToSpeech,
        *,
        chunk_size_bytes: int = 4096,
    ) -> None:
        if tts is None:
            raise ValueError(
                "tts is required."
            )

        if not isinstance(
            tts,
            TextToSpeech,
        ):
            raise TypeError(
                "tts must provide synthesize() and stop()."
            )

        if (
            isinstance(
                chunk_size_bytes,
                bool,
            )
            or not isinstance(
                chunk_size_bytes,
                int,
            )
        ):
            raise ValueError(
                "chunk_size_bytes must be an integer."
            )

        if chunk_size_bytes <= 0:
            raise ValueError(
                "chunk_size_bytes must be greater than zero."
            )

        self._tts = tts
        self._chunk_size_bytes = chunk_size_bytes
        self._state = VoiceStreamingState.IDLE
        self._stop_requested = Event()

    @property
    def state(self) -> VoiceStreamingState:
        """Return the current stream state."""
        return self._state

    @property
    def tts(self) -> TextToSpeech:
        """Return the wrapped text-to-speech provider."""
        return self._tts

    @property
    def chunk_size_bytes(self) -> int:
        """Return the configured output chunk size."""
        return self._chunk_size_bytes

    def stream(
        self,
        output: VoiceOutput,
    ) -> Iterator[bytes]:
        """Synthesize once and expose the result as incremental byte chunks."""
        if not isinstance(
            output,
            VoiceOutput,
        ):
            raise TypeError(
                "output must be a VoiceOutput."
            )

        if self._state == VoiceStreamingState.ACTIVE:
            raise VoiceStreamingError(
                "A TTS stream is already active."
            )

        self._stop_requested.clear()
        self._state = VoiceStreamingState.ACTIVE

        try:
            audio = self._tts.synthesize(
                output
            )

            if not isinstance(
                audio,
                bytes,
            ):
                raise VoiceStreamingError(
                    "TTS provider returned an invalid audio payload."
                )

            if not audio:
                raise VoiceStreamingError(
                    "TTS provider returned empty audio."
                )

            for start in range(
                0,
                len(audio),
                self._chunk_size_bytes,
            ):
                if self._stop_requested.is_set():
                    self._state = VoiceStreamingState.STOPPED
                    return

                yield audio[
                    start : start + self._chunk_size_bytes
                ]

            self._state = VoiceStreamingState.STOPPED

        except VoiceStreamingError:
            self._state = VoiceStreamingState.STOPPED
            raise
        except Exception as exc:
            self._state = VoiceStreamingState.STOPPED
            raise VoiceStreamingError(
                f"Chunked TTS synthesis failed: {exc}"
            ) from exc

    def stop(self) -> None:
        """Request stream cancellation and stop the wrapped TTS provider."""
        if self._state == VoiceStreamingState.STOPPED:
            return

        self._state = VoiceStreamingState.STOPPING
        self._stop_requested.set()

        try:
            self._tts.stop()
        except Exception as exc:
            self._state = VoiceStreamingState.STOPPED
            raise VoiceStreamingError(
                f"TTS stop failed: {exc}"
            ) from exc

        self._state = VoiceStreamingState.STOPPED

    def reset(self) -> None:
        """Reset the stream to IDLE."""
        self._stop_requested.clear()
        self._state = VoiceStreamingState.IDLE


def create_buffered_stt_stream(
    stt: SpeechToText,
) -> BufferedSpeechToTextStream:
    """Create a buffered streaming STT adapter."""
    return BufferedSpeechToTextStream(
        stt
    )


def create_chunked_tts_stream(
    tts: TextToSpeech,
    *,
    chunk_size_bytes: int = 4096,
) -> ChunkedTextToSpeechStream:
    """Create a chunked streaming TTS adapter."""
    return ChunkedTextToSpeechStream(
        tts,
        chunk_size_bytes=chunk_size_bytes,
    )
