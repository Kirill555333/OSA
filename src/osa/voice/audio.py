"""Platform-neutral audio I/O contracts for OSA 0.6.x."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from osa.voice.contracts import VoiceInput


class AudioInputError(
    RuntimeError
):
    """Raised when microphone input fails."""


class AudioOutputError(
    RuntimeError
):
    """Raised when speaker output fails."""


@runtime_checkable
class MicrophoneInput(Protocol):
    """
    Provider-neutral microphone interface.

    Concrete implementations own device access. VoiceSession only consumes
    normalized audio chunks and does not know which operating system or
    audio library produced them.
    """

    def start(self) -> None:
        ...

    def read(
        self,
    ) -> VoiceInput:
        ...

    def stop(self) -> None:
        ...


@runtime_checkable
class SpeakerOutput(Protocol):
    """
    Provider-neutral speaker interface.

    Concrete implementations own device playback. VoiceSession does not
    directly access operating-system audio APIs.
    """

    def play(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> None:
        ...

    def stop(self) -> None:
        ...


class FakeMicrophone:
    """Deterministic microphone implementation for unit tests."""

    def __init__(
        self,
        inputs: tuple[VoiceInput, ...] = (),
    ) -> None:
        self._inputs = tuple(inputs)
        self._index = 0
        self._started = False
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def started(self) -> bool:
        """Return whether the fake microphone is active."""
        return self._started

    @property
    def remaining(self) -> int:
        """Return the number of unread audio inputs."""
        return max(
            0,
            len(self._inputs) - self._index,
        )

    def start(self) -> None:
        if self._started:
            raise AudioInputError(
                "Microphone is already started."
            )

        self._started = True
        self.start_calls += 1

    def read(self) -> VoiceInput:
        if not self._started:
            raise AudioInputError(
                "Microphone is not started."
            )

        if self._index >= len(self._inputs):
            raise AudioInputError(
                "No microphone input is available."
            )

        result = self._inputs[self._index]
        self._index += 1

        return result

    def stop(self) -> None:
        if not self._started:
            return

        self._started = False
        self.stop_calls += 1


class FakeSpeaker:
    """Deterministic speaker implementation for unit tests."""

    def __init__(self) -> None:
        self._started = False
        self.play_calls: list[
            tuple[
                bytes,
                int,
                int,
            ]
        ] = []
        self.stop_calls = 0

    @property
    def started(self) -> bool:
        """Return whether the fake speaker has an active playback session."""
        return self._started

    def play(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> None:
        if not isinstance(
            audio,
            bytes,
        ):
            raise AudioOutputError(
                "audio must be bytes."
            )

        if not audio:
            raise AudioOutputError(
                "audio cannot be empty."
            )

        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(
                sample_rate_hz,
                int,
            )
        ):
            raise AudioOutputError(
                "sample_rate_hz must be an integer."
            )

        if sample_rate_hz <= 0:
            raise AudioOutputError(
                "sample_rate_hz must be greater than zero."
            )

        if (
            isinstance(channels, bool)
            or not isinstance(
                channels,
                int,
            )
        ):
            raise AudioOutputError(
                "channels must be an integer."
            )

        if channels <= 0:
            raise AudioOutputError(
                "channels must be greater than zero."
            )

        self._started = True

        self.play_calls.append(
            (
                audio,
                sample_rate_hz,
                channels,
            )
        )

        self._started = False

    def stop(self) -> None:
        self._started = False
        self.stop_calls += 1


def create_fake_microphone(
    *inputs: VoiceInput,
) -> FakeMicrophone:
    """Create a deterministic microphone loaded with test inputs."""
    return FakeMicrophone(
        tuple(inputs)
    )


def create_fake_speaker() -> FakeSpeaker:
    """Create a deterministic speaker for tests."""
    return FakeSpeaker()
