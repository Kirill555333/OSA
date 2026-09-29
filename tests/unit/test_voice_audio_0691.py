from __future__ import annotations

import pytest

from osa.voice.audio import (
    AudioInputError,
    AudioOutputError,
    FakeMicrophone,
    FakeSpeaker,
    MicrophoneInput,
    SpeakerOutput,
    create_fake_microphone,
    create_fake_speaker,
)
from osa.voice.contracts import VoiceInput


def _input() -> VoiceInput:
    return VoiceInput(
        audio=b"\x00\x00",
        sample_rate_hz=16000,
        channels=1,
    )


def test_fake_microphone_satisfies_protocol():
    microphone = FakeMicrophone(
        (_input(),)
    )

    assert isinstance(
        microphone,
        MicrophoneInput,
    )


def test_fake_speaker_satisfies_protocol():
    speaker = FakeSpeaker()

    assert isinstance(
        speaker,
        SpeakerOutput,
    )


def test_microphone_requires_start_before_read():
    microphone = FakeMicrophone(
        (_input(),)
    )

    with pytest.raises(
        AudioInputError,
        match="not started",
    ):
        microphone.read()


def test_microphone_reads_inputs_in_order():
    first = VoiceInput(
        audio=b"first",
        sample_rate_hz=16000,
    )
    second = VoiceInput(
        audio=b"second",
        sample_rate_hz=16000,
    )

    microphone = FakeMicrophone(
        (
            first,
            second,
        )
    )

    microphone.start()

    assert microphone.read() is first
    assert microphone.read() is second
    assert microphone.remaining == 0


def test_microphone_reports_exhausted_input():
    microphone = FakeMicrophone(
        (_input(),)
    )

    microphone.start()
    microphone.read()

    with pytest.raises(
        AudioInputError,
        match="No microphone input",
    ):
        microphone.read()


def test_microphone_start_and_stop_are_tracked():
    microphone = FakeMicrophone(
        (_input(),)
    )

    microphone.start()

    assert microphone.started is True
    assert microphone.start_calls == 1

    microphone.stop()

    assert microphone.started is False
    assert microphone.stop_calls == 1


def test_microphone_double_start_is_rejected():
    microphone = FakeMicrophone(
        (_input(),)
    )

    microphone.start()

    with pytest.raises(
        AudioInputError,
        match="already started",
    ):
        microphone.start()


def test_microphone_stop_is_idempotent():
    microphone = FakeMicrophone()

    microphone.stop()
    microphone.stop()

    assert microphone.stop_calls == 0


def test_speaker_records_playback():
    speaker = FakeSpeaker()

    speaker.play(
        b"speech",
        sample_rate_hz=16000,
        channels=1,
    )

    assert speaker.play_calls == [
        (
            b"speech",
            16000,
            1,
        )
    ]
    assert speaker.started is False


@pytest.mark.parametrize(
    "audio",
    [
        b"",
        "audio",
        bytearray(b"audio"),
        None,
    ],
)
def test_speaker_rejects_invalid_audio(
    audio,
):
    speaker = FakeSpeaker()

    with pytest.raises(
        AudioOutputError,
        match="audio",
    ):
        speaker.play(
            audio,
            sample_rate_hz=16000,
        )


@pytest.mark.parametrize(
    "sample_rate_hz",
    [
        0,
        -1,
        16000.0,
        True,
    ],
)
def test_speaker_rejects_invalid_sample_rate(
    sample_rate_hz,
):
    speaker = FakeSpeaker()

    with pytest.raises(
        AudioOutputError,
        match="sample_rate_hz",
    ):
        speaker.play(
            b"audio",
            sample_rate_hz=sample_rate_hz,
        )


@pytest.mark.parametrize(
    "channels",
    [
        0,
        -1,
        1.0,
        True,
    ],
)
def test_speaker_rejects_invalid_channels(
    channels,
):
    speaker = FakeSpeaker()

    with pytest.raises(
        AudioOutputError,
        match="channels",
    ):
        speaker.play(
            b"audio",
            sample_rate_hz=16000,
            channels=channels,
        )


def test_speaker_stop_is_safe_when_idle():
    speaker = FakeSpeaker()

    speaker.stop()

    assert speaker.stop_calls == 1
    assert speaker.started is False


def test_factory_helpers_create_expected_types():
    microphone = create_fake_microphone(
        _input()
    )
    speaker = create_fake_speaker()

    assert isinstance(
        microphone,
        FakeMicrophone,
    )
    assert isinstance(
        speaker,
        FakeSpeaker,
    )


def test_audio_errors_are_runtime_errors():
    assert issubclass(
        AudioInputError,
        RuntimeError,
    )
    assert issubclass(
        AudioOutputError,
        RuntimeError,
    )
