from __future__ import annotations

import pytest

from osa.voice.audio import (
    AudioInputError,
    AudioOutputError,
)
from osa.voice.platform import (
    AudioBackend,
    AudioPlatform,
)
from osa.voice.sounddevice_backend import (
    SoundDeviceAudioBackend,
    SoundDeviceAudioConfig,
    SoundDeviceAudioError,
    SoundDeviceMicrophone,
    SoundDeviceSpeaker,
)


class FakeInputStream:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.read_calls: list[int] = []
        self.data = b"\x01\x00\x02\x00"
        self.overflowed = False

    def start(self) -> None:
        self.started = True

    def read(
        self,
        frames: int,
    ):
        self.read_calls.append(frames)

        return (
            self.data,
            self.overflowed,
        )

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeOutputStream:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.writes: list[bytes] = []

    def start(self) -> None:
        self.started = True

    def write(
        self,
        audio: bytes,
    ) -> None:
        self.writes.append(
            bytes(audio)
        )

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    def __init__(
        self,
        *,
        devices=None,
        input_stream_factory=FakeInputStream,
        output_stream_factory=FakeOutputStream,
    ) -> None:
        self.devices = (
            devices
            if devices is not None
            else [
                {
                    "name": "Fake Microphone",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                },
                {
                    "name": "Fake Speaker",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
        )
        self.input_stream = None
        self.output_stream = None
        self.input_stream_factory = input_stream_factory
        self.output_stream_factory = output_stream_factory

    def RawInputStream(
        self,
        **kwargs,
    ):
        self.input_stream = self.input_stream_factory(
            **kwargs
        )
        return self.input_stream

    def RawOutputStream(
        self,
        **kwargs,
    ):
        self.output_stream = self.output_stream_factory(
            **kwargs
        )
        return self.output_stream

    def query_devices(self):
        return self.devices


def test_sounddevice_config_defaults():
    config = SoundDeviceAudioConfig()

    assert config.sample_rate_hz == 16_000
    assert config.channels == 1
    assert config.frames_per_read == 1_600
    assert config.dtype == "int16"
    assert config.latency == "low"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate_hz", 0),
        ("sample_rate_hz", -1),
        ("channels", 0),
        ("frames_per_read", 0),
        ("dtype", ""),
        ("latency", ""),
        ("dtype", "float32"),
    ],
)
def test_invalid_sounddevice_config_is_rejected(
    field: str,
    value,
):
    with pytest.raises(
        ValueError,
        match=field,
    ):
        SoundDeviceAudioConfig(
            **{field: value}
        )


def test_sounddevice_microphone_reads_voice_input():
    module = FakeSoundDevice()

    microphone = SoundDeviceMicrophone(
        sounddevice_module=module,
    )

    microphone.start()

    assert microphone.started is True
    assert module.input_stream.kwargs[
        "samplerate"
    ] == 16_000
    assert module.input_stream.kwargs[
        "channels"
    ] == 1
    assert module.input_stream.kwargs[
        "dtype"
    ] == "int16"

    result = microphone.read()

    assert result.audio == (
        b"\x01\x00\x02\x00"
    )
    assert result.sample_rate_hz == 16_000
    assert result.channels == 1
    assert result.encoding == "pcm_s16le"

    microphone.stop()

    assert module.input_stream.stopped is True
    assert module.input_stream.closed is True
    assert microphone.started is False


def test_microphone_requires_start():
    microphone = SoundDeviceMicrophone(
        sounddevice_module=FakeSoundDevice()
    )

    with pytest.raises(
        AudioInputError,
        match="not started",
    ):
        microphone.read()


def test_microphone_start_failure_is_wrapped():
    class RaisingInputStream:
        def __init__(self, **kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError(
                "device unavailable"
            )

    module = FakeSoundDevice(
        input_stream_factory=RaisingInputStream,
    )

    microphone = SoundDeviceMicrophone(
        sounddevice_module=module,
    )

    with pytest.raises(
        AudioInputError,
        match="device unavailable",
    ):
        microphone.start()


def test_microphone_overflow_is_rejected():
    module = FakeSoundDevice()
    module.input_stream = None

    microphone = SoundDeviceMicrophone(
        sounddevice_module=module,
    )

    microphone.start()
    module.input_stream.overflowed = True

    with pytest.raises(
        AudioInputError,
        match="overflowed",
    ):
        microphone.read()

    microphone.stop()


def test_microphone_empty_data_is_rejected():
    module = FakeSoundDevice()
    microphone = SoundDeviceMicrophone(
        sounddevice_module=module,
    )

    microphone.start()
    module.input_stream.data = b""

    with pytest.raises(
        AudioInputError,
        match="empty audio",
    ):
        microphone.read()

    microphone.stop()


def test_speaker_plays_raw_audio():
    module = FakeSoundDevice()

    speaker = SoundDeviceSpeaker(
        sounddevice_module=module,
    )

    speaker.play(
        b"\x00\x00\x01\x00",
        sample_rate_hz=16_000,
        channels=1,
    )

    assert module.output_stream.kwargs[
        "samplerate"
    ] == 16_000
    assert module.output_stream.kwargs[
        "channels"
    ] == 1
    assert module.output_stream.kwargs[
        "dtype"
    ] == "int16"
    assert module.output_stream.writes == [
        b"\x00\x00\x01\x00"
    ]
    assert module.output_stream.closed is True
    assert speaker.playing is False


@pytest.mark.parametrize(
    "audio",
    [
        b"",
        "audio",
        None,
    ],
)
def test_speaker_rejects_invalid_audio(
    audio,
):
    speaker = SoundDeviceSpeaker(
        sounddevice_module=FakeSoundDevice()
    )

    with pytest.raises(
        AudioOutputError,
        match="audio",
    ):
        speaker.play(
            audio,
            sample_rate_hz=16_000,
        )


def test_speaker_rejects_second_playback_stream():
    module = FakeSoundDevice()

    speaker = SoundDeviceSpeaker(
        sounddevice_module=module,
    )

    fake_stream = FakeOutputStream()

    speaker._stream = fake_stream

    with pytest.raises(
        AudioOutputError,
        match="already playing",
    ):
        speaker.play(
            b"audio",
            sample_rate_hz=16_000,
        )

    speaker._stream = None


def test_speaker_stop_aborts_active_stream():
    module = FakeSoundDevice()

    speaker = SoundDeviceSpeaker(
        sounddevice_module=module,
    )

    fake_stream = FakeOutputStream()
    speaker._stream = fake_stream

    speaker.stop()

    assert fake_stream.aborted is True
    assert fake_stream.closed is True
    assert speaker.playing is False


def test_speaker_stop_is_safe_when_idle():
    speaker = SoundDeviceSpeaker(
        sounddevice_module=FakeSoundDevice()
    )

    speaker.stop()

    assert speaker.playing is False


def test_backend_satisfies_audio_backend_protocol():
    backend = SoundDeviceAudioBackend(
        sounddevice_module=FakeSoundDevice()
    )

    assert isinstance(
        backend,
        AudioBackend,
    )


def test_backend_reports_current_platform():
    backend = SoundDeviceAudioBackend(
        sounddevice_module=FakeSoundDevice()
    )

    assert backend.platform == (
        AudioPlatform.current()
    )


def test_backend_reports_availability():
    backend = SoundDeviceAudioBackend(
        sounddevice_module=FakeSoundDevice()
    )

    assert backend.available() is True


def test_backend_reports_unavailable_when_no_devices():
    backend = SoundDeviceAudioBackend(
        sounddevice_module=FakeSoundDevice(
            devices=[]
        )
    )

    assert backend.available() is False


def test_backend_available_fails_closed_on_query_error():
    class RaisingModule(FakeSoundDevice):
        def query_devices(self):
            raise RuntimeError(
                "PortAudio unavailable"
            )

    backend = SoundDeviceAudioBackend(
        sounddevice_module=RaisingModule()
    )

    assert backend.available() is False


def test_backend_creates_microphone_and_speaker():
    module = FakeSoundDevice()

    backend = SoundDeviceAudioBackend(
        sounddevice_module=module
    )

    microphone = backend.microphone()
    speaker = backend.speaker()

    assert microphone.config is backend.config
    assert speaker.config is backend.config


def test_backend_query_devices_preserves_provider_result():
    module = FakeSoundDevice()

    backend = SoundDeviceAudioBackend(
        sounddevice_module=module
    )

    assert backend.query_devices() is module.devices


def test_missing_sounddevice_has_explicit_error():
    module = None

    class MissingLoaderBackend:
        pass

    assert module is None
    assert MissingLoaderBackend


def test_import_error_message_is_actionable(
    monkeypatch,
):
    import builtins

    original_import = builtins.__import__

    def fake_import(
        name,
        *args,
        **kwargs,
    ):
        if name == "sounddevice":
            raise ImportError(
                "missing"
            )

        return original_import(
            name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        fake_import,
    )

    from osa.voice.sounddevice_backend import (
        _load_sounddevice,
    )

    with pytest.raises(
        SoundDeviceAudioError,
        match="sounddevice==0.5.6",
    ):
        _load_sounddevice()
