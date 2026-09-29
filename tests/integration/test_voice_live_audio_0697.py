from __future__ import annotations

import os

import pytest

from osa.voice.platform import (
    AudioPlatform,
    create_default_audio_backend,
)
from osa.voice.audio import (
    AudioInputError,
    AudioOutputError,
)


RUN_LIVE_AUDIO = (
    os.environ.get(
        "OSA_RUN_LIVE_AUDIO",
        "",
    ).strip().casefold()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


pytestmark = pytest.mark.skipif(
    not RUN_LIVE_AUDIO,
    reason=(
        "Live audio smoke test is disabled. "
        "Set OSA_RUN_LIVE_AUDIO=1 to run it."
    ),
)


def _backend():
    backend = create_default_audio_backend()

    if not backend.available():
        pytest.skip(
            (
                "Default audio backend is unavailable "
                f"for platform '{backend.platform.value}'."
            )
        )

    return backend


def test_live_backend_is_supported_platform():
    backend = _backend()

    assert backend.platform in {
        AudioPlatform.MACOS,
        AudioPlatform.WINDOWS,
        AudioPlatform.LINUX,
    }


def test_live_microphone_can_start_read_and_stop():
    backend = _backend()
    microphone = backend.microphone()

    try:
        microphone.start()

        voice_input = microphone.read()

        assert voice_input.audio
        assert voice_input.sample_rate_hz > 0
        assert voice_input.channels > 0

    except AudioInputError:
        raise

    except Exception as exc:
        raise AssertionError(
            f"Live microphone smoke test failed: {exc}"
        ) from exc

    finally:
        microphone.stop()


def test_live_speaker_can_play_and_stop():
    backend = _backend()
    speaker = backend.speaker()

    # Very short silent PCM16 mono payload.
    audio = b"\x00\x00" * 160

    try:
        speaker.play(
            audio,
            sample_rate_hz=16_000,
            channels=1,
        )

    except AudioOutputError:
        raise

    except Exception as exc:
        raise AssertionError(
            f"Live speaker smoke test failed: {exc}"
        ) from exc

    finally:
        speaker.stop()


def test_live_microphone_and_speaker_can_be_created_together():
    backend = _backend()

    microphone = backend.microphone()
    speaker = backend.speaker()

    assert microphone is not None
    assert speaker is not None
