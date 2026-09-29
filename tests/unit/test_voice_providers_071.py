from __future__ import annotations

import pytest

from osa.voice.providers import (
    VoiceProviderConfig,
    VoiceProviderConfigError,
    VoiceProviderSelection,
    create_default_voice_provider_config,
    load_voice_provider_config,
)


def test_default_voice_provider_config_is_safe_and_offline_oriented() -> None:
    config = create_default_voice_provider_config()

    assert config.stt.provider == "fixed"
    assert config.tts.provider == "fixed"
    assert config.vad_provider == "energy"
    assert config.audio_provider == "default"


def test_provider_selection_normalizes_provider_name() -> None:
    selection = VoiceProviderSelection(
        provider="faster_whisper",
        model="  model-id  ",
        language=" en ",
    )

    assert selection.provider == "faster-whisper"
    assert selection.model == "model-id"
    assert selection.language == "en"


def test_environment_overrides_provider_and_model_selection() -> None:
    config = load_voice_provider_config(
        {
            "OSA_VOICE_STT_PROVIDER": "faster_whisper",
            "OSA_VOICE_STT_MODEL": "small",
            "OSA_VOICE_STT_LANGUAGE": "en",
            "OSA_VOICE_STT_DEVICE": "cpu",
            "OSA_VOICE_STT_COMPUTE_TYPE": "int8",
            "OSA_VOICE_TTS_PROVIDER": "piper",
            "OSA_VOICE_TTS_MODEL": "voice-model",
            "OSA_VOICE_TTS_VOICE": "en_voice",
            "OSA_VOICE_TTS_LANGUAGE": "en",
            "OSA_VOICE_VAD_PROVIDER": "energy",
            "OSA_VOICE_AUDIO_PROVIDER": "sounddevice",
        }
    )

    assert config.stt.provider == "faster-whisper"
    assert config.stt.model == "small"
    assert config.stt.language == "en"
    assert config.stt.device == "cpu"
    assert config.stt.compute_type == "int8"

    assert config.tts.provider == "piper"
    assert config.tts.model == "voice-model"
    assert config.tts.voice == "en_voice"
    assert config.tts.language == "en"

    assert config.vad_provider == "energy"
    assert config.audio_provider == "sounddevice"


def test_environment_keeps_base_values_when_unset() -> None:
    base = VoiceProviderConfig(
        stt=VoiceProviderSelection(
            provider="faster-whisper",
            model="base-model",
            language="ru",
        ),
        tts=VoiceProviderSelection(
            provider="piper",
            model="tts-model",
            voice="voice-a",
        ),
        audio_provider="sounddevice",
    )

    config = load_voice_provider_config({},)

    assert config.stt.provider == "fixed"
    assert config.tts.provider == "fixed"

    config = VoiceProviderConfig.from_environment({}, base=base)

    assert config.stt.provider == "faster-whisper"
    assert config.stt.model == "base-model"
    assert config.tts.provider == "piper"
    assert config.tts.model == "tts-model"
    assert config.audio_provider == "sounddevice"


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(VoiceProviderConfigError):
        VoiceProviderConfig(
            stt=VoiceProviderSelection(provider="unknown-stt")
        )


def test_invalid_vad_is_rejected() -> None:
    with pytest.raises(VoiceProviderConfigError):
        VoiceProviderConfig(vad_provider="unknown-vad")


def test_invalid_audio_provider_is_rejected() -> None:
    with pytest.raises(VoiceProviderConfigError):
        VoiceProviderConfig(audio_provider="unknown-audio")


def test_public_dict_contains_configuration_metadata_only() -> None:
    config = VoiceProviderConfig(
        stt=VoiceProviderSelection(
            provider="faster-whisper",
            model="base",
            language="en",
        ),
        tts=VoiceProviderSelection(
            provider="piper",
            model="model",
            voice="voice",
        ),
    )

    public = config.public_dict()

    assert public["stt"]["provider"] == "faster-whisper"
    assert public["stt"]["model"] == "base"
    assert public["tts"]["provider"] == "piper"
    assert public["tts"]["voice"] == "voice"
