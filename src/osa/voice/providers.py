"""Voice provider configuration and model selection contracts for OSA 0.7.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from osa.utils.config import ConfigError, EnvironmentConfig


class VoiceProviderConfigError(ConfigError):
    """Raised when voice provider configuration is invalid."""


@dataclass(frozen=True)
class VoiceProviderSelection:
    """Provider plus optional model/runtime selection metadata."""

    provider: str
    model: str | None = None
    language: str | None = None
    device: str | None = None
    compute_type: str | None = None
    voice: str | None = None

    def __post_init__(self) -> None:
        normalized_provider = self.provider.strip().lower().replace("_", "-")
        if not normalized_provider:
            raise VoiceProviderConfigError("Voice provider name cannot be empty.")

        if any(char.isspace() for char in normalized_provider):
            raise VoiceProviderConfigError(
                "Voice provider name cannot contain whitespace."
            )

        object.__setattr__(self, "provider", normalized_provider)

        for field_name in (
            "model",
            "language",
            "device",
            "compute_type",
            "voice",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue

            normalized = value.strip()
            object.__setattr__(self, field_name, normalized or None)

    def public_dict(self) -> dict[str, str | None]:
        """Return safe configuration fields suitable for diagnostics/UI."""
        return {
            "provider": self.provider,
            "model": self.model,
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "voice": self.voice,
        }


@dataclass(frozen=True)
class VoiceProviderConfig:
    """Unified configuration for the voice provider layer."""

    stt: VoiceProviderSelection = field(
        default_factory=lambda: VoiceProviderSelection(provider="fixed")
    )
    tts: VoiceProviderSelection = field(
        default_factory=lambda: VoiceProviderSelection(provider="fixed")
    )
    vad_provider: str = "energy"
    audio_provider: str = "default"

    SUPPORTED_STT_PROVIDERS = frozenset(
        {
            "fixed",
            "faster-whisper",
        }
    )
    SUPPORTED_TTS_PROVIDERS = frozenset(
        {
            "fixed",
            "piper",
        }
    )
    SUPPORTED_VAD_PROVIDERS = frozenset(
        {
            "energy",
        }
    )
    SUPPORTED_AUDIO_PROVIDERS = frozenset(
        {
            "default",
            "sounddevice",
            "unavailable",
        }
    )

    def __post_init__(self) -> None:
        vad_provider = _normalize_provider_name(self.vad_provider)
        audio_provider = _normalize_provider_name(self.audio_provider)

        if vad_provider not in self.SUPPORTED_VAD_PROVIDERS:
            raise VoiceProviderConfigError(
                f"Unsupported VAD provider '{vad_provider}'."
            )

        if audio_provider not in self.SUPPORTED_AUDIO_PROVIDERS:
            raise VoiceProviderConfigError(
                f"Unsupported audio provider '{audio_provider}'."
            )

        if self.stt.provider not in self.SUPPORTED_STT_PROVIDERS:
            raise VoiceProviderConfigError(
                f"Unsupported STT provider '{self.stt.provider}'."
            )

        if self.tts.provider not in self.SUPPORTED_TTS_PROVIDERS:
            raise VoiceProviderConfigError(
                f"Unsupported TTS provider '{self.tts.provider}'."
            )

        object.__setattr__(self, "vad_provider", vad_provider)
        object.__setattr__(self, "audio_provider", audio_provider)

    @classmethod
    def defaults(cls) -> VoiceProviderConfig:
        """Return the safe offline/test-oriented default configuration."""
        return cls()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        base: VoiceProviderConfig | None = None,
    ) -> VoiceProviderConfig:
        """Build provider settings with environment values overriding the base."""
        env = EnvironmentConfig.from_environment(environ)
        defaults = base or cls.defaults()

        stt = VoiceProviderSelection(
            provider=env.get("OSA_VOICE_STT_PROVIDER", defaults.stt.provider)
            or defaults.stt.provider,
            model=env.get("OSA_VOICE_STT_MODEL", defaults.stt.model),
            language=env.get("OSA_VOICE_STT_LANGUAGE", defaults.stt.language),
            device=env.get("OSA_VOICE_STT_DEVICE", defaults.stt.device),
            compute_type=env.get(
                "OSA_VOICE_STT_COMPUTE_TYPE",
                defaults.stt.compute_type,
            ),
            voice=env.get("OSA_VOICE_STT_VOICE", defaults.stt.voice),
        )

        tts = VoiceProviderSelection(
            provider=env.get("OSA_VOICE_TTS_PROVIDER", defaults.tts.provider)
            or defaults.tts.provider,
            model=env.get("OSA_VOICE_TTS_MODEL", defaults.tts.model),
            language=env.get("OSA_VOICE_TTS_LANGUAGE", defaults.tts.language),
            device=env.get("OSA_VOICE_TTS_DEVICE", defaults.tts.device),
            compute_type=env.get(
                "OSA_VOICE_TTS_COMPUTE_TYPE",
                defaults.tts.compute_type,
            ),
            voice=env.get("OSA_VOICE_TTS_VOICE", defaults.tts.voice),
        )

        return cls(
            stt=stt,
            tts=tts,
            vad_provider=env.get(
                "OSA_VOICE_VAD_PROVIDER",
                defaults.vad_provider,
            )
            or defaults.vad_provider,
            audio_provider=env.get(
                "OSA_VOICE_AUDIO_PROVIDER",
                defaults.audio_provider,
            )
            or defaults.audio_provider,
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        base: VoiceProviderConfig | None = None,
    ) -> VoiceProviderConfig:
        """Build configuration from a mapping without exposing process environment."""
        defaults = base or cls.defaults()

        stt = _selection_from_mapping(
            values.get("stt"),
            fallback=defaults.stt,
        )
        tts = _selection_from_mapping(
            values.get("tts"),
            fallback=defaults.tts,
        )

        vad_provider = _string_value(
            values.get("vad_provider"),
            defaults.vad_provider,
        )
        audio_provider = _string_value(
            values.get("audio_provider"),
            defaults.audio_provider,
        )

        return cls(
            stt=stt,
            tts=tts,
            vad_provider=vad_provider,
            audio_provider=audio_provider,
        )

    def public_dict(self) -> dict[str, object]:
        """Return only non-secret provider metadata for diagnostics/UI."""
        return {
            "stt": self.stt.public_dict(),
            "tts": self.tts.public_dict(),
            "vad_provider": self.vad_provider,
            "audio_provider": self.audio_provider,
        }


def create_default_voice_provider_config() -> VoiceProviderConfig:
    """Return the default provider configuration."""
    return VoiceProviderConfig.defaults()


def load_voice_provider_config(
    environ: Mapping[str, str] | None = None,
) -> VoiceProviderConfig:
    """Load voice provider configuration from the process environment."""
    return VoiceProviderConfig.from_environment(environ)


def _normalize_provider_name(value: str) -> str:
    if not isinstance(value, str):
        raise VoiceProviderConfigError("Provider name must be a string.")

    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        raise VoiceProviderConfigError("Provider name cannot be empty.")

    if any(char.isspace() for char in normalized):
        raise VoiceProviderConfigError(
            "Provider name cannot contain whitespace."
        )

    return normalized


def _string_value(
    value: object,
    fallback: str,
) -> str:
    if value is None:
        return fallback

    if not isinstance(value, str):
        raise VoiceProviderConfigError("Configuration value must be a string.")

    normalized = value.strip()
    return normalized or fallback


def _selection_from_mapping(
    value: object,
    *,
    fallback: VoiceProviderSelection,
) -> VoiceProviderSelection:
    if value is None:
        return fallback

    if not isinstance(value, Mapping):
        raise VoiceProviderConfigError(
            "Provider selection must be a mapping."
        )

    def optional_string(name: str, default: str | None) -> str | None:
        raw = value.get(name)
        if raw is None:
            return default
        if not isinstance(raw, str):
            raise VoiceProviderConfigError(
                f"Provider field '{name}' must be a string."
            )
        normalized = raw.strip()
        return normalized or None

    provider_value = value.get("provider", fallback.provider)
    if not isinstance(provider_value, str):
        raise VoiceProviderConfigError(
            "Provider field 'provider' must be a string."
        )

    return VoiceProviderSelection(
        provider=provider_value,
        model=optional_string("model", fallback.model),
        language=optional_string("language", fallback.language),
        device=optional_string("device", fallback.device),
        compute_type=optional_string(
            "compute_type",
            fallback.compute_type,
        ),
        voice=optional_string("voice", fallback.voice),
    )
