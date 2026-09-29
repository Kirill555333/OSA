"""Public voice API for OSA 0.6.x."""

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
from osa.voice.composition import (
    VoiceCompositionError,
    create_default_voice_runtime,
    create_voice_runtime_from_backend,
)
from osa.voice.contracts import (
    VoiceContractError,
    VoiceInput,
    VoiceOutput,
    VoiceSessionConfig,
    VoiceSessionState,
    VoiceTranscript,
)
from osa.voice.platform import (
    AudioBackend,
    AudioPlatform,
    AudioPlatformError,
    AudioPlatformUnavailableError,
    DesktopAudioBackendFactory,
    UnavailableAudioBackend,
    create_audio_backend_factory,
    create_default_audio_backend,
)
from osa.voice.runtime import (
    VoiceRuntime,
    VoiceRuntimeConfig,
    VoiceRuntimeError,
    VoiceRuntimeState,
    create_voice_runtime,
)
from osa.voice.session import (
    VoiceAgent,
    VoiceSession,
    VoiceSessionError,
    VoiceSessionResult,
    VoiceSessionStateError,
)
from osa.voice.stt import (
    FixedSpeechToText,
    SpeechToText,
    SpeechToTextError,
)
from osa.voice.tts import (
    FixedTextToSpeech,
    TextToSpeech,
    TextToSpeechError,
)
from osa.voice.vad import (
    EnergyVoiceActivityDetector,
    VoiceActivityDetectionError,
    VoiceActivityDetector,
)

__all__ = [
    "AudioBackend",
    "AudioInputError",
    "AudioOutputError",
    "AudioPlatform",
    "AudioPlatformError",
    "AudioPlatformUnavailableError",
    "DesktopAudioBackendFactory",
    "EnergyVoiceActivityDetector",
    "FakeMicrophone",
    "FakeSpeaker",
    "FixedSpeechToText",
    "FixedTextToSpeech",
    "MicrophoneInput",
    "SpeakerOutput",
    "SpeechToText",
    "SpeechToTextError",
    "TextToSpeech",
    "TextToSpeechError",
    "UnavailableAudioBackend",
    "VoiceActivityDetectionError",
    "VoiceActivityDetector",
    "VoiceAgent",
    "VoiceCompositionError",
    "VoiceContractError",
    "VoiceInput",
    "VoiceOutput",
    "VoiceRuntime",
    "VoiceRuntimeConfig",
    "VoiceRuntimeError",
    "VoiceRuntimeState",
    "VoiceSession",
    "VoiceSessionConfig",
    "VoiceSessionError",
    "VoiceSessionResult",
    "VoiceSessionState",
    "VoiceSessionStateError",
    "VoiceTranscript",
    "create_audio_backend_factory",
    "create_default_audio_backend",
    "create_default_voice_runtime",
    "create_fake_microphone",
    "create_fake_speaker",
    "create_voice_runtime",
    "create_voice_runtime_from_backend",
]
