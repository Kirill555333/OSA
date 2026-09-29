"""Voice composition for unified OSA action execution."""

from __future__ import annotations

from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
)
from osa.tasks.action import TaskActionResolver
from osa.voice.activation import VoiceActivationDetector
from osa.voice.session import (
    VoiceSession,
    VoiceSessionConfig,
)
from osa.voice.stt import SpeechToText
from osa.voice.tts import TextToSpeech
from osa.voice.vad import VoiceActivityDetector


class UnifiedVoiceActionCompositionError(RuntimeError):
    """Raised when unified voice composition is invalid."""


def create_unified_voice_session(
    agent: object,
    task_action_resolver: TaskActionResolver,
    vad: VoiceActivityDetector,
    stt: SpeechToText,
    tts: TextToSpeech,
    *,
    config: VoiceSessionConfig | None = None,
    activation_detector: VoiceActivationDetector | None = None,
) -> VoiceSession:
    """
    Build a VoiceSession whose Agent surface uses unified action recovery.

    The resulting flow is:

        voice -> activation -> action resolver
        -> ActionRequest -> Agent unified recovery
        -> recovery/safety/permission/confirmation/router
        -> text -> TTS
    """
    if agent is None:
        raise ValueError(
            "agent is required."
        )

    if task_action_resolver is None:
        raise ValueError(
            "task_action_resolver is required."
        )

    if vad is None:
        raise ValueError(
            "vad is required."
        )

    if stt is None:
        raise ValueError(
            "stt is required."
        )

    if tts is None:
        raise ValueError(
            "tts is required."
        )

    try:
        voice_agent = AgentVoiceActionAdapter(
            agent,
            _TaskResolverAdapter(task_action_resolver),
        )
    except Exception as exc:
        raise UnifiedVoiceActionCompositionError(
            f"Unable to compose unified voice action session: {exc}"
        ) from exc

    try:
        return VoiceSession(
            voice_agent,
            vad,
            stt,
            tts,
            config=config,
            activation_detector=activation_detector,
        )
    except Exception as exc:
        raise UnifiedVoiceActionCompositionError(
            f"Unable to create VoiceSession: {exc}"
        ) from exc


class _TaskResolverAdapter:
    """Adapt TaskActionResolver to the generic voice resolver protocol."""

    def __init__(
        self,
        resolver: TaskActionResolver,
    ) -> None:
        self._resolver = resolver

    def resolve(self, command: str):
        """Resolve voice text using the existing task action resolver."""
        from osa.core.agent_voice_action import (
            TaskActionResolverVoiceAdapter,
        )

        return TaskActionResolverVoiceAdapter(
            self._resolver,
        ).resolve(command)


__all__ = [
    "UnifiedVoiceActionCompositionError",
    "create_unified_voice_session",
]
