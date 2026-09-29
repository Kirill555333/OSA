from __future__ import annotations

from osa.core import (
    AgentVoiceActionAdapter,
    AgentVoiceActionIntegrationError,
    TaskActionResolverVoiceAdapter,
    VoiceActionResolver,
    VoiceActionResponse,
    create_agent_voice_action_adapter,
)
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter as DirectAgentVoiceActionAdapter,
    AgentVoiceActionIntegrationError as DirectIntegrationError,
    TaskActionResolverVoiceAdapter as DirectTaskResolverAdapter,
    VoiceActionResolver as DirectVoiceActionResolver,
    VoiceActionResponse as DirectVoiceActionResponse,
    create_agent_voice_action_adapter as DirectFactory,
)
from osa.voice import (
    UnifiedVoiceActionCompositionError,
    create_unified_voice_session,
)
from osa.voice.unified_action import (
    UnifiedVoiceActionCompositionError as DirectCompositionError,
    create_unified_voice_session as DirectSessionFactory,
)


def test_core_public_api_exports_voice_action_symbols():
    assert AgentVoiceActionAdapter is (
        DirectAgentVoiceActionAdapter
    )
    assert AgentVoiceActionIntegrationError is (
        DirectIntegrationError
    )
    assert TaskActionResolverVoiceAdapter is (
        DirectTaskResolverAdapter
    )
    assert VoiceActionResolver is (
        DirectVoiceActionResolver
    )
    assert VoiceActionResponse is (
        DirectVoiceActionResponse
    )
    assert create_agent_voice_action_adapter is (
        DirectFactory
    )


def test_voice_public_api_exports_unified_composition():
    assert UnifiedVoiceActionCompositionError is (
        DirectCompositionError
    )
    assert create_unified_voice_session is (
        DirectSessionFactory
    )


def test_core_all_contains_voice_action_symbols():
    import osa.core as core

    expected = {
        "AgentVoiceActionAdapter",
        "AgentVoiceActionIntegrationError",
        "TaskActionResolverVoiceAdapter",
        "VoiceActionResolver",
        "VoiceActionResponse",
        "create_agent_voice_action_adapter",
    }

    assert expected.issubset(
        set(core.__all__)
    )


def test_voice_all_contains_unified_composition_symbols():
    import osa.voice as voice

    expected = {
        "UnifiedVoiceActionCompositionError",
        "create_unified_voice_session",
    }

    assert expected.issubset(
        set(voice.__all__)
    )
