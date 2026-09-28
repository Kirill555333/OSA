import pytest

from osa.core.modes import (
    AgentMode,
    AgentModePolicy,
)


def test_default_profiles_cover_all_modes() -> None:
    policy = AgentModePolicy()

    for mode in AgentMode:
        profile = policy.profile(mode)

        assert profile.mode is mode
        assert profile.instruction
        assert profile.max_tool_rounds > 0


def test_chat_mode_does_not_enable_research() -> None:
    policy = AgentModePolicy()

    profile = policy.profile(AgentMode.CHAT)

    assert profile.allows_tools is True
    assert profile.allows_research is False
    assert profile.autonomous is False
    assert policy.supports_tool(
        AgentMode.CHAT,
        "calculator",
    )
    assert not policy.supports_tool(
        AgentMode.CHAT,
        "web_research",
    )


def test_research_mode_enables_research() -> None:
    policy = AgentModePolicy()

    profile = policy.profile("research")

    assert profile.mode is AgentMode.RESEARCH
    assert profile.allows_research is True
    assert profile.autonomous is False

    assert policy.supports_tool(
        AgentMode.RESEARCH,
        "web_research",
    )


def test_autonomous_mode_has_larger_tool_budget() -> None:
    policy = AgentModePolicy()

    autonomous = policy.profile(
        AgentMode.AUTONOMOUS
    )
    task = policy.profile(AgentMode.TASK)

    assert autonomous.autonomous is True
    assert autonomous.max_tool_rounds > task.max_tool_rounds
    assert autonomous.allows_research is True


def test_unknown_mode_is_rejected() -> None:
    policy = AgentModePolicy()

    with pytest.raises(ValueError):
        policy.profile("unknown")


def test_custom_profile_set_must_cover_every_mode() -> None:
    policy = AgentModePolicy()

    profile = policy.profile(AgentMode.TASK)

    with pytest.raises(ValueError):
        AgentModePolicy(
            {
                AgentMode.CHAT: profile,
            }
        )
