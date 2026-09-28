from osa.main import create_agent
from osa.permissions import PermissionLevel


def test_create_agent_registers_web_research() -> None:
    agent = create_agent()

    names = {
        definition.name
        for definition in agent.tools.definitions()
    }

    assert "browser_fetch" in names
    assert "web_research" in names


def test_create_agent_allows_web_research() -> None:
    agent = create_agent()

    decision = agent.permissions.decide(
        "web_research"
    )

    assert decision.level == PermissionLevel.ALLOW
