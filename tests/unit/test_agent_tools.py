from osa.core import Agent
from osa.models import ModelInterface, ModelRequest, ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry


class FakeModel(ModelInterface):
    """Minimal model for Agent tool integration tests."""

    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="fake response",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def create_agent_with_calculator() -> Agent:
    """Create an Agent with the calculator tool registered and allowed."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    permission_policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    return Agent(
        model=FakeModel(),
        tool_registry=registry,
        permission_policy=permission_policy,
    )


def test_agent_exposes_registered_tools() -> None:
    agent = create_agent_with_calculator()

    assert agent.tools.names() == ("calculator",)


def test_agent_executes_registered_tool() -> None:
    agent = create_agent_with_calculator()

    result = agent.execute_tool(
        "calculator",
        {
            "expression": "25 * 4",
        },
    )

    assert result.success is True
    assert result.output == "100"


def test_agent_returns_tool_failure() -> None:
    agent = create_agent_with_calculator()

    result = agent.execute_tool(
        "calculator",
        {
            "expression": "10 / 0",
        },
    )

    assert result.success is False
    assert result.error is not None
