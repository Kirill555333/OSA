import pytest

from osa.core import Agent, PermissionDeniedError
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
)
from osa.permissions import (
    ConfirmationHandler,
    PermissionLevel,
    PermissionPolicy,
)
from osa.tools import CalculatorTool, ToolRegistry


class FakeModel(ModelInterface):
    """Simple model for permission tests."""

    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="test",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def create_agent(
    policy: PermissionPolicy,
    confirmation_handler: ConfirmationHandler | None = None,
) -> Agent:
    """Create an Agent configured for permission tests."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    return Agent(
        model=FakeModel(),
        tool_registry=registry,
        permission_policy=policy,
        confirmation_handler=confirmation_handler,
    )


def test_default_policy_denies_unknown_tools() -> None:
    policy = PermissionPolicy()

    decision = policy.decide("unknown_tool")

    assert decision.level == PermissionLevel.DENY


def test_policy_allows_explicitly_allowed_tool() -> None:
    policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    decision = policy.decide("calculator")

    assert decision.level == PermissionLevel.ALLOW


def test_policy_can_require_confirmation() -> None:
    policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.CONFIRM,
        }
    )

    decision = policy.decide("calculator")

    assert decision.level == PermissionLevel.CONFIRM


def test_denied_tool_cannot_execute() -> None:
    agent = create_agent(
        PermissionPolicy()
    )

    with pytest.raises(PermissionDeniedError):
        agent.execute_tool(
            "calculator",
            {"expression": "2 + 2"},
        )


def test_confirmed_tool_can_execute() -> None:
    policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.CONFIRM,
        }
    )

    confirmation_handler = ConfirmationHandler(
        lambda _: True
    )

    agent = create_agent(
        policy,
        confirmation_handler,
    )

    result = agent.execute_tool(
        "calculator",
        {"expression": "2 + 2"},
    )

    assert result.success is True
    assert result.output == "4"


def test_rejected_confirmation_blocks_tool() -> None:
    policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.CONFIRM,
        }
    )

    confirmation_handler = ConfirmationHandler(
        lambda _: False
    )

    agent = create_agent(
        policy,
        confirmation_handler,
    )

    with pytest.raises(PermissionDeniedError):
        agent.execute_tool(
            "calculator",
            {"expression": "2 + 2"},
        )
