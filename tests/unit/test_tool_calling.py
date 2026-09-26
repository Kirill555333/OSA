from osa.core import Agent
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry


class FakeToolCallingModel(ModelInterface):
    """Fake model that requests a calculator tool once."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "fake-tool-calling"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)

        if len(self.calls) == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="calculator",
                        arguments={"expression": "347 * 29"},
                    ),
                ),
            )

        assert request.messages[-1].role == "tool"
        assert request.messages[-1].tool_call_id == "call_1"
        assert request.messages[-1].content == "10063"

        return ModelResponse(
            content="Ответ: 10063.",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def create_agent() -> Agent:
    """Create an Agent with an allowed calculator tool."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    permission_policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    return Agent(
        model=FakeToolCallingModel(),
        tool_registry=registry,
        permission_policy=permission_policy,
    )


def test_agent_executes_model_tool_call() -> None:
    model = FakeToolCallingModel()
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    permission_policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=permission_policy,
    )

    response = agent.chat("Сколько будет 347 * 29?")

    assert response.content == "Ответ: 10063."
    assert len(model.calls) == 2


def test_tool_definition_is_sent_to_model() -> None:
    agent = create_agent()

    agent.chat("Посчитай 10 + 5.")

    model = agent.model
    assert isinstance(model, FakeToolCallingModel)

    assert model.calls[0].tools[0].name == "calculator"
    assert model.calls[0].tools[0].parameters["type"] == "object"
