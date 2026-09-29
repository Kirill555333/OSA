from __future__ import annotations

from osa.core import Agent
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry


class MultiToolModel(ModelInterface):
    """Fake model that requests two tools in one round."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "multi-tool-test"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls.append(request)

        if len(self.calls) == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="call_a",
                        name="calculator",
                        arguments={
                            "expression": "10 + 5",
                        },
                    ),
                    ToolCall(
                        id="call_b",
                        name="calculator",
                        arguments={
                            "expression": "20 * 3",
                        },
                    ),
                ),
            )

        tool_messages = [
            message
            for message in request.messages
            if message.role == "tool"
        ]

        assert len(tool_messages) == 2

        assert tool_messages[0].tool_call_id == "call_a"
        assert tool_messages[0].content == "15"

        assert tool_messages[1].tool_call_id == "call_b"
        assert tool_messages[1].content == "60"

        return ModelResponse(
            content="Ответ: 15 и 60.",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def test_agent_executes_all_tool_calls_in_one_round() -> None:
    model = MultiToolModel()

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.ALLOW,
            }
        ),
    )

    response = agent.chat(
        "Посчитай 10 + 5 и 20 * 3."
    )

    assert response.content == "Ответ: 15 и 60."
    assert len(model.calls) == 2
