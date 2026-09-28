from __future__ import annotations

from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.planning import Plan, PlanStep
from osa.tasks import (
    TaskActionResolver,
    TaskManager,
)
from osa.tools import (
    ToolInterface,
    ToolRegistry,
    ToolResult,
)


class FakeTool(ToolInterface):
    """Minimal tool for resolver tests."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Return supplied text."

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    def execute(self, arguments) -> ToolResult:
        return ToolResult(
            success=True,
            output=str(arguments["text"]),
        )


class FakeResolverModel(ModelInterface):
    """Return one deterministic submit_action call."""

    @property
    def model_name(self) -> str:
        return "resolver-test"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        return ModelResponse(
            content="",
            model_name=self.model_name,
            tool_calls=(
                ToolCall(
                    id="action_1",
                    name="submit_action",
                    arguments={
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                    },
                ),
            ),
        )

    def health_check(self) -> bool:
        return True


def create_task():
    manager = TaskManager()

    plan = Plan(
        goal="Test action resolution",
        steps=(
            PlanStep(
                step_id="one",
                description="Echo hello",
            ),
        ),
    )

    run = manager.create_run(plan)

    return manager, run.tasks["one"]


def test_action_resolver_selects_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool())

    _, task = create_task()

    resolver = TaskActionResolver(
        FakeResolverModel(),
        registry,
    )

    action = resolver.resolve(
        task,
        {},
    )

    assert action.tool_name == "echo"
    assert action.arguments == {
        "text": "hello",
    }


def test_action_resolver_rejects_unknown_tool() -> None:
    class UnknownToolModel(FakeResolverModel):
        def generate(
            self,
            request: ModelRequest,
        ) -> ModelResponse:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                tool_calls=(
                    ToolCall(
                        id="action_1",
                        name="submit_action",
                        arguments={
                            "tool_name": "missing",
                            "arguments": {},
                        },
                    ),
                ),
            )

    registry = ToolRegistry()
    registry.register(FakeTool())

    _, task = create_task()

    resolver = TaskActionResolver(
        UnknownToolModel(),
        registry,
    )

    try:
        resolver.resolve(
            task,
            {},
        )
    except RuntimeError as exc:
        assert "Unknown tool selected" in str(exc)
    else:
        raise AssertionError(
            "Unknown tool should be rejected."
        )
