from osa.core import Agent
from osa.core.modes import AgentMode
from osa.models import ModelInterface, ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import ToolRegistry
from osa.tools.registry import ToolInterface, ToolResult


class FakeTool(ToolInterface):
    @property
    def name(self) -> str:
        return "web_research"

    @property
    def description(self) -> str:
        return "test research tool"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output="research result",
        )


class FakeModel(ModelInterface):
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(self, request) -> ModelResponse:
        return ModelResponse(
            content="ok",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


def make_agent(mode: AgentMode) -> Agent:
    registry = ToolRegistry()
    registry.register(FakeTool())

    permissions = PermissionPolicy(
        {
            "web_research": PermissionLevel.ALLOW,
        }
    )

    return Agent(
        model=FakeModel(),
        tool_registry=registry,
        permission_policy=permissions,
        mode=mode,
    )


def test_agent_defaults_to_chat_mode() -> None:
    agent = make_agent(AgentMode.CHAT)

    assert agent.mode is AgentMode.CHAT
    assert agent.mode_profile.mode is AgentMode.CHAT


def test_chat_mode_blocks_web_research() -> None:
    agent = make_agent(AgentMode.CHAT)

    try:
        agent.execute_tool(
            "web_research",
            {},
        )
    except Exception as exc:
        assert "not available" in str(exc)
        assert "chat" in str(exc)
    else:
        raise AssertionError(
            "web_research should be unavailable in chat mode"
        )


def test_research_mode_allows_web_research() -> None:
    agent = make_agent(AgentMode.RESEARCH)

    result = agent.execute_tool(
        "web_research",
        {},
    )

    assert result.success is True
    assert result.output == "research result"


def test_agent_mode_can_be_changed() -> None:
    agent = make_agent(AgentMode.CHAT)

    assert agent.mode is AgentMode.CHAT

    agent.set_mode("research")

    assert agent.mode is AgentMode.RESEARCH
    assert agent.mode_profile.allows_research is True


def test_task_mode_keeps_normal_tools_available() -> None:
    class CalculatorTool(ToolInterface):
        @property
        def name(self) -> str:
            return "calculator"

        @property
        def description(self) -> str:
            return "calculator"

        @property
        def parameters(self):
            return {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }

        def execute(self, arguments):
            return ToolResult(
                success=True,
                output="42",
            )

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    permissions = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    agent = Agent(
        model=FakeModel(),
        tool_registry=registry,
        permission_policy=permissions,
        mode=AgentMode.TASK,
    )

    result = agent.execute_tool(
        "calculator",
        {},
    )

    assert result.output == "42"

class RecordingToolModel(ModelInterface):
    def __init__(self) -> None:
        self.requests = []

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(self, request) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content="ok",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


def test_chat_mode_hides_web_research_from_model() -> None:
    model = RecordingToolModel()

    registry = ToolRegistry()
    registry.register(FakeTool())

    permissions = PermissionPolicy(
        {
            "web_research": PermissionLevel.ALLOW,
        }
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=permissions,
        mode=AgentMode.CHAT,
    )

    agent.chat("Hello")

    names = {
        definition.name
        for definition in model.requests[0].tools
    }

    assert "web_research" not in names


def test_research_mode_exposes_web_research_to_model() -> None:
    model = RecordingToolModel()

    registry = ToolRegistry()
    registry.register(FakeTool())

    permissions = PermissionPolicy(
        {
            "web_research": PermissionLevel.ALLOW,
        }
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=permissions,
        mode=AgentMode.RESEARCH,
    )

    agent.chat("Research something")

    names = {
        definition.name
        for definition in model.requests[0].tools
    }

    assert "web_research" in names

