from dataclasses import dataclass

from osa.core import Agent
from osa.core.modes import AgentMode
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.permissions import (
    PermissionLevel,
    PermissionPolicy,
)
from osa.research import ResearchSource
from osa.research.loop import ResearchLoopReport
from osa.tools import ToolRegistry
from osa.tools.research import WebResearchTool


@dataclass
class FakeResearchLoop:
    calls: int = 0

    def run(
        self,
        query: str,
        follow_up_queries: tuple[str, ...] = (),
    ) -> ResearchLoopReport:
        self.calls += 1

        assert query == "Python 3.12 new features"
        assert follow_up_queries == ()

        source = ResearchSource(
            url="https://example.com/python",
            title="Python documentation",
            text="Python 3.12 introduced several language improvements.",
            status_code=200,
        )

        return ResearchLoopReport(
            original_query=query,
            queries=(query,),
            sources=(source,),
            failed_urls=(),
            failed_queries=(),
        )


class ToolCallingModel(ModelInterface):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.requests.append(request)
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                usage={},
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="web_research",
                        arguments={
                            "query": "Python 3.12 new features",
                        },
                    ),
                ),
            )

        assert any(
            message.role == "tool"
            for message in request.messages
        )

        return ModelResponse(
            content="Research completed successfully.",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


class PermissionDeniedModel(ModelInterface):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.requests.append(request)
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                usage={},
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="web_research",
                        arguments={
                            "query": "Python 3.12 new features",
                        },
                    ),
                ),
            )

        assert any(
            message.role == "tool"
            for message in request.messages
        )

        return ModelResponse(
            content="The requested action was not executed.",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


def test_agent_executes_web_research_tool() -> None:
    model = ToolCallingModel()
    research_loop = FakeResearchLoop()

    registry = ToolRegistry()

    registry.register(
        WebResearchTool(research_loop)
    )

    permissions = PermissionPolicy(
        {
            "web_research": PermissionLevel.ALLOW,
        }
    )

    agent = Agent(
        model=model,
        system_prompt="You are OSA.",
        tool_registry=registry,
        permission_policy=permissions,
        mode=AgentMode.RESEARCH,
    )

    response = agent.chat(
        "Find current information about Python 3.12."
    )

    assert response.content == (
        "Research completed successfully."
    )

    assert model.calls == 2
    assert len(model.requests) == 2
    assert research_loop.calls == 1

    assert model.requests[0].tools
    assert model.requests[0].tools[0].name == "web_research"

    assert model.requests[1].messages[-1].role == "tool"


def test_agent_does_not_allow_web_research_without_permission() -> None:
    model = PermissionDeniedModel()
    research_loop = FakeResearchLoop()

    registry = ToolRegistry()

    registry.register(
        WebResearchTool(research_loop)
    )

    agent = Agent(
        model=model,
        system_prompt="You are OSA.",
        tool_registry=registry,
    )

    response = agent.chat(
        "Find current information about Python 3.12."
    )

    assert response.content == (
        "The requested action was not executed."
    )

    assert model.calls == 2
    assert research_loop.calls == 0
