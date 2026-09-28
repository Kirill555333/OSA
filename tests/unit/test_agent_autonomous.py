from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.core import Agent
from osa.core.modes import AgentMode
from osa.models import ModelInterface, ModelRequest, ModelResponse
from osa.tasks.autonomous import AutonomousRunReport


class FakeModel(ModelInterface):
    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="ok",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


@dataclass
class FakeAutonomousLoop:
    calls: list[str]

    def run(self, goal: str) -> AutonomousRunReport:
        self.calls.append(goal)
        return AutonomousRunReport(
            goal=goal,
            run_id="run-1",
            success=True,
            stopped=False,
            stop_reason=None,
        )


def make_agent(
    *,
    mode: AgentMode,
    autonomous_loop=None,
) -> Agent:
    from osa.tools import ToolRegistry

    return Agent(
        model=FakeModel(),
        tool_registry=ToolRegistry(),
        mode=mode,
        autonomous_loop=autonomous_loop,
    )


def test_agent_runs_autonomous_loop() -> None:
    loop = FakeAutonomousLoop([])
    agent = make_agent(
        mode=AgentMode.AUTONOMOUS,
        autonomous_loop=loop,
    )

    report = agent.run_autonomous("finish the project")

    assert report.success is True
    assert loop.calls == ["finish the project"]


def test_agent_rejects_autonomous_execution_in_chat_mode() -> None:
    loop = FakeAutonomousLoop([])
    agent = make_agent(
        mode=AgentMode.CHAT,
        autonomous_loop=loop,
    )

    with pytest.raises(Exception, match="AUTONOMOUS mode"):
        agent.run_autonomous("finish the project")

    assert loop.calls == []


def test_agent_rejects_unconfigured_autonomous_loop() -> None:
    agent = make_agent(
        mode=AgentMode.AUTONOMOUS,
    )

    with pytest.raises(Exception, match="not configured"):
        agent.run_autonomous("finish the project")
