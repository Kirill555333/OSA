import json

from osa.core import Agent
from osa.models import ModelInterface, ModelRequest, ModelResponse
from osa.tools import CalculatorTool, ToolRegistry
from osa.utils import EventLogger


class FakeModel(ModelInterface):
    """Simple model used for observability tests."""

    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="Test response",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def test_event_logger_writes_json_lines(tmp_path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(
        log_path,
        session_id="test-session",
    )

    logger.log(
        "test.event",
        value=42,
        text="hello",
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    event = json.loads(lines[0])

    assert event["session_id"] == "test-session"
    assert event["event"] == "test.event"
    assert event["data"]["value"] == 42
    assert event["data"]["text"] == "hello"


def test_agent_writes_observability_events(tmp_path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        model=FakeModel(),
        tool_registry=registry,
        event_logger=logger,
    )

    response = agent.chat("Hello")

    assert response.content == "Test response"

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    event_names = [event["event"] for event in events]

    assert "agent.chat.started" in event_names
    assert "model.response" in event_names
    assert "agent.chat.completed" in event_names
