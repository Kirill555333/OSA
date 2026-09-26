from osa.core import Agent
from osa.models import (
    ModelConnectionError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)


class RecoveringModel(ModelInterface):
    """Model that fails once and then succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "recovering-test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            raise ModelConnectionError(
                "temporary failure"
            )

        return ModelResponse(
            content="Recovered successfully.",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


def test_agent_recovers_from_connection_failure() -> None:
    model = RecoveringModel()

    agent = Agent(
        model,
        system_prompt="You are OSA.",
    )

    response = agent.chat(
        "Hello"
    )

    assert response.content == (
        "Recovered successfully."
    )

    assert model.calls == 2


class AlwaysFailingModel(ModelInterface):
    """Model that always raises a connection error."""

    @property
    def model_name(self) -> str:
        return "failing-test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise ModelConnectionError(
            "server unavailable"
        )

    def health_check(self) -> bool:
        return False


def test_agent_reports_error_after_recovery_is_exhausted() -> None:
    agent = Agent(
        AlwaysFailingModel(),
        system_prompt="You are OSA.",
    )

    try:
        agent.chat("Hello")
    except Exception as exc:
        assert "Model request failed" in str(exc)
    else:
        raise AssertionError(
            "Agent should fail after recovery is exhausted."
        )
