import pytest

from osa.models.interface import (
    ChatMessage,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)
from osa.models.local import LlamaCppConfig, LlamaCppModel
from osa.models.registry import ModelRegistry


class DummyModel(ModelInterface):
    """Simple model implementation used only for testing."""

    @property
    def model_name(self) -> str:
        return "dummy"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="test response",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def test_model_request_contains_messages() -> None:
    request = ModelRequest(
        messages=(
            ChatMessage(
                role="user",
                content="Hello OSA",
            ),
        )
    )

    assert len(request.messages) == 1
    assert request.messages[0].content == "Hello OSA"


def test_dummy_model_implements_interface() -> None:
    model = DummyModel()

    assert isinstance(model, ModelInterface)
    assert model.health_check() is True

    response = model.generate(
        ModelRequest(
            messages=(
                ChatMessage(
                    role="user",
                    content="Hello",
                ),
            )
        )
    )

    assert response.content == "test response"
    assert response.model_name == "dummy"


def test_llama_cpp_config_builds_urls() -> None:
    config = LlamaCppConfig(
        base_url="http://localhost:9000/"
    )

    assert config.chat_url == "http://localhost:9000/v1/chat/completions"
    assert config.health_url == "http://localhost:9000/health"


def test_model_registry() -> None:
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(model, as_default=True)

    assert len(registry) == 1
    assert registry.names() == ("dummy",)
    assert registry.get("dummy") is model
    assert registry.get_default() is model


def test_registry_rejects_duplicate_models() -> None:
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(model)

    with pytest.raises(ValueError):
        registry.register(model)
