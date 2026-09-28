import http.client

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


def test_llama_cpp_reuses_http_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return (
                b'{"choices":[{"message":{"content":"ok"}}]}'
            )

        def close(self) -> None:
            pass

    class FakeConnection:
        instances: list["FakeConnection"] = []

        def __init__(
            self,
            host: str,
            port: int | None = None,
            *,
            timeout: float | object = object(),
        ) -> None:
            self.requests: list[tuple[str, str]] = []
            self.responses = 0
            FakeConnection.instances.append(self)

        def request(
            self,
            method: str,
            target: str,
            *,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.requests.append((method, target))

        def getresponse(self) -> FakeResponse:
            self.responses += 1
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        http.client,
        "HTTPConnection",
        FakeConnection,
    )

    model = LlamaCppModel()

    request_data = ModelRequest(
        messages=(
            ChatMessage(
                role="user",
                content="Hello",
            ),
        )
    )

    model.generate(request_data)
    model.generate(request_data)

    assert len(FakeConnection.instances) == 1
    assert FakeConnection.instances[0].responses == 2


def test_llama_cpp_reconnects_after_broken_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return (
                b'{"choices":[{"message":{"content":"ok"}}]}'
            )

        def close(self) -> None:
            pass

    class FakeConnection:
        instances: list["FakeConnection"] = []

        def __init__(
            self,
            host: str,
            port: int | None = None,
            *,
            timeout: float | object = object(),
        ) -> None:
            self.failed = False
            FakeConnection.instances.append(self)

        def request(
            self,
            method: str,
            target: str,
            *,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            if len(FakeConnection.instances) == 1 and not self.failed:
                self.failed = True
                raise ConnectionError("broken connection")

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        http.client,
        "HTTPConnection",
        FakeConnection,
    )

    model = LlamaCppModel()

    request_data = ModelRequest(
        messages=(
            ChatMessage(
                role="user",
                content="Hello",
            ),
        )
    )

    response = model.generate(request_data)

    assert response.content == "ok"
    assert len(FakeConnection.instances) == 2


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
