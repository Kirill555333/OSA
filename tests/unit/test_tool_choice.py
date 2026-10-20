"""Tests for tool_choice (4.2)."""

from __future__ import annotations

import json

from osa.core import Agent
from osa.models import (
    ChatMessage,
    LlamaCppConfig,
    LlamaCppModel,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
)


class CapturingModel(ModelInterface):
    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "capture"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(content="ok", model_name="capture")

    def health_check(self) -> bool:
        return True


class _FakeResponse:
    def __init__(self) -> None:
        self._data = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "model": "fake",
            }
        ).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        return None

    @property
    def status(self) -> int:
        return 200


class _FakeHttpClient:
    def __init__(self) -> None:
        self.last_body: bytes | None = None

    def request(self, method, url, body, headers):
        self.last_body = body
        return _FakeResponse()

    def close(self) -> None:
        return None


def test_model_request_default_tool_choice_is_auto() -> None:
    request = ModelRequest(messages=(ChatMessage(role="user", content="hi"),))
    assert request.tool_choice == "auto"


def test_agent_passes_tool_choice_into_request() -> None:
    model = CapturingModel()
    agent = Agent(model=model, tool_choice="required")
    agent.chat("hi")
    assert model.calls
    assert all(call.tool_choice == "required" for call in model.calls)


def test_agent_default_tool_choice_is_auto() -> None:
    model = CapturingModel()
    agent = Agent(model=model)
    agent.chat("hi")
    assert model.calls
    assert all(call.tool_choice == "auto" for call in model.calls)


def _model_with_fake_client():
    model = LlamaCppModel(LlamaCppConfig())
    fake = _FakeHttpClient()
    model._http_client = fake  # type: ignore[attr-defined]
    return model, fake


def _tools():
    return (ToolDefinition(name="t", description="d", parameters={"type": "object"}),)


def test_llamacpp_payload_omits_tool_choice_when_auto() -> None:
    model, fake = _model_with_fake_client()
    model.generate(
        ModelRequest(
            messages=(ChatMessage(role="user", content="hi"),),
            tools=_tools(),
            tool_choice="auto",
        )
    )
    body = json.loads(fake.last_body)
    assert "tool_choice" not in body


def test_llamacpp_payload_includes_required() -> None:
    model, fake = _model_with_fake_client()
    model.generate(
        ModelRequest(
            messages=(ChatMessage(role="user", content="hi"),),
            tools=_tools(),
            tool_choice="required",
        )
    )
    body = json.loads(fake.last_body)
    assert body["tool_choice"] == "required"


def test_llamacpp_payload_includes_none() -> None:
    model, fake = _model_with_fake_client()
    model.generate(
        ModelRequest(
            messages=(ChatMessage(role="user", content="hi"),),
            tools=_tools(),
            tool_choice="none",
        )
    )
    body = json.loads(fake.last_body)
    assert body["tool_choice"] == "none"
