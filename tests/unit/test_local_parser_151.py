"""Contract tests for the fail-closed llama.cpp response parser (OSA 1.5.1.1)."""

from __future__ import annotations

import json

import pytest

from osa.models import local as local_module
from osa.models.interface import (
    ChatMessage,
    ModelConnectionError,
    ModelRequest,
    ToolDefinition,
)
from osa.models.local import LlamaCppConfig, LlamaCppModel


def _body(message: dict, finish_reason: str = "stop") -> str:
    return json.dumps(
        {
            "model": "test-model",
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {},
        }
    )


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        return None


class _FakeClient:
    def __init__(self, body: str | None = None, error: Exception | None = None) -> None:
        self._body = body
        self._error = error
        self.payloads: list[dict] = []

    def request(self, method, url, body=None, headers=None):
        if self._error is not None:
            raise self._error
        if body:
            self.payloads.append(json.loads(body))
        return _FakeResponse(self._body or "")

    def close(self) -> None:
        return None


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="test tool",
        parameters={"type": "object", "properties": {}},
    )


def _request(*tool_names: str) -> ModelRequest:
    return ModelRequest(
        messages=(ChatMessage(role="user", content="hello"),),
        tools=tuple(_tool(name) for name in tool_names),
    )


def _model_with(client: _FakeClient) -> LlamaCppModel:
    model = LlamaCppModel()
    model._http_client = client
    return model


# --- text that merely mentions a tool never becomes a call -------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Я бы вызвал open_url для этого.",
        "Нужно использовать launch_application и открыть приложение.",
        "close_application Google Chrome",
        "type_text привет",
        "press_key enter",
        "set_volume 50",
        "open_url https://example.com",
    ],
)
def test_text_mentioning_tool_names_creates_no_tool_call(text: str) -> None:
    model = LlamaCppModel()

    response = model._parse_response(_body({"content": text}))

    assert response.tool_calls == ()
    assert response.content == text


def test_text_mentioning_tool_names_creates_no_tool_call_with_allowed_tools() -> None:
    model = LlamaCppModel()
    allowed = frozenset({"open_url", "launch_application"})

    response = model._parse_response(
        _body({"content": "Вызываю open_url https://google.com"}),
        allowed,
    )

    assert response.tool_calls == ()
    assert "open_url" in response.content


# --- structured formats are still understood ---------------------------------------


def test_native_tool_calls_are_parsed() -> None:
    model = LlamaCppModel()
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "abc",
                "type": "function",
                "function": {
                    "name": "open_url",
                    "arguments": json.dumps({"url": "https://example.com"}),
                },
            }
        ],
    }

    response = model._parse_response(_body(message, "tool_calls"))

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "abc"
    assert call.name == "open_url"
    assert dict(call.arguments) == {"url": "https://example.com"}
    assert response.finish_reason == "tool_calls"


def test_tool_call_tag_is_parsed_for_offered_tool() -> None:
    model = LlamaCppModel()
    text = (
        '<tool_call>{"name": "open_url", "arguments": '
        '{"url": "https://example.com"}}</tool_call>'
    )

    response = model._parse_response(_body({"content": text}), frozenset({"open_url"}))

    assert [call.name for call in response.tool_calls] == ["open_url"]
    assert dict(response.tool_calls[0].arguments) == {"url": "https://example.com"}
    assert response.content == ""


def test_tool_call_tag_with_unknown_tool_is_dropped() -> None:
    model = LlamaCppModel()
    text = '<tool_call>{"name": "format_disk", "arguments": {}}</tool_call>'

    response = model._parse_response(_body({"content": text}), frozenset({"open_url"}))

    assert response.tool_calls == ()
    assert "format_disk" not in response.content


def test_json_block_is_parsed_for_offered_tool() -> None:
    model = LlamaCppModel()
    text = (
        '```json\n{"name": "launch_application", '
        '"arguments": {"application_name": "Notes"}}\n```'
    )

    response = model._parse_response(
        _body({"content": text}),
        frozenset({"launch_application"}),
    )

    assert [call.name for call in response.tool_calls] == ["launch_application"]
    assert dict(response.tool_calls[0].arguments) == {"application_name": "Notes"}


def test_json_block_with_unknown_tool_is_ignored() -> None:
    model = LlamaCppModel()
    text = '```json\n{"name": "close_application", "arguments": {}}\n```'

    response = model._parse_response(_body({"content": text}), frozenset({"open_url"}))

    assert response.tool_calls == ()


def test_invalid_response_body_raises() -> None:
    model = LlamaCppModel()

    with pytest.raises(local_module.ModelResponseError):
        model._parse_response("<html>not json</html>")


# --- generate() wiring -------------------------------------------------------------


def test_generate_filters_text_calls_by_offered_tools() -> None:
    text = '<tool_call>{"name": "close_application", "arguments": {}}</tool_call>'
    client = _FakeClient(body=_body({"content": text}))
    model = _model_with(client)

    response = model.generate(_request("open_url"))

    assert response.tool_calls == ()
    assert client.payloads[0]["tools"][0]["function"]["name"] == "open_url"


def test_generate_parses_text_call_when_tool_is_offered() -> None:
    text = '<tool_call>{"name": "open_url", "arguments": {"url": "https://a.b"}}</tool_call>'
    model = _model_with(_FakeClient(body=_body({"content": text})))

    response = model.generate(_request("open_url"))

    assert [call.name for call in response.tool_calls] == ["open_url"]


def test_generate_connection_failure_raises_interface_error() -> None:
    model = _model_with(_FakeClient(error=OSError("refused")))

    with pytest.raises(ModelConnectionError):
        model.generate(_request())


# --- cleanup guarantees ------------------------------------------------------------


def test_local_module_reuses_interface_connection_error() -> None:
    assert local_module.ModelConnectionError is ModelConnectionError


def test_hardcoded_app_dictionary_is_gone() -> None:
    assert not hasattr(local_module, "COMMON_APP_MAP")
    assert not hasattr(local_module, "extract_app_from_text")


def test_model_name_comes_from_config() -> None:
    assert LlamaCppModel().model_name == "qwen2.5-vl-3b"
    assert LlamaCppModel(LlamaCppConfig(model_name="custom")).model_name == "custom"


def test_generate_stream_without_tools_does_not_recurse() -> None:
    model = _model_with(_FakeClient(body=_body({"content": "Привет!"})))

    chunks = list(model.generate_stream(_request()))
    events = list(model.generate_stream_events(_request()))

    assert chunks == ["Привет!"]
    assert [event.content for event in events] == ["Привет!"]


def test_generate_stream_events_with_tools_returns_tool_calls() -> None:
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "open_url", "arguments": "{}"},
            }
        ],
    }
    model = _model_with(_FakeClient(body=_body(message, "tool_calls")))

    events = list(model.generate_stream_events(_request("open_url")))

    assert len(events) == 1
    assert [call.name for call in events[0].tool_calls] == ["open_url"]
