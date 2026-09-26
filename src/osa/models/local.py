"""Local model backend using the llama.cpp HTTP server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from osa.models.interface import (
    ChatMessage,
    ModelConnectionError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ToolCall,
)


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    """Connection settings for a local llama.cpp server."""

    base_url: str = "http://127.0.0.1:8080"
    timeout: float = 120.0

    @property
    def chat_url(self) -> str:
        """Return the OpenAI-compatible chat completion URL."""
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        """Return the llama.cpp health endpoint URL."""
        return f"{self.base_url.rstrip('/')}/health"


class LlamaCppModel(ModelInterface):
    """OSA adapter for a llama.cpp HTTP server."""

    def __init__(self, config: LlamaCppConfig | None = None) -> None:
        self._config = config or LlamaCppConfig()
        self._model_name = "llama.cpp-local"

    @property
    def model_name(self) -> str:
        """Return the adapter name."""
        return self._model_name

    def health_check(self) -> bool:
        """Check whether the llama.cpp server is ready."""
        health_request = request.Request(
            self._config.health_url,
            method="GET",
        )

        try:
            with request.urlopen(
                health_request,
                timeout=self._config.timeout,
            ) as response:
                return response.status == 200
        except (error.URLError, TimeoutError):
            return False

    def generate(self, request_data: ModelRequest) -> ModelResponse:
        """Send a chat completion request to llama.cpp."""
        payload = {
            "messages": [
                self._serialize_message(message)
                for message in request_data.messages
            ],
            "temperature": request_data.temperature,
            "max_tokens": request_data.max_tokens,
            "stream": False,
        }

        if request_data.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in request_data.tools
            ]

        body = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            self._config.chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self._config.timeout,
            ) as response:
                response_body = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise ModelConnectionError(
                f"Failed to contact llama.cpp server: {exc}"
            ) from exc

        return self._parse_response(response_body)

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, object]:
        """Convert an OSA message into the API message format."""
        data: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }

        if message.tool_call_id is not None:
            data["tool_call_id"] = message.tool_call_id

        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            dict(tool_call.arguments),
                            ensure_ascii=False,
                        ),
                    },
                }
                for tool_call in message.tool_calls
            ]

        return data

    def _parse_response(self, response_body: str) -> ModelResponse:
        """Parse a llama.cpp chat completion response."""
        try:
            data = json.loads(response_body)
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelResponseError(
                "llama.cpp returned an unexpected response format."
            ) from exc

        try:
            raw_tool_calls = message.get("tool_calls", [])

            if not isinstance(raw_tool_calls, list):
                raise ModelResponseError(
                    "llama.cpp returned invalid tool_calls data."
                )

            tool_calls = tuple(
                self._parse_tool_call(tool_call)
                for tool_call in raw_tool_calls
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ModelResponseError(
                "llama.cpp returned an invalid tool call."
            ) from exc

        return ModelResponse(
            content=str(message.get("content") or ""),
            model_name=str(data.get("model", self.model_name)),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}),
            tool_calls=tool_calls,
        )

    @staticmethod
    def _parse_tool_call(raw_tool_call: object) -> ToolCall:
        """Convert an API tool call into an OSA ToolCall."""
        if not isinstance(raw_tool_call, dict):
            raise ModelResponseError("Tool call must be a JSON object.")

        function = raw_tool_call.get("function")

        if not isinstance(function, dict):
            raise ModelResponseError("Tool call function is missing.")

        name = function.get("name")
        arguments = function.get("arguments")
        call_id = raw_tool_call.get("id")

        if not isinstance(name, str) or not name:
            raise ModelResponseError("Tool call name is invalid.")

        if not isinstance(arguments, str):
            raise ModelResponseError(
                "Tool call arguments must be a JSON string."
            )

        if not isinstance(call_id, str) or not call_id:
            raise ModelResponseError("Tool call ID is invalid.")

        parsed_arguments = json.loads(arguments)

        if not isinstance(parsed_arguments, dict):
            raise ModelResponseError(
                "Tool call arguments must be a JSON object."
            )

        return ToolCall(
            id=call_id,
            name=name,
            arguments=parsed_arguments,
        )
