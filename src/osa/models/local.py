"""Local model backend using the llama.cpp HTTP server."""

from __future__ import annotations

from collections.abc import Iterator

import http.client
import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from osa.models.interface import (
    ChatMessage,
    ModelConnectionError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
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


class _PersistentHttpClient:
    """Small stdlib HTTP client that reuses one TCP connection."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
    ) -> None:
        parsed = urlsplit(base_url)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "base_url must use http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "base_url must contain a hostname."
            )

        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._timeout = timeout
        self._connection: http.client.HTTPConnection | None = None

    def _get_connection(
        self,
    ) -> http.client.HTTPConnection:
        if self._connection is not None:
            return self._connection

        if self._scheme == "https":
            self._connection = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self._timeout,
            )
        else:
            self._connection = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=self._timeout,
            )

        return self._connection

    def request(
        self,
        method: str,
        target: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> http.client.HTTPResponse:
        """Send one request, retrying once after a broken connection."""
        last_error: Exception | None = None

        for attempt in range(2):
            connection = self._get_connection()

            try:
                connection.request(
                    method,
                    target,
                    body=body,
                    headers=headers or {},
                )

                return connection.getresponse()

            except (
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                OSError,
            ) as exc:
                last_error = exc
                self.close()

                if attempt == 0:
                    continue

                raise

        assert last_error is not None
        raise last_error

    def close(self) -> None:
        """Close the persistent connection when requested."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None



class LlamaCppModel(ModelInterface):
    """OSA adapter for a llama.cpp HTTP server."""

    def __init__(self, config: LlamaCppConfig | None = None) -> None:
        self._config = config or LlamaCppConfig()
        self._model_name = "llama.cpp-local"
        self._http_client = _PersistentHttpClient(
            self._config.base_url,
            self._config.timeout,
        )

        self._chat_target = self._url_target(
            self._config.chat_url
        )
        self._health_target = self._url_target(
            self._config.health_url
        )

    @property
    def model_name(self) -> str:
        """Return the adapter name."""
        return self._model_name

    def health_check(self) -> bool:
        """Check whether the llama.cpp server is ready."""
        response = None

        try:
            response = self._http_client.request(
                "GET",
                self._health_target,
            )
            response.read()
            return response.status == 200

        except (
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            OSError,
        ):
            self._http_client.close()
            return False

        finally:
            if response is not None:
                response.close()

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

        response = None

        try:
            response = self._http_client.request(
                "POST",
                self._chat_target,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            response_body = response.read().decode(
                "utf-8"
            )

        except (
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as exc:
            self._http_client.close()

            raise ModelConnectionError(
                f"Failed to contact llama.cpp server: {exc}"
            ) from exc

        finally:
            if response is not None:
                response.close()

        return self._parse_response(response_body)

    def generate_stream(
        self,
        request_data: ModelRequest,
    ) -> Iterator[str]:
        """Stream generated text from the llama.cpp server."""
        if request_data.tools:
            raise ModelResponseError(
                "Streaming with tool calls requires generate_stream_events()."
            )

        for event in self.generate_stream_events(
            request_data
        ):
            if event.content:
                yield event.content

    def generate_stream_events(
        self,
        request_data: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        """Stream text and tool-call events from llama.cpp."""
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [
                self._serialize_message(message)
                for message in request_data.messages
            ],
            "temperature": request_data.temperature,
            "max_tokens": request_data.max_tokens,
            "stream": True,
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

        data = json.dumps(payload).encode("utf-8")

        tool_calls: dict[
            int,
            dict[str, str],
        ] = {}

        response = None

        try:
            response = self._http_client.request(
                "POST",
                self._chat_target,
                body=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )

            for raw_line in response:
                line = (
                    raw_line.decode(
                        "utf-8",
                        errors="replace",
                    )
                    if isinstance(raw_line, bytes)
                    else raw_line
                )

                line = line.strip()

                if not line or not line.startswith("data:"):
                    continue

                event_data = line[5:].strip()

                if event_data == "[DONE]":
                    break

                try:
                    chunk = json.loads(event_data)
                except json.JSONDecodeError as exc:
                    raise ModelResponseError(
                        "Received invalid JSON in model stream."
                    ) from exc

                choices = chunk.get("choices", [])

                if not isinstance(choices, list) or not choices:
                    continue

                choice = choices[0]

                if not isinstance(choice, dict):
                    continue

                delta = choice.get("delta", {})

                if not isinstance(delta, dict):
                    continue

                content = delta.get("content")

                if isinstance(content, str) and content:
                    yield ModelStreamEvent(
                        content=content
                    )

                raw_tool_calls = delta.get(
                    "tool_calls",
                    [],
                )

                if isinstance(raw_tool_calls, list):
                    for raw_tool_call in raw_tool_calls:
                        if not isinstance(
                            raw_tool_call,
                            dict,
                        ):
                            continue

                        index = raw_tool_call.get(
                            "index",
                            0,
                        )

                        if not isinstance(index, int):
                            continue

                        current = tool_calls.setdefault(
                            index,
                            {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            },
                        )

                        call_id = raw_tool_call.get("id")

                        if isinstance(call_id, str):
                            current["id"] += (
                                call_id
                                if not current["id"]
                                else ""
                            )

                        function = raw_tool_call.get(
                            "function",
                            {},
                        )

                        if not isinstance(
                            function,
                            dict,
                        ):
                            continue

                        name = function.get("name")

                        if isinstance(name, str):
                            current["name"] += (
                                name
                                if not current["name"]
                                else ""
                            )

                        arguments = function.get(
                            "arguments"
                        )

                        if isinstance(
                            arguments,
                            str,
                        ):
                            current["arguments"] += arguments

            parsed_tool_calls: list[ToolCall] = []

            for index in sorted(tool_calls):
                raw = tool_calls[index]

                if not raw["id"]:
                    raise ModelResponseError(
                        f"Tool call {index} is missing its ID."
                    )

                if not raw["name"]:
                    raise ModelResponseError(
                        f"Tool call {index} is missing its name."
                    )

                try:
                    arguments = json.loads(
                        raw["arguments"] or "{}"
                    )
                except json.JSONDecodeError as exc:
                    raise ModelResponseError(
                        "Tool call arguments are not valid JSON."
                    ) from exc

                if not isinstance(arguments, dict):
                    raise ModelResponseError(
                        "Tool call arguments must be a JSON object."
                    )

                parsed_tool_calls.append(
                    ToolCall(
                        id=raw["id"],
                        name=raw["name"],
                        arguments=arguments,
                    )
                )

            if parsed_tool_calls:
                yield ModelStreamEvent(
                    tool_calls=tuple(
                        parsed_tool_calls
                    ),
                    finish_reason="tool_calls",
                )

        except ModelResponseError:
            self._http_client.close()
            raise

        except (
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as exc:
            self._http_client.close()

            raise ModelConnectionError(
                f"Could not connect to model server: {exc}"
            ) from exc

        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _url_target(url: str) -> str:
        """Convert a full URL into an HTTP request target."""
        parsed = urlsplit(url)
        target = parsed.path or "/"

        if parsed.query:
            target += f"?{parsed.query}"

        return target

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
