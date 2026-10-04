"""Local llama.cpp inference client with multimodal vision and strict tool calling.

Tool calls are accepted only in structured form: native OpenAI ``tool_calls``,
Qwen ``<tool_call>`` tags, or a JSON code block describing a call. Free text that
merely mentions a tool name never produces a tool call (fail-closed).
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import re
from typing import Any
import urllib.parse

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
    timeout: float = 60.0
    model_name: str = "qwen2.5-vl-3b"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/health"


class _PersistentHttpClient:
    """HTTP client reusing persistent connection."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._parsed = urllib.parse.urlsplit(base_url)
        self._timeout = timeout
        self._connection: http.client.HTTPConnection | None = None

    def _get_connection(self) -> http.client.HTTPConnection:
        if self._connection is None:
            host = self._parsed.hostname or "127.0.0.1"
            port = self._parsed.port or 8080
            self._connection = http.client.HTTPConnection(
                host=host,
                port=port,
                timeout=self._timeout,
            )
        return self._connection

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> http.client.HTTPResponse:
        conn = self._get_connection()
        try:
            conn.request(method, url, body=body, headers=headers)
            return conn.getresponse()
        except Exception:
            self.close()
            conn = self._get_connection()
            conn.request(method, url, body=body, headers=headers)
            return conn.getresponse()

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None


class LlamaCppModel(ModelInterface):
    """LLM client for llama.cpp server supporting text, vision, and strict tool calls."""

    def __init__(self, config: LlamaCppConfig | None = None) -> None:
        self._config = config or LlamaCppConfig()
        self._http_client = _PersistentHttpClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout,
        )
        parsed = urllib.parse.urlsplit(self._config.chat_url)
        self._chat_target = parsed.path or "/v1/chat/completions"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def _serialize_message(self, message: ChatMessage) -> dict[str, Any]:
        """Convert message to OpenAI-compatible format, encoding images if marker present."""
        data: dict[str, Any] = {"role": message.role}

        match = re.search(r"\[image_path:\s*([^\]]+)\]", str(message.content))
        if match:
            img_path = Path(match.group(1).strip())
            if img_path.exists():
                img_bytes = img_path.read_bytes()
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                clean_text = re.sub(
                    r"\[image_path:\s*[^\]]+\]\n?",
                    "",
                    message.content,
                ).strip()
                data["content"] = [
                    {"type": "text", "text": clean_text or "What is visible on this screen?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    },
                ]
            else:
                data["content"] = message.content
        else:
            data["content"] = message.content

        if message.tool_call_id is not None:
            data["tool_call_id"] = message.tool_call_id

        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(dict(tool_call.arguments)),
                    },
                }
                for tool_call in message.tool_calls
            ]

        return data

    def generate(self, request_data: ModelRequest) -> ModelResponse:
        """Send chat completion to llama-server."""
        payload: dict[str, Any] = {
            "messages": [self._serialize_message(msg) for msg in request_data.messages],
            "temperature": request_data.temperature,
            "max_tokens": request_data.max_tokens,
            "stream": False,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            "chat_template_kwargs": {"thinking": False},
        }

        allowed_tools: frozenset[str] | None = None

        if request_data.tools:
            allowed_tools = frozenset(tool.name for tool in request_data.tools)
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
        resp = None

        try:
            resp = self._http_client.request(
                "POST",
                self._chat_target,
                body=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            resp_body = resp.read().decode("utf-8")
        except Exception as exc:
            self._http_client.close()
            raise ModelConnectionError(f"Failed to contact llama.cpp server: {exc}") from exc
        finally:
            if resp is not None:
                resp.close()

        return self._parse_response(resp_body, allowed_tools)

    def generate_stream(self, request: ModelRequest) -> Iterator[str]:
        """Return the full response as a single chunk (no incremental streaming yet)."""
        response = self.generate(request)

        if response.content:
            yield response.content

    @staticmethod
    def _tool_call_from_mapping(
        parsed: Any,
        call_id: str,
        allowed_tools: frozenset[str] | None,
    ) -> ToolCall | None:
        """Build a ToolCall from a structured mapping, or None if it is not a valid call."""
        if not isinstance(parsed, dict):
            return None

        name = str(parsed.get("name") or parsed.get("tool") or "").strip()

        if not name:
            return None

        if allowed_tools is not None and name not in allowed_tools:
            return None

        args = parsed.get("arguments") or parsed.get("parameters") or {}

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        return ToolCall(
            id=call_id,
            name=name,
            arguments=args if isinstance(args, dict) else {},
        )

    def _parse_response(
        self,
        response_body: str,
        allowed_tools: frozenset[str] | None = None,
    ) -> ModelResponse:
        """Parse a chat completion body.

        ``allowed_tools`` is the set of tool names offered to the model in this request.
        Calls recovered from text (tags or JSON blocks) must name one of them. When it is
        None, no name filtering is applied.
        """
        try:
            data = json.loads(response_body)
            choice = data["choices"][0]
            message = choice["message"]
        except Exception as exc:
            raise ModelResponseError(f"llama.cpp invalid response: {response_body[:300]}") from exc

        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])

        # 1. Standard OpenAI function calling structure
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                name = str(fn.get("name", "")).strip()
                if name:
                    tool_calls.append(
                        ToolCall(
                            id=str(tc.get("id", f"call_{len(tool_calls) + 1}")),
                            name=name,
                            arguments=args if isinstance(args, dict) else {},
                        )
                    )

        content = str(message.get("content") or message.get("reasoning_content") or "").strip()

        # 2. Qwen <tool_call> tags: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        if "<tool_call>" in content:
            tc_matches = re.findall(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                content,
                flags=re.DOTALL,
            )
            for block in tc_matches:
                try:
                    parsed = json.loads(block)
                except Exception:
                    continue
                call = self._tool_call_from_mapping(
                    parsed,
                    f"call_tag_{len(tool_calls) + 1}",
                    allowed_tools,
                )
                if call is not None:
                    tool_calls.append(call)
            content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()

        # 3. JSON code blocks: ```json {"name": ..., "arguments": ...} ```
        if not tool_calls and "```" in content:
            json_blocks = re.findall(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                content,
                flags=re.DOTALL,
            )
            for block in json_blocks:
                try:
                    parsed = json.loads(block)
                except Exception:
                    continue
                call = self._tool_call_from_mapping(
                    parsed,
                    f"call_block_{len(tool_calls) + 1}",
                    allowed_tools,
                )
                if call is not None:
                    tool_calls.append(call)
            if tool_calls:
                content = re.sub(
                    r"```(?:json)?\s*\{.*?\}\s*```",
                    "",
                    content,
                    flags=re.DOTALL,
                ).strip()

        # Fail-closed: text that merely mentions a tool name is never turned into a call.

        return ModelResponse(
            content=content,
            model_name=str(data.get("model", self.model_name)),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}),
            tool_calls=tuple(tool_calls),
        )

    def health_check(self) -> bool:
        try:
            resp = self._http_client.request(
                "GET",
                self._config.health_url,
                body=None,
                headers={},
            )
            return resp.status in (200, 503)
        except Exception:
            return False
