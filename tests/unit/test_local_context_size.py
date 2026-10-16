"""Tests for LlamaCppModel.context_size() and ContextBudget integration (1.5.1.5.3)."""

from __future__ import annotations

import json

from osa.context import ContextBudget
from osa.models import LlamaCppConfig
from osa.models.local import LlamaCppModel


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, response=None, error=None) -> None:
        self._response = response
        self._error = error
        self.calls: list[tuple] = []

    def request(self, method, url, body, headers):
        self.calls.append((method, url, body, headers))
        if self._error is not None:
            raise self._error
        return self._response

    def close(self) -> None:
        return None


def _model_with_client(client: _FakeHttpClient) -> LlamaCppModel:
    model = LlamaCppModel(LlamaCppConfig())
    model._http_client = client  # type: ignore[attr-defined]
    return model


def test_context_size_reads_n_ctx_from_props() -> None:
    body = json.dumps(
        {"default_generation_settings": {"n_ctx": 8192}}
    ).encode("utf-8")
    client = _FakeHttpClient(_FakeResponse(200, body))
    model = _model_with_client(client)

    assert model.context_size() == 8192

    method, target, _, _ = client.calls[0]
    assert method == "GET"
    assert target == "/props"


def test_context_size_returns_none_on_connection_error() -> None:
    client = _FakeHttpClient(error=RuntimeError("connection refused"))
    model = _model_with_client(client)

    assert model.context_size() is None


def test_context_size_returns_none_on_invalid_json() -> None:
    client = _FakeHttpClient(_FakeResponse(200, b"not-json"))
    model = _model_with_client(client)

    assert model.context_size() is None


def test_context_size_returns_none_when_n_ctx_missing() -> None:
    body = json.dumps({"default_generation_settings": {}}).encode("utf-8")
    client = _FakeHttpClient(_FakeResponse(200, body))
    model = _model_with_client(client)

    assert model.context_size() is None


def test_context_size_rejects_bool_and_negative() -> None:
    for raw in (True, False, -1, 0, "8192", None):
        body = json.dumps(
            {"default_generation_settings": {"n_ctx": raw}}
        ).encode("utf-8")
        client = _FakeHttpClient(_FakeResponse(200, body))
        model = _model_with_client(client)
        assert model.context_size() is None, f"expected None for {raw!r}"


def test_context_budget_from_8192_has_7680_input_tokens() -> None:
    budget = ContextBudget.create(
        total_context_limit=8192,
        reserved_output_tokens=512,
    )
    assert budget.total_context_limit == 8192
    assert budget.reserved_output_tokens == 512
    assert budget.max_input_tokens == 7680
