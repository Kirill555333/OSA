"""Tests for OSA_LLAMA_URL env (4.1)."""

from __future__ import annotations


from osa.models.local import LlamaCppConfig


def test_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OSA_LLAMA_URL", "http://192.168.1.5:8080")
    config = LlamaCppConfig()
    assert config.base_url == "http://192.168.1.5:8080"


def test_base_url_default_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("OSA_LLAMA_URL", raising=False)
    config = LlamaCppConfig()
    assert config.base_url == "http://127.0.0.1:8080"


def test_explicit_base_url_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("OSA_LLAMA_URL", "http://192.168.1.5:8080")
    config = LlamaCppConfig(base_url="http://10.0.0.1:9090")
    assert config.base_url == "http://10.0.0.1:9090"


def test_check_llama_server_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("OSA_LLAMA_URL", "http://192.168.1.5:8080")
    from osa.main import check_llama_server

    captured = {}

    def fake_request(url, *args, **kwargs):
        captured["url"] = url
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.Request", fake_request)
    assert check_llama_server() is False
    assert "192.168.1.5" in str(captured["url"])
