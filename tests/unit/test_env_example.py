"""Tests for .env.example (5.3)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

REQUIRED_KEYS = {
    "OSA_LLAMA_URL",
    "OSA_TOOL_CHOICE",
    "OSA_SYSTEM_PROMPT",
    "OSA_HOME",
    "OSA_WORKSPACE",
    "OSA_STT_MODEL",
    "OSA_VOICE_DEBUG",
    "OSA_FOLLOW_UP_SECONDS",
    "OSA_RUN_LIVE_AUDIO",
}

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file(), f"{ENV_EXAMPLE} not found"


def test_all_documented_keys_present() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    keys = {
        match.group(1)
        for match in re.finditer(r"^(OSA_[A-Z_]+)=", text, flags=re.MULTILINE)
    }
    missing = REQUIRED_KEYS - keys
    assert not missing, f"Missing keys in .env.example: {sorted(missing)}"


def test_no_real_secrets() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text), f"Possible secret matched {pattern.pattern}"


def test_all_values_are_empty() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key.startswith("OSA_"):
            assert value.strip() == "", f"{key} should have an empty placeholder"
