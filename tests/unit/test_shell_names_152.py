"""Tests for /name, /heard and follow-up configuration helpers (OSA 1.5.2.4)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from osa.shell.session import (
    describe_names,
    format_heard,
    handle_name_command,
    wake_config_from_env,
)
from osa.voice.names import WakeNameService
from osa.voice.wake import WakeConfig


class _Listener:
    def __init__(self, items=()) -> None:
        self._items = list(items)

    def recent_heard(self) -> list[tuple[str, str]]:
        return list(self._items)


# --- describe / /name --------------------------------------------------------------


def test_describe_names_without_and_with_an_extra_name() -> None:
    service = WakeNameService()

    assert describe_names(service) == "Откликаюсь на «Оса»."
    assert describe_names(None) == "Откликаюсь на «Оса»."

    service.set_custom_name("Джарвис")

    assert describe_names(service) == "Откликаюсь на «Оса» и «Джарвис»."


def test_name_command_without_service_is_unavailable() -> None:
    assert "недоступна" in handle_name_command("/name Джарвис", None)


def test_name_command_without_argument_shows_names_and_usage() -> None:
    reply = handle_name_command("/name", WakeNameService())

    assert "Оса" in reply and "/name <имя>" in reply


def test_name_command_sets_the_name() -> None:
    service = WakeNameService()

    reply = handle_name_command("/name Джарвис", service)

    assert service.custom_name == "джарвис"
    assert "Джарвис" in reply and "Оса" in reply


@pytest.mark.parametrize("word", ["clear", "Clear", "сброс", "убрать", "удалить"])
def test_name_command_clears_the_name(word: str) -> None:
    service = WakeNameService()
    service.set_custom_name("Джарвис")

    reply = handle_name_command(f"/name {word}", service)

    assert service.custom_name is None
    assert "удалено" in reply


def test_name_command_clear_without_a_name_says_so() -> None:
    assert "не было" in handle_name_command("/name clear", WakeNameService())


def test_name_command_reports_invalid_names_and_keeps_the_current_one() -> None:
    service = WakeNameService()
    service.set_custom_name("Джарвис")

    reply = handle_name_command("/name Оса", service)

    assert "всегда" in reply
    assert service.custom_name == "джарвис"


def test_name_command_warns_when_the_name_cannot_be_saved() -> None:
    with tempfile.TemporaryDirectory() as folder:
        blocker = Path(folder) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        service = WakeNameService(blocker / "settings.json")

        reply = handle_name_command("/name Джарвис", service)

        assert "до перезапуска" in reply


# --- /heard ------------------------------------------------------------------------


def test_heard_without_listener_or_phrases() -> None:
    assert "не запущено" in format_heard(None)
    assert "ничего" in format_heard(_Listener())


def test_heard_lists_phrases_with_the_gate_decision() -> None:
    listener = _Listener(
        [
            ("открой хром", "no_wake"),
            ("Оса, привет", "wake"),
            ("", "empty"),
            ("странная причина", "custom"),
        ]
    )

    text = format_heard(listener)

    assert "1. «открой хром» — без обращения" in text
    assert "2. «Оса, привет» — обращение" in text
    assert "3. «…» — пусто" in text
    assert "4. «странная причина» — custom" in text
    assert "только в памяти" in text


# --- follow-up configuration -------------------------------------------------------


def test_follow_up_defaults_when_not_configured() -> None:
    assert wake_config_from_env({}) == WakeConfig()
    assert wake_config_from_env({}).follow_up_seconds == 20.0


@pytest.mark.parametrize(("value", "expected"), [("35", 35.0), ("0", 0.0), (" 12.5 ", 12.5)])
def test_follow_up_can_be_set_from_the_environment(value: str, expected: float) -> None:
    config = wake_config_from_env({"OSA_FOLLOW_UP_SECONDS": value})

    assert config.follow_up_seconds == expected


@pytest.mark.parametrize("value", ["abc", "-5", "301", "nan", ""])
def test_invalid_follow_up_values_fall_back_to_the_default(value: str) -> None:
    config = wake_config_from_env({"OSA_FOLLOW_UP_SECONDS": value})

    assert config.follow_up_seconds == 20.0


def test_follow_up_override_keeps_the_other_settings() -> None:
    base = WakeConfig(max_leading_words=2)

    config = wake_config_from_env({"OSA_FOLLOW_UP_SECONDS": "40"}, base)

    assert config.max_leading_words == 2
    assert config.follow_up_seconds == 40.0
