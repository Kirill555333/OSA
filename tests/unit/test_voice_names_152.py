"""Tests for the user's extra wake name and its tools (OSA 1.5.2.4)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from osa.tools.wake_name import ClearWakeNameTool, SetWakeNameTool
from osa.voice.names import (
    NameChange,
    WakeNameService,
    alias_forms,
    cyrillic_to_latin,
    latin_to_cyrillic,
    normalize_wake_name,
)
from osa.voice.wake import WakeWordGate

# --- validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Джарвис", "джарвис"),
        ("  джарвис  ", "джарвис"),
        ("«Джарвис»", "джарвис"),
        ('"Jarvis".', "jarvis"),
        ("Пятница!", "пятница"),
        ("Ёжик", "ежик"),
        ("Анна-Мария", "анна-мария"),
    ],
)
def test_valid_names_are_normalized(raw: str, expected: str) -> None:
    assert normalize_wake_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "ab", "Дж", "Джарвис Второй", "Джарвис2", "123", "a" * 30, "-ab", "!!!"],
)
def test_invalid_names_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_wake_name(raw)


@pytest.mark.parametrize("raw", ["Оса", "оса", "OSA", "Выход", "стоп", "Да", "нет", "exit"])
def test_builtin_and_control_words_are_not_allowed_as_names(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_wake_name(raw)


def test_rejecting_the_builtin_name_explains_that_it_always_works() -> None:
    with pytest.raises(ValueError) as caught:
        normalize_wake_name("Оса")

    assert "всегда" in str(caught.value)


# --- alphabets ---------------------------------------------------------------------


def test_latin_name_gets_a_russian_spelling() -> None:
    assert latin_to_cyrillic("Jarvis") == "джарвис"
    assert latin_to_cyrillic("Shadow") == "шадов"
    assert alias_forms("jarvis") == ("jarvis", "джарвис")


def test_russian_name_gets_a_latin_spelling() -> None:
    assert cyrillic_to_latin("Джарвис") == "jarvis"
    assert alias_forms("джарвис") == ("джарвис", "jarvis")


def test_alias_forms_do_not_repeat_the_name() -> None:
    assert alias_forms("123") == ("123",)


def test_gate_accepts_both_spellings_of_a_custom_name() -> None:
    service = WakeNameService()
    service.set_custom_name("Джарвис")
    gate = WakeWordGate(extra_names=service.alias_names())

    assert gate.process("Jarvis, открой хром").command == "открой хром"
    gate.close_follow_up()
    assert gate.process("Джарвис, открой хром").command == "открой хром"


# --- service -----------------------------------------------------------------------


def test_service_starts_without_a_name_and_without_a_file() -> None:
    service = WakeNameService()

    assert service.custom_name is None
    assert service.alias_names() == ()


def test_set_replaces_the_previous_name_and_reports_persistence() -> None:
    with tempfile.TemporaryDirectory() as folder:
        service = WakeNameService(Path(folder) / "data" / "osa-settings.json")

        first = service.set_custom_name("Джарвис")
        second = service.set_custom_name("Пятница")

        assert first == NameChange("джарвис", True)
        assert second == NameChange("пятница", True)
        assert service.custom_name == "пятница"


def test_name_survives_a_restart() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "osa-settings.json"
        WakeNameService(path).set_custom_name("Джарвис")

        reloaded = WakeNameService(path)

        assert reloaded.custom_name == "джарвис"
        assert reloaded.alias_names() == ("джарвис", "jarvis")


def test_clear_removes_the_name_and_the_stored_value() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "osa-settings.json"
        service = WakeNameService(path)
        service.set_custom_name("Джарвис")

        assert service.clear_custom_name() is True
        assert service.clear_custom_name() is False
        assert WakeNameService(path).custom_name is None
        assert "wake_name" not in json.loads(path.read_text(encoding="utf-8"))


def test_other_settings_in_the_file_are_preserved() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "osa-settings.json"
        path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

        WakeNameService(path).set_custom_name("Джарвис")

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["theme"] == "dark"
        assert saved["wake_name"] == "джарвис"


@pytest.mark.parametrize(
    "content",
    ["", "not json", "[1, 2]", '{"wake_name": 5}', '{"wake_name": "оса"}'],
)
def test_corrupt_or_invalid_settings_are_ignored(content: str) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "osa-settings.json"
        path.write_text(content, encoding="utf-8")

        assert WakeNameService(path).custom_name is None


def test_unwritable_location_keeps_the_name_for_the_session() -> None:
    with tempfile.TemporaryDirectory() as folder:
        blocker = Path(folder) / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        service = WakeNameService(blocker / "osa-settings.json")

        change = service.set_custom_name("Джарвис")

        assert change.persisted is False
        assert service.custom_name == "джарвис"


def test_invalid_name_leaves_the_current_one_untouched() -> None:
    service = WakeNameService()
    service.set_custom_name("Джарвис")

    with pytest.raises(ValueError):
        service.set_custom_name("Оса")

    assert service.custom_name == "джарвис"


def test_subscribers_are_notified_immediately_and_on_every_change() -> None:
    service = WakeNameService()
    service.set_custom_name("Джарвис")
    seen: list[tuple[str, ...]] = []

    service.subscribe(seen.append)
    service.set_custom_name("Пятница")
    service.clear_custom_name()

    assert seen == [
        ("джарвис", "jarvis"),
        ("пятница", "pyatnitsa"),
        (),
    ]


def test_subscribe_without_immediate_notification() -> None:
    service = WakeNameService()
    seen: list[tuple[str, ...]] = []

    service.subscribe(seen.append, notify=False)

    assert seen == []


def test_failing_subscriber_does_not_break_the_change() -> None:
    service = WakeNameService()
    seen: list[tuple[str, ...]] = []

    def broken(names: tuple[str, ...]) -> None:
        raise RuntimeError("listener crashed")

    service.subscribe(broken, notify=False)
    service.subscribe(seen.append, notify=False)

    service.set_custom_name("Джарвис")

    assert seen == [("джарвис", "jarvis")]


# --- tools -------------------------------------------------------------------------


def test_set_tool_changes_the_name_and_reports_both_names() -> None:
    service = WakeNameService()
    tool = SetWakeNameTool(service)

    result = tool.execute({"name": "Джарвис"})

    assert result.success is True
    assert "Джарвис" in result.output and "Оса" in result.output
    assert service.custom_name == "джарвис"
    assert result.metadata["name"] == "джарвис"


def test_set_tool_reports_invalid_names_as_failures() -> None:
    tool = SetWakeNameTool(WakeNameService())

    assert tool.execute({"name": "Оса"}).success is False
    assert tool.execute({"name": ""}).success is False
    assert tool.execute({}).success is False


def test_set_tool_mentions_when_the_name_could_not_be_saved() -> None:
    with tempfile.TemporaryDirectory() as folder:
        blocker = Path(folder) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        tool = SetWakeNameTool(WakeNameService(blocker / "settings.json"))

        result = tool.execute({"name": "Джарвис"})

        assert result.success is True
        assert "до перезапуска" in result.output


def test_clear_tool_removes_the_name() -> None:
    service = WakeNameService()
    service.set_custom_name("Джарвис")
    tool = ClearWakeNameTool(service)

    first = tool.execute({})
    second = tool.execute({})

    assert first.success is True and first.metadata["removed"] is True
    assert second.success is True and second.metadata["removed"] is False
    assert service.custom_name is None


def test_tools_expose_names_and_schemas() -> None:
    service = WakeNameService()

    assert SetWakeNameTool(service).name == "set_wake_name"
    assert ClearWakeNameTool(service).name == "clear_wake_name"
    assert SetWakeNameTool(service).parameters["required"] == ["name"]
    assert ClearWakeNameTool(service).parameters["required"] == []
