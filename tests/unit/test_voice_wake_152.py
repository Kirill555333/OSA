"""Tests for wake-word gating (OSA 1.5.2.1)."""

from __future__ import annotations

import pytest

from osa.voice.wake import GateResult, ListenMode, WakeConfig, WakeWordGate


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _gate(mode: ListenMode = ListenMode.WAKE, clock: _Clock | None = None) -> WakeWordGate:
    return WakeWordGate(mode=mode, clock=clock or _Clock())


# --- wake mode ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("Оса, открой хром", "открой хром"),
        ("оса открой хром", "открой хром"),
        ("Осы, открой хром.", "открой хром"),
        ("Эй, Оса, открой хром", "открой хром"),
        ("ОСА! Закрой калькулятор", "Закрой калькулятор"),
        ("Osa open the calculator", "open the calculator"),
        ("Оса — покажи окна", "покажи окна"),
        ("Оса́, открой хром", "открой хром"),
    ],
)
def test_wake_word_is_recognized_and_stripped(spoken: str, expected: str) -> None:
    result = _gate().process(spoken)

    assert result == GateResult(True, expected, "wake")


def test_utterance_without_wake_word_is_rejected() -> None:
    result = _gate().process("открой хром")

    assert result == GateResult(False, "", "no_wake")


def test_wake_word_beyond_leading_words_is_ignored() -> None:
    result = _gate().process("Покажи мне пожалуйста осу на картинке")

    assert result.accepted is False


def test_similar_but_different_words_do_not_wake() -> None:
    gate = _gate()

    for spoken in ("ось в телеге", "ос много летом", "оса́дить крепость", "осень пришла"):
        assert gate.process(spoken).accepted is False, spoken


def test_wake_word_alone_opens_follow_up_window() -> None:
    clock = _Clock()
    gate = _gate(clock=clock)

    first = gate.process("Оса.")

    assert first == GateResult(True, "", "wake_only")
    assert gate.follow_up_active is True
    assert gate.process("открой хром") == GateResult(True, "открой хром", "follow_up")


def test_follow_up_window_expires() -> None:
    clock = _Clock()
    gate = _gate(clock=clock)
    gate.process("Оса, привет")

    clock.now += 9.0
    assert gate.process("а теперь калькулятор").accepted is True

    gate.open_follow_up()
    clock.now += 10.5
    assert gate.process("а теперь калькулятор").accepted is False


def test_wake_word_inside_follow_up_is_still_stripped() -> None:
    gate = _gate()
    gate.open_follow_up()

    assert gate.process("Оса, закрой окно") == GateResult(True, "закрой окно", "wake")


def test_close_follow_up_requires_wake_word_again() -> None:
    gate = _gate()
    gate.open_follow_up()
    gate.close_follow_up()

    assert gate.process("открой хром").accepted is False


# --- other modes -------------------------------------------------------------------


def test_off_mode_rejects_everything() -> None:
    gate = _gate(ListenMode.OFF)

    assert gate.process("Оса, открой хром") == GateResult(False, "", "off")


def test_always_mode_accepts_everything_non_empty() -> None:
    gate = _gate(ListenMode.ALWAYS)

    assert gate.process("  открой   хром ") == GateResult(True, "открой хром", "always")
    assert gate.process("...").accepted is False


def test_empty_text_is_rejected() -> None:
    for text in ("", "   ", "...", "—"):
        assert _gate().process(text) == GateResult(False, "", "empty")


def test_changing_mode_clears_follow_up() -> None:
    gate = _gate()
    gate.open_follow_up()

    gate.mode = ListenMode.ALWAYS
    gate.mode = ListenMode.WAKE

    assert gate.follow_up_active is False
    assert gate.process("открой хром").accepted is False


# --- configuration -----------------------------------------------------------------


def test_custom_name_with_fuzzy_match() -> None:
    gate = WakeWordGate(WakeConfig(names=("джарвис",)), clock=_Clock())

    assert gate.process("Привет, как дела").accepted is False
    assert gate.process("Джарвиз, открой хром").command == "открой хром"


def test_follow_up_can_be_disabled_with_zero_seconds() -> None:
    gate = WakeWordGate(WakeConfig(follow_up_seconds=0), clock=_Clock())
    gate.process("Оса, привет")

    assert gate.process("а теперь хром").accepted is False


def test_listen_mode_parse() -> None:
    assert ListenMode.parse(" WAKE ") is ListenMode.WAKE
    assert ListenMode.parse("always") is ListenMode.ALWAYS
    assert ListenMode.parse("off") is ListenMode.OFF

    with pytest.raises(ValueError):
        ListenMode.parse("loud")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"names": ()},
        {"names": ("  ",)},
        {"max_leading_words": 0},
        {"min_similarity": 0},
        {"follow_up_seconds": -1},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        WakeConfig(**kwargs)
