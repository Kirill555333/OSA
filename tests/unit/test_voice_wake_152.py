"""Tests for wake-word gating (OSA 1.5.2.1, extra names in 1.5.2.4)."""

from __future__ import annotations

import pytest

from osa.voice.wake import GateResult, ListenMode, WakeConfig, WakeWordGate


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _gate(
    mode: ListenMode = ListenMode.WAKE,
    clock: _Clock | None = None,
    extra_names: tuple[str, ...] = (),
) -> WakeWordGate:
    return WakeWordGate(mode=mode, clock=clock or _Clock(), extra_names=extra_names)


# --- built-in name -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("Оса, открой хром", "открой хром"),
        ("оса открой хром", "открой хром"),
        ("Осы, открой хром.", "открой хром"),
        ("Ося, открой хром", "открой хром"),
        ("Эй, Оса, открой хром", "открой хром"),
        ("Эй, слушай, Оса, открой хром", "открой хром"),
        ("ОСА! Закрой калькулятор", "Закрой калькулятор"),
        ("Osa open the calculator", "open the calculator"),
        ("Оса — покажи окна", "покажи окна"),
        ("Оса\u0301, открой хром", "открой хром"),
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
    stressed = "оса\u0301дить крепость"

    for spoken in ("ось в телеге", "ос много летом", stressed, "осень пришла"):
        assert gate.process(spoken).accepted is False, spoken


def test_builtin_name_at_the_end_of_a_phrase_does_not_wake_by_default() -> None:
    gate = _gate()

    for spoken in ("Открой хром, Оса", "Это была оса", "Я видел осу"):
        assert gate.process(spoken).accepted is False, spoken


def test_builtin_name_at_the_end_can_be_enabled_in_config() -> None:
    gate = WakeWordGate(WakeConfig(trailing_names=("оса",)), clock=_Clock())

    assert gate.process("Открой хром, Оса") == GateResult(True, "Открой хром", "wake")

    gate.close_follow_up()

    assert gate.process("Я видел осу").accepted is False


def test_wake_word_alone_opens_follow_up_window() -> None:
    clock = _Clock()
    gate = _gate(clock=clock)

    first = gate.process("Оса.")

    assert first == GateResult(True, "", "wake_only")
    assert gate.follow_up_active is True
    assert gate.process("открой хром") == GateResult(True, "открой хром", "follow_up")


def test_follow_up_window_defaults_to_twenty_seconds() -> None:
    clock = _Clock()
    gate = _gate(clock=clock)
    gate.process("Оса, привет")

    clock.now += 19.0
    assert gate.process("а теперь калькулятор").accepted is True

    gate.open_follow_up()
    clock.now += 21.0
    assert gate.process("а теперь калькулятор").accepted is False


def test_follow_up_window_expires() -> None:
    clock = _Clock()
    gate = WakeWordGate(WakeConfig(follow_up_seconds=10.0), clock=clock)
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


# --- extra (user) names ------------------------------------------------------------


def test_extra_name_wakes_and_builtin_name_keeps_working() -> None:
    gate = _gate(extra_names=("джарвис",))

    assert gate.process("Джарвис, открой хром") == GateResult(True, "открой хром", "wake")
    assert gate.process("Оса, открой хром") == GateResult(True, "открой хром", "wake")


def test_extra_name_tolerates_recognition_errors() -> None:
    gate = _gate(extra_names=("джарвис",))

    assert gate.process("Джарвиз, открой хром").command == "открой хром"


def test_extra_name_works_at_the_end_of_a_phrase() -> None:
    gate = _gate(extra_names=("джарвис",))

    assert gate.process("Открой хром, Джарвис.") == GateResult(True, "Открой хром", "wake")
    assert gate.process("Слушай мне нужен хром, Джарвис") == GateResult(
        True,
        "Слушай мне нужен хром",
        "wake",
    )


def test_extra_name_alone_is_wake_only() -> None:
    gate = _gate(extra_names=("джарвис",))

    assert gate.process("Джарвис") == GateResult(True, "", "wake_only")


def test_extra_names_can_be_replaced_and_cleared_at_runtime() -> None:
    gate = _gate(extra_names=("джарвис",))

    gate.set_extra_names(["пятница"])

    assert gate.extra_names == ("пятница",)
    assert gate.process("Пятница, привет").accepted is True

    gate.close_follow_up()

    assert gate.process("Джарвис, привет").accepted is False

    gate.set_extra_names([])

    assert gate.extra_names == ()
    assert gate.process("Пятница, привет").accepted is False
    assert gate.process("Оса, привет").accepted is True


def test_extra_names_are_normalized_and_deduplicated() -> None:
    gate = _gate(extra_names=("Джарвис", " джарвис ", ""))

    assert gate.extra_names == ("джарвис",)


def test_display_names_list_builtin_first() -> None:
    assert _gate().display_names() == ("Оса",)
    assert _gate(extra_names=("джарвис",)).display_names() == ("Оса", "Джарвис")


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


def test_custom_builtin_name_with_fuzzy_match() -> None:
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
        {"trailing_names": ("",)},
        {"max_leading_words": 0},
        {"min_similarity": 0},
        {"follow_up_seconds": -1},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        WakeConfig(**kwargs)
