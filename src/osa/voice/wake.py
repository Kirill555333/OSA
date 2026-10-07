"""Wake-word gating for background listening (OSA 1.5.2.1, names in 1.5.2.4).

Decides whether a transcribed utterance is addressed to OSA. This is activation
detection only: it looks for the assistant's names among the first words (and, for the
user's own extra name, at the end) and never interprets what the user wants. The
built-in name always works; the user may add one more name at runtime. Pure logic with
an injectable clock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import difflib
from enum import Enum
import re
import time
import unicodedata

_WORD = re.compile(r"[\w\u0300-\u036f]+", re.UNICODE)
_EDGE_PUNCTUATION = " \t\r\n,.:;!?-—–…"


class ListenMode(str, Enum):
    """How background listening decides what to accept."""

    OFF = "off"
    WAKE = "wake"
    ALWAYS = "always"

    @classmethod
    def parse(cls, value: str) -> ListenMode:
        text = str(value).strip().lower()

        for mode in cls:
            if mode.value == text:
                return mode

        raise ValueError(f"Unknown listen mode: {value!r}. Use off, wake or always.")


@dataclass(frozen=True, slots=True)
class WakeConfig:
    """Wake-word settings.

    ``names`` lists the accepted spellings of the built-in name, including the variants
    a speech recognizer tends to produce (inflections and transliteration); the first
    entry is the display name. ``trailing_names`` are built-in spellings that may also
    be used at the END of a phrase ("открой хром, Оса"). It is empty by default because
    the built-in name is also an ordinary word. Extra names set by the user are always
    accepted at both ends.
    """

    names: tuple[str, ...] = ("оса", "осы", "осу", "осе", "осой", "ося", "osa")
    trailing_names: tuple[str, ...] = ()
    max_leading_words: int = 3
    min_similarity: float = 0.8
    follow_up_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not self.names or any(not name.strip() for name in self.names):
            raise ValueError("names must contain at least one non-empty name.")

        if any(not name.strip() for name in self.trailing_names):
            raise ValueError("trailing_names cannot contain empty names.")

        if self.max_leading_words < 1:
            raise ValueError("max_leading_words must be at least 1.")

        if not 0 < self.min_similarity <= 1:
            raise ValueError("min_similarity must be in (0, 1].")

        if self.follow_up_seconds < 0:
            raise ValueError("follow_up_seconds cannot be negative.")


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of gating one utterance.

    reason is one of: off, empty, always, wake, wake_only, follow_up, no_wake.
    """

    accepted: bool
    command: str = ""
    reason: str = ""


def _normalize(word: str) -> str:
    """Casefold and drop combining marks (stress, diaeresis) for comparison only."""
    decomposed = unicodedata.normalize("NFD", word)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")

    return stripped.casefold().strip()


class WakeWordGate:
    """Accept or reject utterances according to the listening mode."""

    def __init__(
        self,
        config: WakeConfig | None = None,
        mode: ListenMode = ListenMode.WAKE,
        clock: Callable[[], float] | None = None,
        extra_names: Iterable[str] = (),
    ) -> None:
        self._config = config or WakeConfig()
        self._mode = ListenMode(mode)
        self._clock = clock or time.monotonic
        self._names = tuple(_normalize(name) for name in self._config.names)
        self._trailing = tuple(_normalize(name) for name in self._config.trailing_names)
        self._extra: tuple[str, ...] = ()
        self._follow_until = 0.0
        self.set_extra_names(extra_names)

    @property
    def mode(self) -> ListenMode:
        return self._mode

    @mode.setter
    def mode(self, value: ListenMode) -> None:
        self._mode = ListenMode(value)
        self._follow_until = 0.0

    @property
    def extra_names(self) -> tuple[str, ...]:
        """The user's extra names, normalized."""
        return self._extra

    def set_extra_names(self, names: Iterable[str]) -> None:
        """Replace the user's extra names; the built-in name is not affected."""
        cleaned = (_normalize(name) for name in names)
        self._extra = tuple(dict.fromkeys(name for name in cleaned if name))

    def display_names(self) -> tuple[str, ...]:
        """Names for display and for biasing speech recognition, built-in first."""
        primary = self._config.names[0].strip().capitalize()

        return (primary, *(name.capitalize() for name in self._extra))

    @property
    def follow_up_active(self) -> bool:
        return self._clock() < self._follow_until

    def open_follow_up(self) -> None:
        """Allow the next utterances without the wake word for a short window."""
        self._follow_until = self._clock() + self._config.follow_up_seconds

    def close_follow_up(self) -> None:
        self._follow_until = 0.0

    def process(self, text: str) -> GateResult:
        cleaned = " ".join(str(text).split())

        if self._mode is ListenMode.OFF:
            return GateResult(False, reason="off")

        if not any(char.isalnum() for char in cleaned):
            return GateResult(False, reason="empty")

        if self._mode is ListenMode.ALWAYS:
            return GateResult(True, cleaned, "always")

        found = self._find_wake(cleaned)

        if found is not None:
            position, start, end = found
            rest = cleaned[end:] if position == "lead" else cleaned[:start]
            remainder = rest.strip(_EDGE_PUNCTUATION)
            self.open_follow_up()

            if not remainder:
                return GateResult(True, "", "wake_only")

            return GateResult(True, remainder, "wake")

        if self.follow_up_active:
            return GateResult(True, cleaned, "follow_up")

        return GateResult(False, reason="no_wake")

    def _find_wake(self, text: str) -> tuple[str, int, int] | None:
        matches = list(_WORD.finditer(text))
        extra = self._extra
        leading_names = self._names + extra
        trailing_names = self._trailing + extra
        last = len(matches) - 1

        for index, match in enumerate(matches[: self._config.max_leading_words]):
            word = _normalize(match.group())

            if index == last and last >= 1:
                if self._matches(word, trailing_names):
                    return ("tail", match.start(), match.end())

                continue

            if self._matches(word, leading_names):
                return ("lead", match.start(), match.end())

        if last >= 1 and last >= self._config.max_leading_words:
            tail = matches[last]

            if self._matches(_normalize(tail.group()), trailing_names):
                return ("tail", tail.start(), tail.end())

        return None

    def _matches(self, word: str, names: tuple[str, ...]) -> bool:
        if word in names:
            return True

        return any(
            len(name) >= 4
            and difflib.SequenceMatcher(None, word, name).ratio() >= self._config.min_similarity
            for name in names
        )
