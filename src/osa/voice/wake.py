"""Wake-word gating for background listening (OSA 1.5.2.1).

Decides whether a transcribed utterance is addressed to OSA. This is activation
detection only: it looks for the assistant's name among the first words and never
interprets what the user wants. Pure logic with an injectable clock.
"""

from __future__ import annotations

from collections.abc import Callable
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

    ``names`` lists the accepted spellings of the assistant's name, including the
    variants a speech recognizer tends to produce (inflections and transliteration).
    """

    names: tuple[str, ...] = ("оса", "осы", "осу", "осе", "osa")
    max_leading_words: int = 2
    min_similarity: float = 0.8
    follow_up_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.names or any(not name.strip() for name in self.names):
            raise ValueError("names must contain at least one non-empty name.")

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
    ) -> None:
        self._config = config or WakeConfig()
        self._mode = ListenMode(mode)
        self._clock = clock or time.monotonic
        self._names = tuple(_normalize(name) for name in self._config.names)
        self._follow_until = 0.0

    @property
    def mode(self) -> ListenMode:
        return self._mode

    @mode.setter
    def mode(self, value: ListenMode) -> None:
        self._mode = ListenMode(value)
        self._follow_until = 0.0

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

        wake_end = self._find_wake_end(cleaned)

        if wake_end is not None:
            remainder = cleaned[wake_end:].strip(_EDGE_PUNCTUATION)
            self.open_follow_up()

            if not remainder:
                return GateResult(True, "", "wake_only")

            return GateResult(True, remainder, "wake")

        if self.follow_up_active:
            return GateResult(True, cleaned, "follow_up")

        return GateResult(False, reason="no_wake")

    def _find_wake_end(self, text: str) -> int | None:
        for index, match in enumerate(_WORD.finditer(text)):
            if index >= self._config.max_leading_words:
                break

            if self._is_wake_word(_normalize(match.group())):
                return match.end()

        return None

    def _is_wake_word(self, word: str) -> bool:
        if word in self._names:
            return True

        return any(
            len(name) >= 4
            and difflib.SequenceMatcher(None, word, name).ratio() >= self._config.min_similarity
            for name in self._names
        )
