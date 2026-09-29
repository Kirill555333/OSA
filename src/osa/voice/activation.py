"""Voice activation policies for OSA 0.7.7."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from osa.voice.contracts import VoiceTranscript


class VoiceActivationError(
    RuntimeError
):
    """Raised when voice activation cannot be evaluated."""


@dataclass(frozen=True)
class VoiceActivationResult:
    """Result of evaluating a transcript for activation."""

    activated: bool
    command_text: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.activated,
            bool,
        ):
            raise VoiceActivationError(
                "activated must be boolean."
            )

        if not isinstance(
            self.command_text,
            str,
        ):
            raise VoiceActivationError(
                "command_text must be a string."
            )

        object.__setattr__(
            self,
            "command_text",
            _normalize_text(
                self.command_text
            ),
        )


@runtime_checkable
class VoiceActivationDetector(Protocol):
    """Provider-neutral transcript activation interface."""

    def detect(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceActivationResult:
        ...


class AlwaysActiveVoiceActivationDetector:
    """Default activation policy preserving existing OSA behavior."""

    def detect(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceActivationResult:
        if not isinstance(
            transcript,
            VoiceTranscript,
        ):
            raise TypeError(
                "transcript must be a VoiceTranscript."
            )

        return VoiceActivationResult(
            activated=True,
            command_text=transcript.text,
        )


class KeywordVoiceActivationDetector:
    """
    Activate when one of the configured wake phrases occurs.

    Matching is case-insensitive and whitespace-normalized. When activated,
    the matching phrase and surrounding separators are removed before the
    command reaches the Agent.
    """

    def __init__(
        self,
        phrases: tuple[str, ...] | list[str],
        *,
        strip_phrase: bool = True,
    ) -> None:
        normalized_phrases: list[str] = []

        for phrase in phrases:
            if not isinstance(
                phrase,
                str,
            ):
                raise ValueError(
                    "Activation phrases must be strings."
                )

            normalized = _normalize_text(
                phrase
            ).casefold()

            if normalized and normalized not in normalized_phrases:
                normalized_phrases.append(
                    normalized
                )

        if not normalized_phrases:
            raise ValueError(
                "At least one activation phrase is required."
            )

        if not isinstance(
            strip_phrase,
            bool,
        ):
            raise ValueError(
                "strip_phrase must be boolean."
            )

        self._phrases = tuple(
            normalized_phrases
        )
        self._strip_phrase = strip_phrase

    @property
    def phrases(self) -> tuple[str, ...]:
        """Return normalized activation phrases."""
        return self._phrases

    @property
    def strip_phrase(self) -> bool:
        """Return whether the wake phrase is removed from the command."""
        return self._strip_phrase

    def detect(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceActivationResult:
        if not isinstance(
            transcript,
            VoiceTranscript,
        ):
            raise TypeError(
                "transcript must be a VoiceTranscript."
            )

        text = _normalize_text(
            transcript.text
        )

        folded_text = text.casefold()

        for phrase in self._phrases:
            pattern = re.compile(
                rf"(?<!\w){re.escape(phrase)}(?!\w)",
                re.IGNORECASE,
            )

            match = pattern.search(
                folded_text
            )

            if match is None:
                continue

            command_text = (
                _remove_match(
                    text,
                    match.start(),
                    match.end(),
                )
                if self._strip_phrase
                else text
            )

            return VoiceActivationResult(
                activated=True,
                command_text=command_text or text,
            )

        return VoiceActivationResult(
            activated=False,
            command_text="",
        )


def create_always_active_detector() -> AlwaysActiveVoiceActivationDetector:
    """Create the default no-gating activation detector."""
    return AlwaysActiveVoiceActivationDetector()


def create_keyword_activation_detector(
    *phrases: str,
    strip_phrase: bool = True,
) -> KeywordVoiceActivationDetector:
    """Create a keyword-based activation detector."""
    return KeywordVoiceActivationDetector(
        phrases,
        strip_phrase=strip_phrase,
    )


def _normalize_text(
    text: str,
) -> str:
    normalized = " ".join(
        text.strip().split()
    )

    return re.sub(
        r"\s+([,;:.!?])",
        r"\1",
        normalized,
    )


def _remove_match(
    text: str,
    start: int,
    end: int,
) -> str:
    command = (
        text[:start]
        + " "
        + text[end:]
    )

    command = re.sub(
        r"^[\s,;:!?—–-]+",
        "",
        command,
    )
    command = re.sub(
        r"[\s,;:!?—–-]+$",
        "",
        command,
    )

    return _normalize_text(
        command
    )
