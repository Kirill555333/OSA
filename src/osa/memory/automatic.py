"""Conservative automatic long-term memory capture for OSA."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from osa.memory.long_term import LongTermMemory


_TOKEN_PATTERN = re.compile(
    r"\w+",
    re.UNICODE,
)

_EXPLICIT_MEMORY_PATTERN = re.compile(
    r"\b(?:"
    r"запомни|"
    r"не\s+забудь|"
    r"remember|"
    r"don't\s+forget|"
    r"save\s+this"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_START_PATTERN = re.compile(
    r"^\s*(?:"
    r"что|как|почему|зачем|где|когда|кто|"
    r"какой|какая|какое|какие|"
    r"what|how|why|where|when|who|which"
    r")\b",
    re.IGNORECASE,
)

_TRANSIENT_PATTERN = re.compile(
    r"\b(?:"
    r"сегодня|"
    r"сейчас|"
    r"завтра|"
    r"вчера|"
    r"временно|"
    r"на\s+этой\s+неделе|"
    r"today|"
    r"now|"
    r"tomorrow|"
    r"yesterday|"
    r"temporarily|"
    r"this\s+week"
    r")\b",
    re.IGNORECASE,
)

_SENSITIVE_PATTERN = re.compile(
    r"\b(?:"
    r"парол\w*|"
    r"password\w*|"
    r"api[\s_-]*key\w*|"
    r"token\w*|"
    r"секрет\w*|"
    r"secret\w*|"
    r"private[\s_-]*key\w*|"
    r"ssh[\s_-]*key\w*|"
    r"seed[\s_-]*phrase\w*|"
    r"recovery[\s_-]*phrase\w*|"
    r"телефон\w*|"
    r"номер\w*|"
    r"email\w*|"
    r"e-mail\w*|"
    r"адрес\w*"
    r")\b",
    re.IGNORECASE,
)

_PREFERENCE_PATTERNS = (
    re.compile(
        r"\bя\s+(?:предпочитаю|люблю|не\s+люблю|выбираю)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+(?:prefer|love|usually\s+choose)\b",
        re.IGNORECASE,
    ),
)

_PLATFORM_PATTERNS = (
    re.compile(
        r"\b(?:"
        r"целевая|основная|целевой"
        r")\s+(?:ос|"
        r"операционная\s+система|"
        r"платформа|"
        r"компьютер)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bя\s+использую\s+"
        r"(?:windows|macos|mac|linux|ubuntu)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:target\s+runtime|target\s+platform)\b",
        re.IGNORECASE,
    ),
)

_PROJECT_PATTERNS = (
    re.compile(
        r"\b(?:"
        r"мой|моя|мое|мой\s+основной"
        r")\s+проект\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:название|имя)\s+проекта\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:"
        r"основной\s+язык|"
        r"язык\s+разработки|"
        r"язык\s+программирования"
        r")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bOSA\b.*\b(?:"
        r"использует|"
        r"разрабатывается|"
        r"работает|"
        r"uses|"
        r"developed|"
        r"built"
        r")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+project\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bproject\s+(?:is\s+called|is\s+named|called|named)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bprimary\s+development\s+language\b",
        re.IGNORECASE,
    ),
)

_USER_FACT_PATTERNS = (
    re.compile(
        r"\bя\s+(?:работаю|учусь|разрабатываю|пишу)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+(?:work|study|develop|write)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A conservative automatic-memory candidate."""

    content: str
    category: str
    importance: int


@dataclass(frozen=True, slots=True)
class AutomaticMemoryResult:
    """Result of an automatic-memory capture attempt."""

    saved: bool
    memory_id: int | None
    reason: str


class AutomaticMemory:
    """Capture high-confidence, non-sensitive long-term user facts."""

    def __init__(
        self,
        memory: LongTermMemory,
        *,
        duplicate_check_limit: int = 50,
        min_length: int = 12,
        max_length: int = 500,
    ) -> None:
        if duplicate_check_limit <= 0:
            raise ValueError(
                "duplicate_check_limit must be greater than zero."
            )

        if min_length <= 0:
            raise ValueError(
                "min_length must be greater than zero."
            )

        if max_length < min_length:
            raise ValueError(
                "max_length must be greater than or equal to min_length."
            )

        self._memory = memory
        self._duplicate_check_limit = duplicate_check_limit
        self._min_length = min_length
        self._max_length = max_length

    def capture(
        self,
        user_input: str,
    ) -> AutomaticMemoryResult:
        """Capture one stable fact when confidence is high enough."""
        content = self._normalize_content(
            user_input
        )

        if not content:
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="empty",
            )

        if len(content) < self._min_length:
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="too_short",
            )

        if len(content) > self._max_length:
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="too_long",
            )

        if _EXPLICIT_MEMORY_PATTERN.search(content):
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="explicit_memory_request",
            )

        if "?" in content or _QUESTION_START_PATTERN.search(
            content
        ):
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="question",
            )

        if _TRANSIENT_PATTERN.search(content):
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="transient_content",
            )

        if _SENSITIVE_PATTERN.search(content):
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="sensitive_content",
            )

        candidate = self._extract_candidate(
            content
        )

        if candidate is None:
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="not_stable_fact",
            )

        if self._is_duplicate(
            candidate.content
        ):
            return AutomaticMemoryResult(
                saved=False,
                memory_id=None,
                reason="duplicate",
            )

        record = self._memory.save(
            candidate.content,
            category=candidate.category,
            importance=candidate.importance,
        )

        return AutomaticMemoryResult(
            saved=True,
            memory_id=record.id,
            reason="saved",
        )

    def _extract_candidate(
        self,
        content: str,
    ) -> MemoryCandidate | None:
        """Classify high-confidence stable facts."""
        if self._matches_any(
            _PREFERENCE_PATTERNS,
            content,
        ):
            return MemoryCandidate(
                content=content,
                category="preference",
                importance=8,
            )

        if self._matches_any(
            _PLATFORM_PATTERNS,
            content,
        ):
            return MemoryCandidate(
                content=content,
                category="platform",
                importance=8,
            )

        if self._matches_any(
            _PROJECT_PATTERNS,
            content,
        ):
            return MemoryCandidate(
                content=content,
                category="project",
                importance=8,
            )

        if self._matches_any(
            _USER_FACT_PATTERNS,
            content,
        ):
            return MemoryCandidate(
                content=content,
                category="user",
                importance=7,
            )

        return None

    def _is_duplicate(
        self,
        content: str,
    ) -> bool:
        """Check recent memories for an exact normalized duplicate."""
        normalized_content = (
            self._normalize_for_comparison(
                content
            )
        )

        for record in self._memory.recent(
            limit=self._duplicate_check_limit
        ):
            existing = self._normalize_for_comparison(
                record.content
            )

            if existing == normalized_content:
                return True

        return False

    @staticmethod
    def _matches_any(
        patterns: tuple[re.Pattern[str], ...],
        content: str,
    ) -> bool:
        """Return whether any pattern matches the content."""
        return any(
            pattern.search(content)
            for pattern in patterns
        )

    @staticmethod
    def _normalize_content(
        value: str,
    ) -> str:
        """Normalize whitespace and Unicode composition."""
        normalized = unicodedata.normalize(
            "NFKC",
            value,
        )

        return " ".join(
            normalized.split()
        )

    @staticmethod
    def _normalize_for_comparison(
        value: str,
    ) -> str:
        """Normalize content for duplicate comparison."""
        normalized = unicodedata.normalize(
            "NFKC",
            value,
        ).casefold()

        return " ".join(
            normalized.split()
        )
