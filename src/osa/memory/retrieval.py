"""Memory retrieval services for OSA."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from osa.memory.long_term import LongTermMemory, MemoryRecord


_STOP_WORDS = frozenset(
    {
        # English.
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "do",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "that",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "you",
        "your",
        # Russian.
        "а",
        "без",
        "бы",
        "в",
        "во",
        "вот",
        "вы",
        "где",
        "да",
        "для",
        "до",
        "его",
        "ее",
        "же",
        "за",
        "и",
        "из",
        "или",
        "к",
        "как",
        "какая",
        "какие",
        "какой",
        "какое",
        "кто",
        "мне",
        "меня",
        "может",
        "на",
        "над",
        "не",
        "но",
        "о",
        "об",
        "один",
        "он",
        "она",
        "они",
        "по",
        "при",
        "про",
        "с",
        "со",
        "так",
        "тебе",
        "ты",
        "у",
        "хочу",
        "что",
        "это",
        "я",
    }
)

_TOKEN_PATTERN = re.compile(
    r"\w+",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A memory together with its retrieval score."""

    memory: MemoryRecord
    score: float


class MemoryRetriever:
    """Retrieve relevant memories using strict and fuzzy keyword matching."""

    def __init__(
        self,
        memory: LongTermMemory,
        *,
        max_candidates_per_term: int = 20,
    ) -> None:
        if max_candidates_per_term <= 0:
            raise ValueError(
                "max_candidates_per_term must be greater than zero."
            )

        self._memory = memory
        self._max_candidates_per_term = (
            max_candidates_per_term
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> tuple[MemorySearchResult, ...]:
        """Search memories using strict matching with fuzzy fallback."""
        normalized_query = self._normalize_text(
            query
        ).strip()

        if not normalized_query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        terms = self._keyword_terms(
            normalized_query
        )

        if not terms:
            return ()

        strict_results = self._strict_search(
            terms,
            limit=limit,
        )

        if strict_results:
            return strict_results

        return self._fuzzy_search(
            terms,
            limit=limit,
        )

    def _strict_search(
        self,
        terms: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MemorySearchResult, ...]:
        """Return memories matching every significant term."""
        try:
            records = self._memory.search(
                " ".join(terms),
                limit=max(
                    limit,
                    self._max_candidates_per_term,
                ),
            )
        except ValueError:
            return ()

        results: list[MemorySearchResult] = []

        for record in records:
            score = self._score_record(
                record,
                terms,
                exact_only=True,
            )

            if score is None:
                continue

            results.append(
                MemorySearchResult(
                    memory=record,
                    score=score,
                )
            )

        return self._sort_and_limit(
            results,
            limit,
        )

    def _fuzzy_search(
        self,
        terms: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MemorySearchResult, ...]:
        """Return best-effort matches when strict search finds nothing."""
        candidates: dict[int, MemoryRecord] = {}

        for term in terms:
            relaxed_term = self._relaxed_term(
                term
            )

            if not relaxed_term:
                continue

            records = self._memory.search(
                relaxed_term,
                limit=self._max_candidates_per_term,
            )

            for record in records:
                candidates[record.id] = record

        results: list[MemorySearchResult] = []

        for record in candidates.values():
            score = self._score_record(
                record,
                terms,
                exact_only=False,
            )

            if score is None:
                continue

            results.append(
                MemorySearchResult(
                    memory=record,
                    score=score,
                )
            )

        return self._sort_and_limit(
            results,
            limit,
        )

    def _score_record(
        self,
        record: MemoryRecord,
        terms: tuple[str, ...],
        *,
        exact_only: bool,
    ) -> float | None:
        """Calculate relevance for one memory."""
        content = self._normalize_text(
            record.content
        )
        category = self._normalize_text(
            record.category
        )

        searchable_text = (
            f"{content} {category}"
        )

        tokens = _TOKEN_PATTERN.findall(
            searchable_text
        )

        matched_terms = 0

        for term in terms:
            if self._term_matches(
                term,
                tokens,
                exact_only=exact_only,
            ):
                matched_terms += 1

        if matched_terms == 0:
            return None

        if exact_only and matched_terms != len(terms):
            return None

        phrase_bonus = 0.0

        normalized_phrase = " ".join(terms)

        if normalized_phrase in searchable_text:
            phrase_bonus = 2.0

        importance_bonus = (
            record.importance / 10
        )

        return (
            matched_terms
            + phrase_bonus
            + importance_bonus
        )

    @staticmethod
    def _term_matches(
        term: str,
        tokens: list[str],
        *,
        exact_only: bool,
    ) -> bool:
        """Match a term against tokenized memory text."""
        if exact_only:
            return term in tokens

        relaxed_term = MemoryRetriever._relaxed_term(
            term
        )

        if not relaxed_term:
            return False

        return any(
            token.startswith(relaxed_term)
            for token in tokens
        )

    @staticmethod
    def _relaxed_term(
        term: str,
    ) -> str:
        """Return a conservative prefix for fuzzy matching."""
        if len(term) < 6:
            return term

        prefix_length = max(
            4,
            len(term) - 2,
        )

        return term[:prefix_length]

    @staticmethod
    def _keyword_terms(
        query: str,
    ) -> tuple[str, ...]:
        """Extract meaningful Unicode-aware keywords."""
        tokens = _TOKEN_PATTERN.findall(
            query
        )

        return tuple(
            dict.fromkeys(
                token
                for token in tokens
                if len(token) >= 3
                and token not in _STOP_WORDS
            )
        )

    @staticmethod
    def _sort_and_limit(
        results: list[MemorySearchResult],
        limit: int,
    ) -> tuple[MemorySearchResult, ...]:
        """Sort results by relevance and apply the result limit."""
        results.sort(
            key=lambda result: (
                -result.score,
                -result.memory.importance,
                result.memory.id,
            )
        )

        return tuple(
            results[:limit]
        )

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """Normalize text for Unicode-aware matching."""
        return unicodedata.normalize(
            "NFKC",
            value,
        ).casefold()
