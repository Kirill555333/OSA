"""Memory retrieval services for OSA."""

from __future__ import annotations

from dataclasses import dataclass

from osa.memory.long_term import LongTermMemory, MemoryRecord


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A memory together with its retrieval score."""

    memory: MemoryRecord
    score: float


class MemoryRetriever:
    """Retrieve relevant memories from long-term storage."""

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> tuple[MemorySearchResult, ...]:
        """Search memories and assign simple relevance scores."""
        records = self._memory.search(
            query,
            limit=limit,
        )

        normalized_query = query.strip().lower()
        words = {
            word
            for word in normalized_query.split()
            if word
        }

        results: list[MemorySearchResult] = []

        for record in records:
            content = record.content.lower()

            matched_words = sum(
                1
                for word in words
                if word in content
            )

            score = (
                matched_words
                + record.importance / 10
            )

            results.append(
                MemorySearchResult(
                    memory=record,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                -result.memory.importance,
                result.memory.id,
            )
        )

        return tuple(results)
