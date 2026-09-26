"""Memory components for OSA."""

from osa.memory.long_term import (
    LongTermMemory,
    MemoryError,
    MemoryRecord,
)
from osa.memory.retrieval import (
    MemoryRetriever,
    MemorySearchResult,
)

from osa.memory.automatic import (
    AutomaticMemory,
    AutomaticMemoryResult,
    MemoryCandidate,
)

__all__ = [
    "LongTermMemory",
    "MemoryError",
    "MemoryRecord",
    "MemoryRetriever",
    "MemorySearchResult",
    "AutomaticMemory",
    "AutomaticMemoryResult",
    "MemoryCandidate",
]
