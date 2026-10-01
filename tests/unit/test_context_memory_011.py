"""Unit tests for BudgetedMemoryRetriever (0.8.11.4)."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from osa.context import (
    BudgetedMemoryResult,
    BudgetedMemoryRetriever,
    ConservativeTokenEstimator,
)
from osa.memory.integration import MemoryIntegration, MemoryIntegrationResult
from osa.memory.long_term import MemoryRecord
from osa.memory.retrieval import MemoryRetriever, MemorySearchResult


def _make_record(mem_id: int, content: str, category: str = "general", importance: int = 5) -> MemoryRecord:
    return MemoryRecord(
        id=mem_id,
        content=content,
        category=category,
        importance=importance,
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )


def test_budgeted_memory_retriever_validation() -> None:
    mock_retriever = MagicMock(spec=MemoryRetriever)
    with pytest.raises(ValueError, match="max_candidates must be greater than zero"):
        BudgetedMemoryRetriever(mock_retriever, max_candidates=0)

    retriever = BudgetedMemoryRetriever(mock_retriever)
    with pytest.raises(ValueError, match="token_budget must be greater than zero"):
        retriever.retrieve("test query", token_budget=0)


def test_budgeted_memory_retriever_empty_query() -> None:
    mock_retriever = MagicMock(spec=MemoryRetriever)
    retriever = BudgetedMemoryRetriever(mock_retriever)

    result = retriever.retrieve("", token_budget=100)
    assert result.memories == ()
    assert result.prompt is None
    assert result.estimated_tokens == 0
    mock_retriever.search.assert_not_called()


def test_budgeted_memory_retriever_fits_within_budget() -> None:
    mock_retriever = MagicMock(spec=MemoryRetriever)
    candidates = (
        MemorySearchResult(memory=_make_record(1, "User prefers dark mode", "preference"), score=3.0),
        MemorySearchResult(memory=_make_record(2, "Project uses Python 3.12", "project"), score=2.5),
        MemorySearchResult(
            memory=_make_record(3, "Long historical fact about previous deployments in the cloud", "history"),
            score=1.0,
        ),
    )
    mock_retriever.search.return_value = candidates

    retriever = BudgetedMemoryRetriever(mock_retriever)
    # Generous budget: fits all
    result = retriever.retrieve("preferences", token_budget=500)

    assert len(result.memories) == 3
    assert result.prompt is not None
    assert "User prefers dark mode" in result.prompt
    assert "Project uses Python 3.12" in result.prompt
    assert result.estimated_tokens <= 500


def test_budgeted_memory_retriever_enforces_budget_boundary() -> None:
    mock_retriever = MagicMock(spec=MemoryRetriever)
    candidates = (
        MemorySearchResult(memory=_make_record(1, "Fact one about user preferences", "preference"), score=3.0),
        MemorySearchResult(
            memory=_make_record(2, "Fact two with a very long descriptive explanation that takes many tokens", "project"),
            score=2.0,
        ),
        MemorySearchResult(memory=_make_record(3, "Fact three", "general"), score=1.0),
    )
    mock_retriever.search.return_value = candidates

    retriever = BudgetedMemoryRetriever(mock_retriever)
    estimator = ConservativeTokenEstimator()
    header_cost = estimator.estimate_text(BudgetedMemoryRetriever.MEMORY_HEADER)

    # Budget allows header + first candidate, but not second
    first_line_cost = estimator.estimate_text("\n- [Memory #1] (preference): Fact one about user preferences")
    tight_budget = header_cost + first_line_cost + 4

    result = retriever.retrieve("query", token_budget=tight_budget)

    assert len(result.memories) == 1
    assert result.memories[0].memory.id == 1
    assert result.prompt is not None
    assert "Fact one about user preferences" in result.prompt
    assert "Fact two with a very long" not in result.prompt
    assert result.estimated_tokens <= tight_budget


def test_budgeted_memory_retriever_budget_too_small_for_header() -> None:
    mock_retriever = MagicMock(spec=MemoryRetriever)
    mock_retriever.search.return_value = (
        MemorySearchResult(memory=_make_record(1, "Fact"), score=1.0),
    )

    retriever = BudgetedMemoryRetriever(mock_retriever)
    # Extremely small budget (e.g. 5 tokens, less than safety header)
    result = retriever.retrieve("query", token_budget=5)

    assert result.memories == ()
    assert result.prompt is None
    assert result.estimated_tokens == 0


def test_budgeted_memory_retriever_with_memory_integration() -> None:
    mock_integration = MagicMock(spec=MemoryIntegration)
    candidates = (
        MemorySearchResult(memory=_make_record(10, "Fact via integration", "platform"), score=2.0),
    )
    mock_integration.retrieve.return_value = MemoryIntegrationResult(
        memories=candidates,
        prompt="raw",
    )

    retriever = BudgetedMemoryRetriever(mock_integration)
    result = retriever.retrieve("platform", token_budget=200)

    assert len(result.memories) == 1
    assert result.memories[0].memory.id == 10
    assert result.prompt is not None
    assert "Fact via integration" in result.prompt
    assert result.estimated_tokens <= 200
