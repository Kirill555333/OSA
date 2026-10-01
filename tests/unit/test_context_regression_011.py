"""End-to-end regression matrix for Context & Memory Runtime (0.8.11.9)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.context import (
    ContextBudget,
    ContextManager,
    SessionStore,
)
from osa.core import Agent
from osa.core.execution_context import ExecutionContext
from osa.memory.integration import MemoryIntegration
from osa.memory.long_term import LongTermMemory
from osa.memory.retrieval import MemoryRetriever
from osa.models import ChatMessage, ModelInterface, ModelRequest, ModelResponse, ToolCall, ToolDefinition


def _mock_model(content: str = "Response") -> MagicMock:
    model = MagicMock(spec=ModelInterface)
    model.model_name = "mock-model"
    model.generate.return_value = ModelResponse(
        content=content,
        model_name="mock-model",
    )
    return model


def test_short_chat_flow() -> None:
    """Short conversations must pass directly to model without summary overhead."""
    model = _mock_model("I can help with that.")
    agent = Agent(model=model, system_prompt="You are OSA.")

    resp1 = agent.chat("What can you do?")
    assert resp1.content == "I can help with that."

    resp2 = agent.chat("Tell me more.")
    assert resp2.content == "I can help with that."

    # Context stores system + 2 turns (5 messages total)
    assert len(agent.context.messages()) == 5
    # No summary generated for short chat
    assert agent.context_manager.current_summary.is_empty


def test_long_chat_sliding_compaction() -> None:
    """Conversations exceeding history budget must slide old turns into structured summary."""
    model = _mock_model("Acknowledged.")
    # Set a tight history budget (120 tokens)
    budget = ContextBudget.create(
        total_context_limit=1200,
        reserved_output_tokens=100,
        history_budget=120,
    )
    agent = Agent(
        model=model,
        system_prompt="System prompt",
        context_budget=budget,
    )

    for i in range(1, 15):
        agent.chat(f"Discussion step {i}: we should proceed with phase {i}")

    # Full context has accumulated all turns (system + 14*2 = 29 messages)
    assert len(agent.context.messages()) == 29

    # Summary has been populated
    assert not agent.context_manager.current_summary.is_empty
    assert agent.context_manager.current_summary.messages_summarized_count > 0

    # Model was provided bounded messages
    last_req: ModelRequest = model.generate.call_args[0][0]
    sent_msgs = last_req.messages

    # Tokens must strictly respect max input tokens
    estimator = agent.context_manager._estimator
    est_tokens = estimator.estimate_messages(sent_msgs)
    assert est_tokens <= budget.max_input_tokens

    # Verify structured summary is present in model request
    assert any("[Prior Conversation Summary]" in m.content for m in sent_msgs)


def test_huge_chat_infinite_stability() -> None:
    """A 100+ message chat must never crash or overflow the model context window."""
    model = _mock_model("Turn completed.")
    budget = ContextBudget.create(
        total_context_limit=1000,
        reserved_output_tokens=100,
        history_budget=200,
    )
    agent = Agent(
        model=model,
        system_prompt="Base assistant prompt.",
        context_budget=budget,
    )

    # Simulate 50 turns (100 user/assistant interactions)
    for i in range(50):
        agent.chat(f"User request #{i} about continuous operation")

    # Complete session history contains 101 messages
    assert len(agent.context.messages()) == 101

    # But the model request remains bounded and safe
    last_req: ModelRequest = model.generate.call_args[0][0]
    estimator = agent.context_manager._estimator
    est_tokens = estimator.estimate_messages(last_req.messages)

    assert est_tokens <= budget.max_input_tokens
    assert len(last_req.messages) < 20


def test_tool_heavy_chat_compaction() -> None:
    """Old tool outputs must be truncated while latest tool rounds remain intact."""
    tool_call1 = ToolCall(id="c1", name="fetch_data", arguments={"page": 1})
    tool_call2 = ToolCall(id="c2", name="fetch_data", arguments={"page": 2})

    model = MagicMock(spec=ModelInterface)
    model.model_name = "mock"
    # Round 1: calls tool 1 -> executes -> finalizes
    # Round 2: calls tool 2 -> executes -> finalizes
    model.generate.side_effect = [
        ModelResponse(content="", model_name="mock", tool_calls=(tool_call1,)),
        ModelResponse(content="Page 1 parsed.", model_name="mock"),
        ModelResponse(content="", model_name="mock", tool_calls=(tool_call2,)),
        ModelResponse(content="Page 2 parsed.", model_name="mock"),
    ]

    huge_tool_data = "STATUS OK: DATA LINE " * 100  # Large tool output

    tool_def = ToolDefinition(name="fetch_data", description="fetch", parameters={"type": "object"})
    agent = Agent(model=model, system_prompt="System")

    # Execute round 1 manually in context
    agent.context.add(ChatMessage(role="user", content="Fetch page 1"))
    agent.context.add(ChatMessage(role="assistant", content="", tool_calls=(tool_call1,)))
    agent.context.add(ChatMessage(role="tool", content=huge_tool_data, tool_call_id="c1"))
    agent.context.add(ChatMessage(role="assistant", content="Page 1 parsed."))

    # Execute round 2
    agent.context.add(ChatMessage(role="user", content="Fetch page 2"))
    agent.context.add(ChatMessage(role="assistant", content="", tool_calls=(tool_call2,)))
    agent.context.add(ChatMessage(role="tool", content=huge_tool_data, tool_call_id="c2"))
    agent.context.add(ChatMessage(role="assistant", content="Page 2 parsed."))

    assembled = agent.context_manager.assemble(
        agent.context.messages(),
        tools=(tool_def,),
    )

    # Tool round 1 was compacted
    tool1_msgs = [m for m in assembled.messages if m.tool_call_id == "c1"]
    assert len(tool1_msgs) == 1
    assert "...[output truncated for context budget]" in tool1_msgs[0].content

    # Tool round 2 (most recent) was protected
    tool2_msgs = [m for m in assembled.messages if m.tool_call_id == "c2"]
    assert len(tool2_msgs) == 1
    assert "...[output truncated for context budget]" not in tool2_msgs[0].content


def test_cross_session_memory_sharing_and_message_isolation(tmp_path: Path) -> None:
    """Multiple sessions share LongTermMemory while maintaining strictly isolated message histories."""
    db_file = tmp_path / "osa-memory.db"
    ltm = LongTermMemory(db_file)
    retriever = MemoryRetriever(ltm)
    integration = MemoryIntegration(retriever)

    # Store a long-term fact containing the keyword "editor"
    ltm.save("User prefers dark mode and Neovim editor", category="preference", importance=9)

    store = SessionStore()
    sess_a = store.create("session-alpha", system_prompt="Session A sys")
    sess_b = store.create("session-beta", system_prompt="Session B sys")

    # Session A records messages
    sess_a.add_message(ChatMessage(role="user", content="Top secret conversation in Alpha"))
    sess_a.add_message(ChatMessage(role="assistant", content="Understood secret Alpha"))

    # Session B records messages
    sess_b.add_message(ChatMessage(role="user", content="Regular conversation in Beta"))

    # Verify message isolation
    assert len(sess_a.messages()) == 3
    assert len(sess_b.messages()) == 2
    assert "Top secret" not in [m.content for m in sess_b.messages()]

    # Both sessions can query the same LongTermMemory
    res_a = integration.retrieve("editor")
    res_b = integration.retrieve("editor")

    assert len(res_a.memories) == 1
    assert len(res_b.memories) == 1
    assert "Neovim editor" in res_a.memories[0].memory.content
    assert "Neovim editor" in res_b.memories[0].memory.content


def test_agent_reset_preserves_memory_clears_working_summary(tmp_path: Path) -> None:
    """Agent.reset() clears session turns and summary, but keeps LongTermMemory intact."""
    db_file = tmp_path / "osa-memory.db"
    ltm = LongTermMemory(db_file)
    ltm.save("Permanent project rule: Always write tests", category="project", importance=10)

    model = _mock_model("Done")
    agent = Agent(model=model, system_prompt="You are OSA.")

    agent.chat("First conversation turn")
    assert len(agent.context.messages()) == 3

    # Reset agent
    agent.reset()

    # Raw context reset to system prompt only
    assert len(agent.context.messages()) == 1
    assert agent.context.messages()[0].role == "system"
    # Summary is reset
    assert agent.context_manager.current_summary.is_empty
    # LongTermMemory still has the permanent fact
    assert ltm.count() == 1
    record = ltm.get(1)
    assert record.content == "Permanent project rule: Always write tests"


def test_voice_and_autonomous_execution_context_invariants() -> None:
    """ExecutionContext contracts for voice and autonomous flows remain strictly preserved."""
    # Voice invariant
    voice_req = ActionRequest(
        kind=ActionKind.TOOL,
        name="calculator",
        arguments={"expression": "10 * 10"},
        request_id="voice-req-001",
        metadata={"source": "voice", "voice_task_id": "vtask-42"},
    )
    voice_ctx = ExecutionContext.from_action_request(voice_req)
    assert voice_ctx.request_id == "voice-req-001"
    assert voice_ctx.source == "voice"
    assert voice_ctx.task_id == "vtask-42"
    assert voice_ctx.round_number is None

    # Autonomous invariant
    auto_req = ActionRequest(
        kind=ActionKind.TOOL,
        name="browser_open",
        arguments={"url": "https://example.com"},
        request_id="auto-req-002",
        metadata={},
    )
    auto_ctx = ExecutionContext.from_autonomous_action_request(
        auto_req,
        run_id="run-auto-1",
        task_id="task-auto-1",
    )
    assert auto_ctx.request_id == "auto-req-002"
    assert auto_ctx.source == "autonomous"
    assert auto_ctx.run_id == "run-auto-1"
    assert auto_ctx.task_id == "task-auto-1"
    assert auto_ctx.round_number is None
