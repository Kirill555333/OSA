from __future__ import annotations

import pytest

from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableAgent,
)
from osa.tasks.autonomous import AutonomousRunReport


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _Agent:
    def __init__(
        self,
        *,
        response: object | None = None,
        chat_error: Exception | None = None,
        autonomous_result: object | None = None,
        autonomous_error: Exception | None = None,
        chunks: tuple[str, ...] = (),
        stream_error: Exception | None = None,
    ) -> None:
        self.response = (
            _Response("hello")
            if response is None
            else response
        )
        self.chat_error = chat_error
        self.autonomous_result = (
            autonomous_result
            if autonomous_result is not None
            else AutonomousRunReport(
                goal="goal",
                run_id="run-1",
                success=True,
                stopped=False,
                stop_reason=None,
            )
        )
        self.autonomous_error = autonomous_error
        self.chunks = chunks
        self.stream_error = stream_error
        self.chat_calls: list[str] = []
        self.stream_calls: list[str] = []
        self.autonomous_calls: list[str] = []

    def chat(self, user_input: str):
        self.chat_calls.append(user_input)

        if self.chat_error is not None:
            raise self.chat_error

        return self.response

    def chat_stream(self, user_input: str):
        self.stream_calls.append(user_input)

        for chunk in self.chunks:
            yield chunk

        if self.stream_error is not None:
            raise self.stream_error

    def run_autonomous(self, goal: str):
        self.autonomous_calls.append(goal)

        if self.autonomous_error is not None:
            raise self.autonomous_error

        return self.autonomous_result


def test_chat_emits_started_and_completed() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent()

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    result = observed.chat("hello world")

    assert result.content == "hello"
    assert agent.chat_calls == ["hello world"]

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.chat.started",
        "agent.chat.completed",
    ]
    assert events[0].status == "started"
    assert events[-1].status == "completed"
    assert events[-1].metadata["response_length"] == 5


def test_chat_exception_is_logged_and_reraised() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent(
        chat_error=RuntimeError("chat exploded")
    )

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    with pytest.raises(
        RuntimeError,
        match="chat exploded",
    ):
        observed.chat("hello")

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.chat.started",
        "agent.chat.failed",
    ]
    assert events[-1].error == "chat exploded"


def test_streaming_emits_single_started_and_completed_pair() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent(
        chunks=("Hel", "lo", "!")
    )

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    result = "".join(
        observed.chat_stream("stream me")
    )

    assert result == "Hello!"
    assert agent.stream_calls == ["stream me"]

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.chat.started",
        "agent.chat.completed",
    ]
    assert events[0].metadata["streaming"] is True
    assert events[-1].metadata["emitted_length"] == 6


def test_streaming_exception_is_logged_and_reraised() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent(
        chunks=("before",),
        stream_error=RuntimeError(
            "stream exploded"
        ),
    )

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    with pytest.raises(
        RuntimeError,
        match="stream exploded",
    ):
        list(
            observed.chat_stream("stream")
        )

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.chat.started",
        "agent.chat.failed",
    ]
    assert events[-1].error == "stream exploded"


def test_autonomous_success_emits_completed() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent()

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    result = observed.run_autonomous(
        "finish the task"
    )

    assert result.success is True
    assert agent.autonomous_calls == [
        "finish the task"
    ]

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.autonomous.started",
        "agent.autonomous.completed",
    ]
    assert events[-1].status == "completed"
    assert events[-1].metadata["success"] is True


def test_autonomous_failed_report_emits_failed() -> None:
    logger = InMemoryObservabilityLogger()

    failed_report = AutonomousRunReport(
        goal="goal",
        run_id="run-failed",
        success=False,
        stopped=False,
        stop_reason="task_failed",
    )

    agent = _Agent(
        autonomous_result=failed_report
    )

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    result = observed.run_autonomous(
        "fail"
    )

    assert result.success is False

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.autonomous.started",
        "agent.autonomous.failed",
    ]
    assert events[-1].status == "failed"
    assert events[-1].metadata["success"] is False


def test_autonomous_exception_is_logged_and_reraised() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent(
        autonomous_error=RuntimeError(
            "autonomous exploded"
        )
    )

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    with pytest.raises(
        RuntimeError,
        match="autonomous exploded",
    ):
        observed.run_autonomous("boom")

    events = logger.events()

    assert [event.event for event in events] == [
        "agent.autonomous.started",
        "agent.autonomous.failed",
    ]
    assert events[-1].error == "autonomous exploded"


def test_input_and_goal_are_normalized() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent()

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    observed.chat("  hello  ")
    observed.run_autonomous("  goal  ")

    assert agent.chat_calls == ["hello"]
    assert agent.autonomous_calls == ["goal"]


def test_empty_input_is_rejected() -> None:
    observed = ObservableAgent(
        _Agent()
    )

    with pytest.raises(
        ValueError,
        match="user_input cannot be empty",
    ):
        observed.chat("   ")


def test_empty_goal_is_rejected() -> None:
    observed = ObservableAgent(
        _Agent()
    )

    with pytest.raises(
        ValueError,
        match="goal cannot be empty",
    ):
        observed.run_autonomous("   ")


def test_logger_identity_is_preserved_when_empty() -> None:
    logger = InMemoryObservabilityLogger()

    observed = ObservableAgent(
        _Agent(),
        logger=logger,
    )

    assert observed.logger is logger


def test_logging_failure_does_not_break_agent() -> None:
    class _BrokenLogger:
        def log(self, event, **data) -> None:
            raise RuntimeError("logger broken")

    observed = ObservableAgent(
        _Agent(),
        logger=_BrokenLogger(),
    )

    result = observed.chat("hello")

    assert result.content == "hello"


def test_raw_user_input_is_not_logged() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent()

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    secret = "TOP_SECRET_USER_INPUT"

    observed.chat(secret)

    for event in logger.events():
        assert secret not in str(event.to_dict())


def test_raw_autonomous_goal_is_not_logged() -> None:
    logger = InMemoryObservabilityLogger()
    agent = _Agent()

    observed = ObservableAgent(
        agent,
        logger=logger,
    )

    secret = "TOP_SECRET_AUTONOMOUS_GOAL"

    observed.run_autonomous(secret)

    for event in logger.events():
        assert secret not in str(event.to_dict())


def test_agent_property_preserves_wrapped_instance() -> None:
    agent = _Agent()

    observed = ObservableAgent(
        agent
    )

    assert observed.agent is agent


def test_none_agent_is_rejected() -> None:
    with pytest.raises(
        Exception,
        match="agent is required",
    ):
        ObservableAgent(None)
