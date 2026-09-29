from __future__ import annotations

from osa.actions import (
    ActionKind,
    ActionRequest,
)
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
    AgentVoiceActionIntegrationError,
    TaskActionResolverVoiceAdapter,
    create_agent_voice_action_adapter,
)
from osa.recovery_contracts import (
    RecoveryFailureKind,
    RecoveryResult,
)
from osa.tasks.action import TaskAction


class FakeTaskActionResolver:
    def __init__(
        self,
        action: TaskAction | None = None,
        *,
        exception: Exception | None = None,
    ) -> None:
        self.action = (
            action
            if action is not None
            else TaskAction(
                tool_name="demo_tool",
                arguments={
                    "value": 42,
                },
            )
        )
        self.exception = exception
        self.calls = []

    def resolve(
        self,
        task,
        outputs,
    ) -> TaskAction:
        self.calls.append(
            (
                task,
                outputs,
            )
        )

        if self.exception is not None:
            raise self.exception

        return self.action


class FakeRecoveringAgent:
    def __init__(
        self,
        result: RecoveryResult | None = None,
        *,
        exception: Exception | None = None,
    ) -> None:
        self.result = (
            result
            if result is not None
            else RecoveryResult.succeeded(
                result="done",
            )
        )
        self.exception = exception
        self.chat_calls = []
        self.recovery_calls = []

    def chat(
        self,
        user_input: str,
    ) -> str:
        self.chat_calls.append(user_input)

        raise AssertionError(
            "Voice action adapter must not call Agent.chat()."
        )

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.recovery_calls.append(
            (
                request,
                run_id,
                task_id,
            )
        )

        if self.exception is not None:
            raise self.exception

        return self.result


class FakeVoiceResolver:
    def __init__(
        self,
        request: ActionRequest | None = None,
    ) -> None:
        self.request = (
            request
            if request is not None
            else ActionRequest(
                kind=ActionKind.TOOL,
                name="demo_tool",
                arguments={
                    "value": 42,
                },
                metadata={
                    "source": "voice",
                    "voice_task_id": "task-voice-1",
                },
            )
        )
        self.commands: list[str] = []

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        self.commands.append(command)

        return self.request


def test_task_action_adapter_builds_tool_action_request():
    resolver = FakeTaskActionResolver()

    adapter = TaskActionResolverVoiceAdapter(
        resolver,
    )

    request = adapter.resolve(
        "open the demo",
    )

    assert request.kind is ActionKind.TOOL
    assert request.name == "demo_tool"
    assert dict(request.arguments) == {
        "value": 42,
    }

    assert request.metadata["source"] == "voice"
    assert request.metadata["voice_task_id"]

    assert len(resolver.calls) == 1

    task, outputs = resolver.calls[0]

    assert task.description == "open the demo"
    assert task.task_id == request.metadata[
        "voice_task_id"
    ]
    assert outputs == {}


def test_task_action_adapter_rejects_empty_command():
    resolver = FakeTaskActionResolver()

    adapter = TaskActionResolverVoiceAdapter(
        resolver,
    )

    try:
        adapter.resolve("   ")
    except ValueError as exc:
        assert "command" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )

    assert resolver.calls == []


def test_task_action_adapter_wraps_resolver_failure():
    adapter = TaskActionResolverVoiceAdapter(
        FakeTaskActionResolver(
            exception=RuntimeError(
                "resolver failed",
            )
        )
    )

    try:
        adapter.resolve(
            "do something"
        )
    except AgentVoiceActionIntegrationError as exc:
        assert "resolver failed" in str(exc)
    else:
        raise AssertionError(
            "Expected integration error."
        )


def test_agent_voice_action_adapter_uses_unified_recovery():
    agent = FakeRecoveringAgent(
        RecoveryResult.succeeded(
            result="completed",
        )
    )

    resolver = FakeVoiceResolver()

    adapter = AgentVoiceActionAdapter(
        agent,
        resolver,
    )

    result = adapter.chat(
        "open the demo",
    )

    assert result.content == "completed"
    assert resolver.commands == [
        "open the demo",
    ]
    assert agent.chat_calls == []
    assert len(agent.recovery_calls) == 1

    request, run_id, task_id = (
        agent.recovery_calls[0]
    )

    assert request is resolver.request
    assert isinstance(
        run_id,
        str,
    )
    assert run_id.startswith(
        "voice-",
    )
    assert task_id == "task-voice-1"


def test_agent_voice_action_adapter_fails_without_fallback():
    agent = FakeRecoveringAgent(
        RecoveryResult.failed(
            error="Action confirmation was not granted.",
            failure_kind=RecoveryFailureKind.CONFIRMATION_DENIED,
        )
    )

    adapter = AgentVoiceActionAdapter(
        agent,
        FakeVoiceResolver(),
    )

    result = adapter.chat(
        "do the sensitive action",
    )

    assert result.content == (
        "Action confirmation was not granted."
    )
    assert agent.chat_calls == []


def test_agent_voice_action_adapter_rejects_invalid_request():
    class InvalidResolver:
        def resolve(
            self,
            command: str,
        ):
            return object()

    agent = FakeRecoveringAgent()

    adapter = AgentVoiceActionAdapter(
        agent,
        InvalidResolver(),
    )

    try:
        adapter.chat(
            "invalid action"
        )
    except AgentVoiceActionIntegrationError as exc:
        assert "invalid ActionRequest" in str(exc)
    else:
        raise AssertionError(
            "Expected integration error."
        )

    assert agent.recovery_calls == []


def test_agent_voice_action_adapter_formats_non_string_result():
    agent = FakeRecoveringAgent(
        RecoveryResult.succeeded(
            result={
                "ok": True,
                "items": 2,
            }
        )
    )

    adapter = AgentVoiceActionAdapter(
        agent,
        FakeVoiceResolver(),
    )

    result = adapter.chat(
        "report status",
    )

    assert '"ok": true' in result.content
    assert '"items": 2' in result.content


def test_factory_wires_existing_task_action_resolver():
    task_resolver = FakeTaskActionResolver()

    agent = FakeRecoveringAgent()

    adapter = create_agent_voice_action_adapter(
        agent,
        task_resolver,
    )

    result = adapter.chat(
        "run demo",
    )

    assert result.content == "done"
    assert len(
        task_resolver.calls
    ) == 1
    assert len(
        agent.recovery_calls
    ) == 1


def test_factory_does_not_use_agent_chat():
    task_resolver = FakeTaskActionResolver()

    agent = FakeRecoveringAgent(
        RecoveryResult.succeeded(
            result="safe path",
        )
    )

    adapter = create_agent_voice_action_adapter(
        agent,
        task_resolver,
    )

    result = adapter.chat(
        "execute demo",
    )

    assert result.content == "safe path"
    assert agent.chat_calls == []
