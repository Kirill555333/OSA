from __future__ import annotations

from osa.actions import ActionKind, ActionRequest
from osa.core.agent_voice_action import (
    AgentVoiceActionIntegrationError,
)
from osa.recovery_contracts import RecoveryResult
from osa.tasks.action import TaskAction
from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceTranscript,
)
from osa.voice.session import VoiceSession
from osa.voice.unified_action import (
    create_unified_voice_session,
)


class FakeTaskActionResolver:
    def resolve(
        self,
        task,
        outputs,
    ) -> TaskAction:
        assert task.description == "open the demo"
        assert outputs == {}

        return TaskAction(
            tool_name="demo_tool",
            arguments={
                "value": 42,
            },
        )


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                ActionRequest,
                str | None,
                str | None,
            ]
        ] = []

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls.append(
            (
                request,
                run_id,
                task_id,
            )
        )

        return RecoveryResult.succeeded(
            result="demo completed",
        )


class FakeVAD:
    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        assert isinstance(
            audio,
            bytes,
        )
        assert sample_rate_hz == 16000
        assert channels == 1

        return True

    def reset(self) -> None:
        return None


class FakeSTT:
    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        assert isinstance(
            voice_input,
            VoiceInput,
        )
        assert voice_input.sample_rate_hz == 16000
        assert voice_input.channels == 1
        assert voice_input.audio == b"\x00\x00"

        return VoiceTranscript(
            text="open the demo",
        )


class FakeTTS:
    def synthesize(
        self,
        output: VoiceOutput,
    ) -> bytes:
        assert output.text == "demo completed"

        return b"speech"

    def stop(self) -> None:
        return None


def _build_session(
    agent: FakeAgent,
) -> VoiceSession:
    return create_unified_voice_session(
        agent,
        FakeTaskActionResolver(),
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )


def _process_audio(
    session: VoiceSession,
):
    return session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
        channels=1,
    )


def test_factory_returns_voice_session():
    session = _build_session(
        FakeAgent(),
    )

    assert isinstance(
        session,
        VoiceSession,
    )


def test_factory_wires_voice_into_unified_recovery():
    agent = FakeAgent()

    session = _build_session(
        agent,
    )

    result = _process_audio(
        session,
    )

    assert result is not None
    assert result.response.text == "demo completed"
    assert result.audio == b"speech"

    assert len(agent.calls) == 1

    request, run_id, task_id = agent.calls[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "demo_tool"
    assert dict(request.arguments) == {
        "value": 42,
    }
    assert request.metadata["source"] == "voice"

    assert isinstance(
        run_id,
        str,
    )
    assert run_id.startswith(
        "voice-",
    )

    assert isinstance(
        task_id,
        str,
    )
    assert task_id


def test_factory_does_not_create_separate_voice_execution_path():
    agent = FakeAgent()

    session = _build_session(
        agent,
    )

    result = _process_audio(
        session,
    )

    assert result is not None
    assert result.response.text == "demo completed"
    assert len(agent.calls) == 1


def test_factory_rejects_missing_agent():
    try:
        create_unified_voice_session(
            None,
            FakeTaskActionResolver(),
            FakeVAD(),
            FakeSTT(),
            FakeTTS(),
        )
    except ValueError as exc:
        assert "agent" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError.",
        )


def test_factory_rejects_missing_resolver():
    agent = FakeAgent()

    try:
        create_unified_voice_session(
            agent,
            None,
            FakeVAD(),
            FakeSTT(),
            FakeTTS(),
        )
    except ValueError as exc:
        assert "task_action_resolver" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError.",
        )


def test_factory_wraps_invalid_voice_composition():
    class InvalidAgent:
        pass

    try:
        create_unified_voice_session(
            InvalidAgent(),
            FakeTaskActionResolver(),
            FakeVAD(),
            FakeSTT(),
            FakeTTS(),
        )
    except Exception as exc:
        assert isinstance(
            exc,
            AgentVoiceActionIntegrationError,
        ) or "unified voice action" in str(
            exc
        ).lower()
    else:
        raise AssertionError(
            "Expected composition failure.",
        )
