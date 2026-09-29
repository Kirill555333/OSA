from typing import Any

import pytest

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.actions.pipeline import (
    ActionDecision,
    ActionPolicyDecision,
    ActionSafetyPipeline,
)
from osa.actions.router import ActionRouter
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.core.agent_voice_action import AgentVoiceActionAdapter
from osa.recovery_action import RecoverableActionExecutor
from osa.voice.contracts import VoiceInput, VoiceTranscript
from osa.voice.session import VoiceSession


class FakeVAD:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int, int]] = []

    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        self.calls.append((audio, sample_rate_hz, channels))
        return True

    def reset(self) -> None:
        self.calls.clear()


class FakeSTT:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.inputs: list[VoiceInput] = []

    def transcribe(self, voice_input: VoiceInput) -> VoiceTranscript:
        self.inputs.append(voice_input)
        return VoiceTranscript(
            text=self.transcript,
            language="en",
        )


class FakeTTS:
    def __init__(self) -> None:
        self.outputs: list[Any] = []
        self.stop_calls = 0

    def synthesize(self, output: Any) -> bytes:
        self.outputs.append(output)
        return output.text.encode("utf-8")

    def stop(self) -> None:
        self.stop_calls += 1


class StaticVoiceResolver:
    def __init__(
        self,
        *,
        kind: ActionKind,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.kind = kind
        self.name = name
        self.arguments = dict(arguments)
        self.commands: list[str] = []

    def resolve(self, command: str) -> ActionRequest:
        self.commands.append(command)
        return ActionRequest(
            kind=self.kind,
            name=self.name,
            arguments=self.arguments,
            metadata={"source": "voice"},
        )


class RecordingHandler:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[ActionRequest] = []

    def execute(self, request: ActionRequest) -> ActionResult:
        self.requests.append(request)
        return ActionResult.succeeded(
            request_id=request.request_id,
            output=self.output,
        )


class FlakyRecordingHandler:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[ActionRequest] = []

    def execute(self, request: ActionRequest) -> ActionResult:
        self.requests.append(request)

        if len(self.requests) == 1:
            raise RuntimeError("transient backend failure")

        return ActionResult.succeeded(
            request_id=request.request_id,
            output=self.output,
        )


class AllowPermissionPolicy:
    def __init__(self) -> None:
        self.requests: list[ActionRequest] = []

    def evaluate(self, request: ActionRequest) -> ActionPolicyDecision:
        self.requests.append(request)
        return ActionPolicyDecision(ActionDecision.ALLOW)


class DenyPermissionPolicy:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.requests: list[ActionRequest] = []

    def evaluate(self, request: ActionRequest) -> ActionPolicyDecision:
        self.requests.append(request)
        return ActionPolicyDecision(
            ActionDecision.DENY,
            self.reason,
        )


class ConfirmPolicy:
    def __init__(self, reason: str = "confirmation required") -> None:
        self.reason = reason
        self.requests: list[ActionRequest] = []

    def evaluate(self, request: ActionRequest) -> ActionPolicyDecision:
        self.requests.append(request)
        return ActionPolicyDecision(
            ActionDecision.CONFIRM,
            self.reason,
        )


class RecordingConfirmation:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.requests: list[ActionRequest] = []
        self.reasons: list[str] = []

    def confirm(self, request: ActionRequest, reason: str) -> bool:
        self.requests.append(request)
        self.reasons.append(reason)
        return self.approved


class RecoveringAgent:
    def __init__(self, recovery: AgentRecoveryIntegration) -> None:
        self.recovery = recovery

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ):
        return self.recovery.execute(
            request,
            run_id=run_id,
            task_id=task_id,
        )


def _build_session(
    *,
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
    pipeline: ActionSafetyPipeline,
) -> tuple[
    VoiceSession,
    StaticVoiceResolver,
    FakeVAD,
    FakeSTT,
    FakeTTS,
]:
    recovery_executor = RecoverableActionExecutor(
        pipeline.dispatch,
        max_attempts=3,
    )
    recovery = AgentRecoveryIntegration(recovery_executor)
    agent = RecoveringAgent(recovery)

    resolver = StaticVoiceResolver(
        kind=kind,
        name=name,
        arguments=arguments,
    )
    voice_agent = AgentVoiceActionAdapter(agent, resolver)

    vad = FakeVAD()
    stt = FakeSTT("perform the action")
    tts = FakeTTS()

    session = VoiceSession(
        voice_agent,
        vad,
        stt,
        tts,
    )

    return session, resolver, vad, stt, tts


@pytest.mark.parametrize(
    ("kind", "name", "arguments", "output"),
    (
        (
            ActionKind.BROWSER,
            "browser_open",
            {"url": "https://example.test"},
            "browser opened",
        ),
        (
            ActionKind.DESKTOP,
            "desktop_applications",
            {},
            "desktop inspected",
        ),
    ),
)
def test_voice_reaches_real_unified_router_for_browser_and_desktop(
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
    output: str,
) -> None:
    handler = RecordingHandler(output)
    router = ActionRouter({kind: handler})
    pipeline = ActionSafetyPipeline(router)

    session, resolver, vad, stt, tts = _build_session(
        kind=kind,
        name=name,
        arguments=arguments,
        pipeline=pipeline,
    )

    result = session.process_audio(
        b"audio",
        sample_rate_hz=16_000,
    )

    assert result is not None
    assert result.response.text == output
    assert result.audio == output.encode("utf-8")

    assert resolver.commands == ["perform the action"]
    assert len(vad.calls) == 1
    assert len(stt.inputs) == 1
    assert len(tts.outputs) == 1
    assert len(handler.requests) == 1

    request = handler.requests[0]
    assert request.kind is kind
    assert request.name == name
    assert request.arguments == arguments
    assert request.metadata["source"] == "voice"


@pytest.mark.parametrize(
    ("kind", "name", "arguments"),
    (
        (
            ActionKind.BROWSER,
            "browser_open",
            {"url": "https://example.test"},
        ),
        (
            ActionKind.DESKTOP,
            "desktop_applications",
            {},
        ),
    ),
)
def test_voice_backend_failure_does_not_retry_without_explicit_policy(
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
) -> None:
    handler = FlakyRecordingHandler("should not reach second attempt")
    router = ActionRouter({kind: handler})
    permission_policy = AllowPermissionPolicy()

    pipeline = ActionSafetyPipeline(
        router,
        permission_policy=permission_policy,
    )

    session, resolver, _, _, _ = _build_session(
        kind=kind,
        name=name,
        arguments=arguments,
        pipeline=pipeline,
    )

    result = session.process_audio(
        b"audio",
        sample_rate_hz=16_000,
    )

    assert result is not None
    assert result.response.text == "Action handler failed: transient backend failure"

    assert resolver.commands == ["perform the action"]

    assert len(handler.requests) == 1
    assert len(permission_policy.requests) == 1

    request = handler.requests[0]
    assert request.kind is kind
    assert request.name == name
    assert request.arguments == arguments
    assert request.metadata["source"] == "voice"

    assert permission_policy.requests[0].request_id == request.request_id


@pytest.mark.parametrize(
    ("kind", "name", "arguments", "reason"),
    (
        (
            ActionKind.BROWSER,
            "browser_click",
            {"selector": "#submit"},
            "browser confirmation required",
        ),
        (
            ActionKind.DESKTOP,
            "desktop_close_application",
            {"application": "TestApp"},
            "desktop confirmation required",
        ),
    ),
)
def test_voice_confirmation_denial_stops_before_browser_or_desktop_backend(
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
    reason: str,
) -> None:
    handler = RecordingHandler("must not execute")
    router = ActionRouter({kind: handler})

    confirmation_policy = ConfirmPolicy(reason)
    confirmation_handler = RecordingConfirmation(False)

    pipeline = ActionSafetyPipeline(
        router,
        safety_policy=confirmation_policy,
        confirmation_handler=confirmation_handler,
    )

    session, resolver, _, _, _ = _build_session(
        kind=kind,
        name=name,
        arguments=arguments,
        pipeline=pipeline,
    )

    result = session.process_audio(
        b"audio",
        sample_rate_hz=16_000,
    )

    assert result is not None
    assert result.response.text == "Action confirmation was not granted."
    assert result.audio == b"Action confirmation was not granted."

    assert resolver.commands == ["perform the action"]

    assert len(confirmation_policy.requests) == 1
    assert len(confirmation_handler.requests) == 1
    assert len(confirmation_handler.reasons) == 1
    assert confirmation_handler.reasons[0] == reason
    assert len(handler.requests) == 0

    request = confirmation_policy.requests[0]
    assert request.kind is kind
    assert request.name == name
    assert request.arguments == arguments
    assert request.metadata["source"] == "voice"

    assert confirmation_handler.requests[0].request_id == request.request_id


def test_voice_browser_permission_denial_stops_before_router() -> None:
    handler = RecordingHandler("must not execute")
    router = ActionRouter({ActionKind.BROWSER: handler})

    permission_policy = DenyPermissionPolicy(
        "browser permission denied",
    )

    pipeline = ActionSafetyPipeline(
        router,
        permission_policy=permission_policy,
    )

    session, _, _, _, _ = _build_session(
        kind=ActionKind.BROWSER,
        name="browser_click",
        arguments={"selector": "#danger"},
        pipeline=pipeline,
    )

    result = session.process_audio(
        b"audio",
        sample_rate_hz=16_000,
    )

    assert result is not None
    assert result.response.text == "browser permission denied"
    assert result.audio == b"browser permission denied"

    assert len(permission_policy.requests) == 1
    denied_request = permission_policy.requests[0]
    assert denied_request.kind is ActionKind.BROWSER
    assert denied_request.name == "browser_click"

    assert handler.requests == []
