from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from osa.actions import (
    ActionDecision,
    ActionKind,
    ActionPolicyDecision,
    ActionRequest,
    ActionSafetyPipeline,
    ActionRouter,
    BrowserActionAdapter,
    DesktopActionAdapter,
)
from osa.actions.contracts import ActionResult
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.core.agent_voice_action import AgentVoiceActionAdapter
from osa.desktop.automation import DesktopAutomationInterface, DesktopElementLocator, DesktopPoint
from osa.recovery_action import RecoverableActionExecutor
from osa.voice.contracts import VoiceInput, VoiceTranscript
from osa.voice.session import VoiceSession


@dataclass(frozen=True)
class FakeBrowserToolResult:
    success: bool
    output: str = ""
    error: str | None = None


class FakeBrowserTool:
    def __init__(
        self,
        name: str,
        result: FakeBrowserToolResult | None = None,
    ) -> None:
        self.name = name
        self.result = result or FakeBrowserToolResult(
            success=True,
            output=f"browser-executed:{name}",
        )
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> FakeBrowserToolResult:
        self.calls.append(dict(arguments))
        return self.result


class FakeDesktopBackend(DesktopAutomationInterface):
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def applications(self):
        self.calls.append(("applications", None))
        return [{"name": "Calculator"}]

    def launch_application(self, command: str):
        self.calls.append(("launch_application", command))
        return f"launched:{command}"

    def close_application(self, name: str):
        self.calls.append(("close_application", name))
        return f"closed:{name}"

    def windows(self):
        self.calls.append(("windows", None))
        return [{"title": "Main"}]

    def activate_window(self, title: str):
        self.calls.append(("activate_window", title))
        return f"activated:{title}"

    def close_window(self, title: str):
        self.calls.append(("close_window", title))
        return f"closed-window:{title}"

    def find_element(self, locator: DesktopElementLocator):
        self.calls.append(("find_element", locator))
        return {
            "kind": locator.kind,
            "value": locator.value,
        }

    def click_element(self, element):
        self.calls.append(("click_element", element))
        return "clicked-element"

    def click_point(self, point: DesktopPoint):
        self.calls.append(("click_point", point))
        return "clicked-point"

    def type_text(self, text: str):
        self.calls.append(("type_text", text))
        return f"typed:{text}"

    def press(self, key: str):
        self.calls.append(("press", key))
        return f"pressed:{key}"

    def hotkey(self, *keys: str):
        self.calls.append(("hotkey", keys))
        return f"hotkey:{','.join(keys)}"

    def screenshot(self):
        self.calls.append(("screenshot", None))
        return "desktop-screenshot"

    def health_check(self):
        self.calls.append(("health_check", None))
        return True


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


class RecordingConfirmation:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.requests: list[ActionRequest] = []
        self.reasons: list[str] = []

    def confirm(
        self,
        request: ActionRequest,
        reason: str,
    ) -> bool:
        self.requests.append(request)
        self.reasons.append(reason)
        return self.approved


class ConfirmPolicy:
    def __init__(self, reason: str = "confirmation required") -> None:
        self.reason = reason
        self.requests: list[ActionRequest] = []

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        self.requests.append(request)
        return ActionPolicyDecision(
            ActionDecision.CONFIRM,
            self.reason,
        )


class DenyPermissionPolicy:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.requests: list[ActionRequest] = []

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        self.requests.append(request)
        return ActionPolicyDecision(
            ActionDecision.DENY,
            self.reason,
        )


class AllowPermissionPolicy:
    def __init__(self) -> None:
        self.requests: list[ActionRequest] = []

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        self.requests.append(request)
        return ActionPolicyDecision(ActionDecision.ALLOW)


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
    voice_agent = AgentVoiceActionAdapter(
        agent,
        resolver,
    )

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


def _build_real_adapter(
    kind: ActionKind,
) -> tuple[Any, Any]:
    if kind is ActionKind.BROWSER:
        tool = FakeBrowserTool(
            "browser_click",
            FakeBrowserToolResult(
                success=True,
                output="real browser adapter completed",
            ),
        )
        return BrowserActionAdapter([tool]), tool

    backend = FakeDesktopBackend()
    return DesktopActionAdapter(backend), backend


@pytest.mark.parametrize(
    ("kind", "name", "arguments", "expected_output"),
    (
        (
            ActionKind.BROWSER,
            "browser_click",
            {"selector": "#submit"},
            "real browser adapter completed",
        ),
        (
            ActionKind.DESKTOP,
            "desktop_close_application",
            {"name": "TestApp"},
            "Application closed.",
        ),
    ),
)
def test_voice_reaches_real_browser_or_desktop_adapter(
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
    expected_output: str,
) -> None:
    adapter, backend = _build_real_adapter(kind)
    router = ActionRouter({kind: adapter})
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
    assert result.response.text == expected_output
    assert result.audio == expected_output.encode("utf-8")

    assert resolver.commands == ["perform the action"]
    assert len(vad.calls) == 1
    assert len(stt.inputs) == 1
    assert len(tts.outputs) == 1

    if kind is ActionKind.BROWSER:
        assert backend.calls == [arguments]
    else:
        assert backend.calls == [
            ("close_application", "TestApp"),
        ]


def test_voice_backend_failure_does_not_retry_without_explicit_policy() -> None:
    handler = FlakyRecordingHandler("should not reach second attempt")
    router = ActionRouter({ActionKind.BROWSER: handler})
    permission_policy = AllowPermissionPolicy()

    pipeline = ActionSafetyPipeline(
        router,
        permission_policy=permission_policy,
    )

    session, resolver, _, _, _ = _build_session(
        kind=ActionKind.BROWSER,
        name="browser_open",
        arguments={
            "url": "https://example.test",
        },
        pipeline=pipeline,
    )

    result = session.process_audio(
        b"audio",
        sample_rate_hz=16_000,
    )

    assert result is not None
    assert result.response.text == (
        "Action handler failed: transient backend failure"
    )

    assert resolver.commands == ["perform the action"]
    assert len(handler.requests) == 1
    assert len(permission_policy.requests) == 1

    request = handler.requests[0]
    assert request.kind is ActionKind.BROWSER
    assert request.name == "browser_open"
    assert request.arguments == {
        "url": "https://example.test",
    }
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
            {"name": "TestApp"},
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
    adapter, backend = _build_real_adapter(kind)
    router = ActionRouter({kind: adapter})

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
    assert confirmation_handler.reasons == [reason]

    if kind is ActionKind.BROWSER:
        assert backend.calls == []
    else:
        assert backend.calls == []


@pytest.mark.parametrize(
    ("kind", "name", "arguments", "reason", "expected_output"),
    (
        (
            ActionKind.BROWSER,
            "browser_click",
            {"selector": "#submit"},
            "browser confirmation required",
            "real browser adapter completed",
        ),
        (
            ActionKind.DESKTOP,
            "desktop_close_application",
            {"name": "TestApp"},
            "desktop confirmation required",
            "Application closed.",
        ),
    ),
)
def test_voice_confirmation_approval_reaches_real_adapter(
    kind: ActionKind,
    name: str,
    arguments: dict[str, Any],
    reason: str,
    expected_output: str,
) -> None:
    adapter, backend = _build_real_adapter(kind)
    router = ActionRouter({kind: adapter})

    confirmation_policy = ConfirmPolicy(reason)
    confirmation_handler = RecordingConfirmation(True)

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
    assert result.response.text == expected_output
    assert result.audio == expected_output.encode("utf-8")

    assert resolver.commands == ["perform the action"]
    assert len(confirmation_policy.requests) == 1
    assert len(confirmation_handler.requests) == 1
    assert confirmation_handler.reasons == [reason]

    if kind is ActionKind.BROWSER:
        assert backend.calls == [arguments]
    else:
        assert backend.calls == [
            ("close_application", "TestApp"),
        ]


def test_voice_browser_permission_denial_stops_before_router() -> None:
    adapter = BrowserActionAdapter(
        [
            FakeBrowserTool(
                "browser_click",
                FakeBrowserToolResult(
                    success=True,
                    output="must not execute",
                ),
            )
        ]
    )
    router = ActionRouter({ActionKind.BROWSER: adapter})

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
