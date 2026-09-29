from __future__ import annotations

from osa.actions import (
    ActionDecision,
    ActionKind,
    ActionPolicyDecision,
    ActionResult,
    ActionRouter,
    ActionSafetyPipeline,
    CallbackActionConfirmationHandler,
)
from osa.core.agent import Agent
from osa.core.agent_recovery import (
    AgentRecoveryIntegration,
)
from osa.recovery_action import (
    DefaultActionResultClassifier,
    RecoverableActionExecutor,
)
from osa.recovery_contracts import (
    RecoveryFailureKind,
    RecoveryRequest,
)
from osa.tasks.action import TaskAction
from osa.voice.contracts import (
    VoiceInput,
    VoiceOutput,
    VoiceTranscript,
)
from osa.voice.unified_action import (
    create_unified_voice_session,
)


class FakeModel:
    pass


class FakeTaskActionResolver:
    def resolve(
        self,
        task,
        outputs,
    ) -> TaskAction:
        return TaskAction(
            tool_name="demo_tool",
            arguments={
                "value": 42,
            },
        )


class FakeVAD:
    def detect(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> bool:
        return True

    def reset(self) -> None:
        return None


class FakeSTT:
    def transcribe(
        self,
        voice_input: VoiceInput,
    ) -> VoiceTranscript:
        return VoiceTranscript(
            text="open the demo",
            language="en",
        )


class FakeTTS:
    def synthesize(
        self,
        output: VoiceOutput,
    ) -> bytes:
        return b"speech"

    def stop(self) -> None:
        return None


class RecordingHandler:
    def __init__(self) -> None:
        self.calls = []

    def execute(
        self,
        request,
    ) -> ActionResult:
        self.calls.append(request)

        return ActionResult.succeeded(
            request.request_id,
            "executed",
        )


def _build_agent(
    pipeline: ActionSafetyPipeline,
) -> Agent:
    recoverable = RecoverableActionExecutor(
        pipeline.dispatch,
        max_attempts=3,
    )

    integration = AgentRecoveryIntegration(
        recoverable,
    )

    return Agent(
        FakeModel(),
        recovery_integration=integration,
    )


def _build_session(
    agent: Agent,
) :
    return create_unified_voice_session(
        agent,
        FakeTaskActionResolver(),
        FakeVAD(),
        FakeSTT(),
        FakeTTS(),
    )


def test_voice_allow_path_reaches_real_router():
    handler = RecordingHandler()

    pipeline = ActionSafetyPipeline(
        ActionRouter(
            {
                ActionKind.TOOL: handler,
            }
        )
    )

    agent = _build_agent(
        pipeline,
    )

    session = _build_session(
        agent,
    )

    result = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
        channels=1,
    )

    assert result is not None
    assert result.response.text == "executed"
    assert result.audio == b"speech"

    assert len(handler.calls) == 1

    request = handler.calls[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "demo_tool"
    assert dict(request.arguments) == {
        "value": 42,
    }
    assert request.metadata["source"] == "voice"


def test_voice_permission_denial_never_reaches_router():
    handler = RecordingHandler()

    class PermissionDenyPolicy:
        def evaluate(
            self,
            request,
        ) -> ActionPolicyDecision:
            return ActionPolicyDecision(
                ActionDecision.DENY,
                "permission denied",
            )

    pipeline = ActionSafetyPipeline(
        ActionRouter(
            {
                ActionKind.TOOL: handler,
            }
        ),
        permission_policy=PermissionDenyPolicy(),
    )

    agent = _build_agent(
        pipeline,
    )

    session = _build_session(
        agent,
    )

    result = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
        channels=1,
    )

    assert result is not None
    assert result.response.text == "permission denied"
    assert handler.calls == []


def test_voice_confirmation_denial_never_reaches_router():
    handler = RecordingHandler()
    confirmation_calls = []

    class ConfirmPolicy:
        def evaluate(
            self,
            request,
        ) -> ActionPolicyDecision:
            return ActionPolicyDecision(
                ActionDecision.CONFIRM,
                "confirmation required",
            )

    confirmation = CallbackActionConfirmationHandler(
        lambda request, reason: (
            confirmation_calls.append(
                (
                    request,
                    reason,
                )
            )
            or False
        )
    )

    pipeline = ActionSafetyPipeline(
        ActionRouter(
            {
                ActionKind.TOOL: handler,
            }
        ),
        safety_policy=ConfirmPolicy(),
        confirmation_handler=confirmation,
    )

    agent = _build_agent(
        pipeline,
    )

    session = _build_session(
        agent,
    )

    result = session.process_audio(
        b"\x00\x00",
        sample_rate_hz=16000,
        channels=1,
    )

    assert result is not None
    assert result.response.text == (
        "Action confirmation was not granted."
    )

    assert handler.calls == []
    assert len(confirmation_calls) == 1

    request, reason = confirmation_calls[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "demo_tool"
    assert reason == "confirmation required"


def test_classifier_preserves_permission_denial():
    classifier = DefaultActionResultClassifier()

    request = RecoveryRequest(
        request_id="req-permission",
        attempt=1,
        max_attempts=3,
    )

    result = ActionResult.failed(
        "req-permission",
        "permission denied",
        metadata={
            "pipeline_stage": "permission",
            "pipeline_error": "denied",
        },
    )

    attempt = classifier(
        request,
        result,
    )

    assert attempt.failure_kind is (
        RecoveryFailureKind.PERMISSION_DENIED
    )


def test_classifier_preserves_confirmation_denial():
    classifier = DefaultActionResultClassifier()

    request = RecoveryRequest(
        request_id="req-confirmation",
        attempt=1,
        max_attempts=3,
    )

    result = ActionResult.failed(
        "req-confirmation",
        "Action confirmation was not granted.",
        metadata={
            "pipeline_stage": "confirmation",
            "pipeline_error": "not_confirmed",
        },
    )

    attempt = classifier(
        request,
        result,
    )

    assert attempt.failure_kind is (
        RecoveryFailureKind.CONFIRMATION_DENIED
    )


def test_classifier_keeps_generic_backend_failure():
    classifier = DefaultActionResultClassifier()

    request = RecoveryRequest(
        request_id="req-backend",
        attempt=1,
        max_attempts=1,
    )

    result = ActionResult.failed(
        "req-backend",
        "backend failed",
    )

    attempt = classifier(
        request,
        result,
    )

    assert attempt.failure_kind is (
        RecoveryFailureKind.BACKEND_ERROR
    )
