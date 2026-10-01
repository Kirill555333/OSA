"""0.8.10.7 End-to-end autonomous execution regression tests."""

from __future__ import annotations

from dataclasses import dataclass

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.actions.pipeline import (
    ActionDecision,
    ActionPolicyDecision,
    ActionSafetyPipeline,
    CallbackActionPolicy,
)
from osa.observability import InMemoryObservabilityLogger
from osa.observability.autonomous_action import (
    ObservableAutonomousActionBridge,
)
from osa.tasks.action_bridge import AutonomousActionBridge
from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_recovery import (
    AutonomousRecoveryDecision,
    AutonomousRecoveryExecutor,
    CallbackAutonomousRecoveryPolicy,
)
from osa.tasks.autonomous_executor import UnifiedAutonomousExecutor


@dataclass
class FixedResolver:
    request: ActionRequest

    def resolve(
        self,
        run_id: str,
        task_id: str,
    ) -> ActionRequest:
        return self.request


class SequenceRouter:
    """Return deterministic action results for recovery regression."""

    def __init__(
        self,
        results: list[ActionResult],
    ) -> None:
        self.results = list(results)
        self.calls: list[ActionRequest] = []

    def dispatch(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        self.calls.append(request)

        if not self.results:
            raise RuntimeError(
                "no router result configured"
            )

        return self.results.pop(0)


class RecordingRouter:
    """Router used to verify denied actions never execute."""

    def __init__(self) -> None:
        self.calls: list[ActionRequest] = []

    def dispatch(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        self.calls.append(request)

        return ActionResult.succeeded(
            request.request_id,
            output="should-not-run",
        )


def _request(
    request_id: str,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="autonomous_tool",
        arguments={
            "token": "TOP_SECRET_AUTONOMOUS_TOKEN",
        },
        request_id=request_id,
    )


def _unified(
    request: ActionRequest,
    bridge,
) -> UnifiedAutonomousExecutor:
    return UnifiedAutonomousExecutor(
        FixedResolver(request),
        bridge,
    )


def test_autonomous_allowed_path_is_fully_observable() -> None:
    logger = InMemoryObservabilityLogger()

    request = _request(
        "autonomous-regression-success",
    )

    router = SequenceRouter(
        [
            ActionResult.succeeded(
                request.request_id,
                output="completed",
            )
        ]
    )

    pipeline = ActionSafetyPipeline(
        router,
    )

    bridge = ObservableAutonomousActionBridge(
        AutonomousActionBridge(
            pipeline
        ),
        logger=logger,
    )

    executor = _unified(
        request,
        bridge,
    )

    result = executor.execute(
        "run-regression-success",
        "task-regression-success",
    )

    assert result.status == "completed"
    assert result.output == "completed"
    assert len(router.calls) == 1
    assert router.calls[0] is request

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.action.started",
        "autonomous.action.completed",
    ]

    for event in events:
        assert event.request_id == request.request_id
        assert event.run_id == "run-regression-success"
        assert event.task_id == "task-regression-success"
        assert event.source == "autonomous"

    serialized = str(
        events[-1].to_dict()
    )

    assert "TOP_SECRET_AUTONOMOUS_TOKEN" not in serialized
    assert "token" not in serialized


def test_autonomous_denial_stops_before_router() -> None:
    logger = InMemoryObservabilityLogger()

    request = _request(
        "autonomous-regression-denied",
    )

    router = RecordingRouter()

    pipeline = ActionSafetyPipeline(
        router,
        safety_policy=CallbackActionPolicy(
            lambda current: ActionPolicyDecision(
                ActionDecision.DENY,
                "autonomous safety denied",
            )
        ),
    )

    bridge = ObservableAutonomousActionBridge(
        AutonomousActionBridge(
            pipeline
        ),
        logger=logger,
    )

    executor = _unified(
        request,
        bridge,
    )

    result = executor.execute(
        "run-regression-denied",
        "task-regression-denied",
    )

    assert result.status == "failed"
    assert result.error == (
        "autonomous safety denied"
    )
    assert router.calls == []

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.action.started",
        "autonomous.action.failed",
    ]

    assert events[-1].request_id == request.request_id
    assert events[-1].run_id == "run-regression-denied"
    assert events[-1].task_id == "task-regression-denied"


def test_autonomous_recovery_preserves_full_identity() -> None:
    logger = InMemoryObservabilityLogger()

    request = _request(
        "autonomous-regression-recovery",
    )

    router = SequenceRouter(
        [
            ActionResult.failed(
                request.request_id,
                error="temporary backend failure",
            ),
            ActionResult.succeeded(
                request.request_id,
                output="recovered",
            ),
        ]
    )

    pipeline = ActionSafetyPipeline(
        router,
    )

    bridge = ObservableAutonomousActionBridge(
        AutonomousActionBridge(
            pipeline
        ),
        logger=logger,
    )

    executor = _unified(
        request,
        bridge,
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda run_id, task_id, result, attempt:
                AutonomousRecoveryDecision(
                    retry=True,
                    reason="temporary",
                )
        ),
        max_attempts=2,
    )

    result = recovery.execute(
        "run-regression-recovery",
        "task-regression-recovery",
    )

    assert result.status == "completed"
    assert result.output == "recovered"
    assert len(router.calls) == 2
    assert router.calls[0] is request
    assert router.calls[1] is request
    assert [
        current.request_id
        for current in router.calls
    ] == [
        request.request_id,
        request.request_id,
    ]

    events = logger.events()

    assert [
        event.event
        for event in events
    ] == [
        "autonomous.action.started",
        "autonomous.action.failed",
        "autonomous.action.started",
        "autonomous.action.completed",
    ]

    for event in events:
        assert event.request_id == request.request_id
        assert event.run_id == "run-regression-recovery"
        assert event.task_id == "task-regression-recovery"
        assert event.source == "autonomous"

    for event in events:
        serialized = str(
            event.to_dict()
        )
        assert "TOP_SECRET_AUTONOMOUS_TOKEN" not in serialized
        assert "token" not in serialized


def test_autonomous_invalid_bridge_result_fails_closed() -> None:
    logger = InMemoryObservabilityLogger()

    class InvalidBridge:
        def execute(
            self,
            run_id: str,
            task_id: str,
            request: ActionRequest,
        ):
            return {
                "status": "completed",
            }

    request = _request(
        "autonomous-regression-invalid",
    )

    executor = _unified(
        request,
        ObservableAutonomousActionBridge(
            InvalidBridge(),
            logger=logger,
        ),
    )

    result = executor.execute(
        "run-regression-invalid",
        "task-regression-invalid",
    )

    assert result.status == "failed"
    assert result.error == (
        "wrapped autonomous bridge returned "
        "invalid task result"
    )

    assert logger.events()[-1].event == (
        "autonomous.action.failed"
    )


def test_autonomous_recovery_does_not_retry_success() -> None:
    logger = InMemoryObservabilityLogger()

    request = _request(
        "autonomous-regression-no-retry",
    )

    router = SequenceRouter(
        [
            ActionResult.succeeded(
                request.request_id,
                output="done",
            )
        ]
    )

    bridge = ObservableAutonomousActionBridge(
        AutonomousActionBridge(
            ActionSafetyPipeline(
                router,
            )
        ),
        logger=logger,
    )

    executor = _unified(
        request,
        bridge,
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda run_id, task_id, result, attempt:
                AutonomousRecoveryDecision(
                    retry=True,
                    reason="must-not-run",
                )
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-regression-no-retry",
        "task-regression-no-retry",
    )

    assert result.status == "completed"
    assert result.output == "done"
    assert len(router.calls) == 1

    assert [
        event.event
        for event in logger.events()
    ] == [
        "autonomous.action.started",
        "autonomous.action.completed",
    ]
