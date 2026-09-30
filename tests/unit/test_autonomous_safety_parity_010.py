"""0.8.10.5 Autonomous execution safety parity tests."""

from __future__ import annotations

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.actions.pipeline import (
    ActionDecision,
    ActionPolicyDecision,
    ActionSafetyPipeline,
    CallbackActionConfirmationHandler,
    CallbackActionPolicy,
)
from osa.actions.router import ActionRouter
from osa.tasks.action_bridge import AutonomousActionBridge
from osa.tasks.autonomous import AutonomousTaskResult


class ExplodingRouter:
    """Router that proves denied actions never reach execution."""

    def __init__(self) -> None:
        self.calls = 0

    def dispatch(self, request):
        self.calls += 1
        raise AssertionError(
            "denied autonomous action reached the router"
        )


def _request(
    *,
    name: str = "protected",
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name=name,
        arguments={},
        request_id=f"safety-010-{name}",
    )


def test_autonomous_permission_denial_matches_unified_pipeline() -> None:
    router = ExplodingRouter()

    pipeline = ActionSafetyPipeline(
        router,
        permission_policy=CallbackActionPolicy(
            lambda request: ActionPolicyDecision(
                ActionDecision.DENY,
                "permission denied",
            )
        ),
    )

    bridge = AutonomousActionBridge(
        pipeline
    )

    result = bridge.execute(
        "run-010",
        "task-permission",
        _request(),
    )

    assert isinstance(
        result,
        AutonomousTaskResult,
    )
    assert result.task_id == "task-permission"
    assert result.status == "failed"
    assert result.error == "permission denied"
    assert router.calls == 0


def test_autonomous_confirmation_denial_matches_unified_pipeline() -> None:
    router = ExplodingRouter()

    confirmations: list[str] = []

    pipeline = ActionSafetyPipeline(
        router,
        permission_policy=CallbackActionPolicy(
            lambda request: ActionPolicyDecision(
                ActionDecision.CONFIRM,
                "confirmation required",
            )
        ),
        confirmation_handler=CallbackActionConfirmationHandler(
            lambda request, reason: (
                confirmations.append(reason)
                or False
            )
        ),
    )

    bridge = AutonomousActionBridge(
        pipeline
    )

    result = bridge.execute(
        "run-010",
        "task-confirmation",
        _request(
            name="sensitive",
        ),
    )

    assert result.task_id == "task-confirmation"
    assert result.status == "failed"
    assert result.error == (
        "Action confirmation was not granted."
    )
    assert confirmations == [
        "confirmation required",
    ]
    assert router.calls == 0


def test_autonomous_safety_denial_preserves_request_identity() -> None:
    router = ExplodingRouter()

    request = _request(
        name="blocked",
    )

    pipeline = ActionSafetyPipeline(
        router,
        safety_policy=CallbackActionPolicy(
            lambda current: ActionPolicyDecision(
                ActionDecision.DENY,
                "safety blocked",
            )
        ),
    )

    bridge = AutonomousActionBridge(
        pipeline
    )

    result = bridge.execute(
        "run-identity",
        "task-identity",
        request,
    )

    assert result.status == "failed"
    assert result.error == "safety blocked"
    assert request.request_id == (
        "safety-010-blocked"
    )
    assert router.calls == 0


def test_autonomous_allowed_action_reaches_router() -> None:
    class SuccessRouter:
        def __init__(self) -> None:
            self.calls: list[ActionRequest] = []

        def dispatch(
            self,
            request: ActionRequest,
        ):
            from osa.actions.contracts import ActionResult

            self.calls.append(request)

            return ActionResult.succeeded(
                request.request_id,
                output="executed",
            )

    router = SuccessRouter()

    pipeline = ActionSafetyPipeline(
        router,
    )

    bridge = AutonomousActionBridge(
        pipeline
    )

    request = _request(
        name="allowed",
    )

    result = bridge.execute(
        "run-allowed",
        "task-allowed",
        request,
    )

    assert result.status == "completed"
    assert result.output == "executed"
    assert router.calls == [request]
