from __future__ import annotations

from osa.actions import (
    ActionDecision,
    ActionKind,
    ActionPolicyDecision,
    ActionRequest,
    ActionResult,
    ActionRouter,
    ActionSafetyPipeline,
    CallbackActionConfirmationHandler,
    MappingActionModePolicy,
    RuleActionSafetyPolicy,
)


class RecordingPolicy:
    def __init__(
        self,
        name: str,
        decision: ActionPolicyDecision,
        calls: list[str],
    ) -> None:
        self.name = name
        self.decision = decision
        self.calls = calls

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        self.calls.append(self.name)
        return self.decision


class RecordingHandler:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        self.calls.append("router")
        return ActionResult.succeeded(
            request.request_id,
            "executed",
        )


def make_request(
    name: str = "browser_open",
    arguments: dict[str, object] | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.BROWSER,
        name=name,
        arguments=arguments or {},
    )


def make_router(handler: RecordingHandler) -> ActionRouter:
    return ActionRouter(
        {ActionKind.BROWSER: handler}
    )


def test_pipeline_runs_in_required_order() -> None:
    calls: list[str] = []

    mode = RecordingPolicy(
        "mode",
        ActionPolicyDecision(ActionDecision.ALLOW),
        calls,
    )
    permission = RecordingPolicy(
        "permission",
        ActionPolicyDecision(ActionDecision.ALLOW),
        calls,
    )
    safety = RecordingPolicy(
        "safety",
        ActionPolicyDecision(ActionDecision.ALLOW),
        calls,
    )

    result = ActionSafetyPipeline(
        make_router(RecordingHandler(calls)),
        mode_policy=mode,
        permission_policy=permission,
        safety_policy=safety,
    ).dispatch(make_request())

    assert result.success is True
    assert calls == [
        "mode",
        "permission",
        "safety",
        "router",
    ]


def test_mode_denial_stops_pipeline() -> None:
    calls: list[str] = []

    result = ActionSafetyPipeline(
        make_router(RecordingHandler(calls)),
        mode_policy=RecordingPolicy(
            "mode",
            ActionPolicyDecision(
                ActionDecision.DENY,
                "mode denied",
            ),
            calls,
        ),
        permission_policy=RecordingPolicy(
            "permission",
            ActionPolicyDecision(ActionDecision.ALLOW),
            calls,
        ),
        safety_policy=RecordingPolicy(
            "safety",
            ActionPolicyDecision(ActionDecision.ALLOW),
            calls,
        ),
    ).dispatch(make_request())

    assert result.success is False
    assert result.error == "mode denied"
    assert result.metadata["pipeline_stage"] == "mode"
    assert calls == ["mode"]


def test_permission_denial_stops_before_safety() -> None:
    calls: list[str] = []

    result = ActionSafetyPipeline(
        make_router(RecordingHandler(calls)),
        mode_policy=RecordingPolicy(
            "mode",
            ActionPolicyDecision(ActionDecision.ALLOW),
            calls,
        ),
        permission_policy=RecordingPolicy(
            "permission",
            ActionPolicyDecision(
                ActionDecision.DENY,
                "permission denied",
            ),
            calls,
        ),
        safety_policy=RecordingPolicy(
            "safety",
            ActionPolicyDecision(ActionDecision.ALLOW),
            calls,
        ),
    ).dispatch(make_request())

    assert result.success is False
    assert result.error == "permission denied"
    assert result.metadata["pipeline_stage"] == "permission"
    assert calls == ["mode", "permission"]


def test_confirmation_is_required_before_router() -> None:
    calls: list[str] = []

    class Confirmation:
        def confirm(
            self,
            request: ActionRequest,
            reason: str,
        ) -> bool:
            calls.append("confirmation")
            assert reason == "needs approval"
            return True

    result = ActionSafetyPipeline(
        make_router(RecordingHandler(calls)),
        mode_policy=RecordingPolicy(
            "mode",
            ActionPolicyDecision(ActionDecision.ALLOW),
            calls,
        ),
        permission_policy=RecordingPolicy(
            "permission",
            ActionPolicyDecision(ActionDecision.ALLOW),
            calls,
        ),
        safety_policy=RecordingPolicy(
            "safety",
            ActionPolicyDecision(
                ActionDecision.CONFIRM,
                "needs approval",
            ),
            calls,
        ),
        confirmation_handler=Confirmation(),
    ).dispatch(make_request())

    assert result.success is True
    assert calls == [
        "mode",
        "permission",
        "safety",
        "confirmation",
        "router",
    ]


def test_confirmation_rejection_stops_router() -> None:
    calls: list[str] = []

    result = ActionSafetyPipeline(
        make_router(RecordingHandler(calls)),
        safety_policy=RecordingPolicy(
            "safety",
            ActionPolicyDecision(
                ActionDecision.CONFIRM,
                "needs approval",
            ),
            calls,
        ),
        confirmation_handler=CallbackActionConfirmationHandler(
            lambda request, reason: False
        ),
    ).dispatch(make_request())

    assert result.success is False
    assert result.error == "Action confirmation was not granted."
    assert result.metadata["pipeline_stage"] == "confirmation"
    assert calls == ["safety"]


def test_confirm_without_handler_fails_closed() -> None:
    result = ActionSafetyPipeline(
        make_router(RecordingHandler([])),
        safety_policy=RecordingPolicy(
            "safety",
            ActionPolicyDecision(
                ActionDecision.CONFIRM,
                "needs approval",
            ),
            [],
        ),
    ).dispatch(make_request())

    assert result.success is False
    assert result.metadata["pipeline_error"] == (
        "confirmation_not_configured"
    )


def test_rule_safety_policy_blocks_action_name() -> None:
    policy = RuleActionSafetyPolicy(
        blocked_action_names=frozenset({"browser_delete"}),
    )

    result = policy.evaluate(
        make_request("browser_delete")
    )

    assert result.decision is ActionDecision.DENY
    assert "blocked by safety policy" in result.reason


def test_rule_safety_policy_confirms_sensitive_argument() -> None:
    policy = RuleActionSafetyPolicy(
        confirmation_argument_keys=frozenset({
            "password",
            "token",
        }),
    )

    result = policy.evaluate(
        make_request(
            arguments={"password": "secret"}
        )
    )

    assert result.decision is ActionDecision.CONFIRM
    assert "password" in result.reason


def test_mapping_mode_policy_allows_matching_kind() -> None:
    policy = MappingActionModePolicy(
        allowed_kinds_by_mode={
            "chat": frozenset({ActionKind.TOOL}),
            "automation": frozenset({
                ActionKind.BROWSER,
                ActionKind.DESKTOP,
                ActionKind.TOOL,
            }),
        },
        current_mode="automation",
    )

    result = policy.evaluate(make_request())

    assert result.decision is ActionDecision.ALLOW


def test_mapping_mode_policy_denies_missing_kind() -> None:
    policy = MappingActionModePolicy(
        allowed_kinds_by_mode={
            "chat": frozenset({ActionKind.TOOL}),
        },
        current_mode="chat",
    )

    result = policy.evaluate(make_request())

    assert result.decision is ActionDecision.DENY
    assert "not available" in result.reason
