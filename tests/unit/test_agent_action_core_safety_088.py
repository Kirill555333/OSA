"""0.8.8.5 Safety invariants for the unified action core."""

from __future__ import annotations

import pytest

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.actions.pipeline import ActionSafetyPipeline
from osa.actions.router import ActionRouter
from osa.actions.tool import ToolRegistryActionAdapter
from osa.core import Agent, PermissionDeniedError
from osa.core.agent_action_policy import (
    LegacyConfirmationActionHandler,
    LegacyPermissionActionPolicy,
)
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import (
    ConfirmationHandler,
    PermissionLevel,
    PermissionPolicy,
)
from osa.recovery_action import RecoverableActionExecutor
from osa.recovery_contracts import RecoveryFailureKind
from osa.recovery_policy import (
    DefaultRecoveryPolicy,
    RecoveryPolicyConfig,
)
from osa.tools import ToolRegistry, ToolResult


class FakeModel:
    """Minimal model required by Agent."""

    def generate(self, request):
        return ModelResponse(
            content="ok",
            model_name="fake",
        )

    def generate_stream_events(self, request):
        raise NotImplementedError


class CountingTool:
    """Backend used to verify denied actions never execute."""

    name = "protected"
    description = "Protected test tool"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self,
        arguments,
    ) -> ToolResult:
        self.calls += 1

        return ToolResult(
            success=True,
            output="should-not-run",
        )


def _build_recoverable_agent(
    *,
    tool: CountingTool,
    permission_level: PermissionLevel,
    confirmation_handler: ConfirmationHandler | None = None,
    max_attempts: int = 3,
) -> Agent:
    registry = ToolRegistry()
    registry.register(tool)

    router = ActionRouter(
        {
            ActionKind.TOOL: ToolRegistryActionAdapter(
                registry
            )
        }
    )

    pipeline = ActionSafetyPipeline(
        router=router,
        permission_policy=LegacyPermissionActionPolicy(
            PermissionPolicy(
                {
                    "protected": permission_level,
                }
            )
        ),
        confirmation_handler=(
            LegacyConfirmationActionHandler(
                confirmation_handler
                or ConfirmationHandler(
                    callback=lambda _: True,
                )
            )
        ),
    )

    recoverable = RecoverableActionExecutor(
        pipeline.dispatch,
        max_attempts=max_attempts,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.BACKEND_ERROR,
                    }
                )
            )
        ),
        sleeper=lambda _: None,
    )

    return Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "protected": permission_level,
            }
        ),
        confirmation_handler=(
            confirmation_handler
            or ConfirmationHandler(
                callback=lambda _: True,
            )
        ),
        recovery_integration=AgentRecoveryIntegration(
            recoverable
        ),
    )


def test_permission_denial_stops_before_backend_and_retry() -> None:
    tool = CountingTool()

    agent = _build_recoverable_agent(
        tool=tool,
        permission_level=PermissionLevel.DENY,
        max_attempts=3,
    )

    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="protected",
        arguments={},
        request_id="safety-permission-088",
    )

    result = agent.execute_action_with_recovery(
        request
    )

    assert result.success is False
    assert result.failure_kind is (
        RecoveryFailureKind.PERMISSION_DENIED
    )
    assert result.attempt_count == 1
    assert result.exhausted is False
    assert tool.calls == 0


def test_confirmation_denial_stops_before_backend_and_retry() -> None:
    tool = CountingTool()

    confirmations: list[str] = []

    def deny_confirmation(
        description: str,
    ) -> bool:
        confirmations.append(description)
        return False

    agent = _build_recoverable_agent(
        tool=tool,
        permission_level=PermissionLevel.CONFIRM,
        confirmation_handler=ConfirmationHandler(
            callback=deny_confirmation,
        ),
        max_attempts=3,
    )

    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="protected",
        arguments={},
        request_id="safety-confirmation-088",
    )

    result = agent.execute_action_with_recovery(
        request
    )

    assert result.success is False
    assert result.failure_kind is (
        RecoveryFailureKind.CONFIRMATION_DENIED
    )
    assert result.attempt_count == 1
    assert tool.calls == 0
    assert len(confirmations) == 1
    assert "protected" in confirmations[0]


def test_legacy_execute_tool_preserves_permission_denial() -> None:
    tool = CountingTool()

    agent = _build_recoverable_agent(
        tool=tool,
        permission_level=PermissionLevel.DENY,
        max_attempts=3,
    )

    with pytest.raises(
        PermissionDeniedError,
        match="Permission denied",
    ):
        agent.execute_tool(
            "protected",
            {},
            request_id="safety-legacy-088",
        )

    assert tool.calls == 0


def test_safety_denial_cannot_become_successful_action_result() -> None:
    tool = CountingTool()

    agent = _build_recoverable_agent(
        tool=tool,
        permission_level=PermissionLevel.DENY,
        max_attempts=3,
    )

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="protected",
            arguments={},
            request_id="safety-success-088",
        )
    )

    assert result.success is False
    assert result.failure_kind is (
        RecoveryFailureKind.PERMISSION_DENIED
    )

    underlying = result.result

    if isinstance(
        underlying,
        ActionResult,
    ):
        assert underlying.success is False

    assert tool.calls == 0
