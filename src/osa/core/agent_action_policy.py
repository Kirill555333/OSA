"""Compatibility policies for the unified agent TOOL action pipeline."""

from __future__ import annotations

from collections.abc import Callable

from osa.actions.contracts import ActionKind, ActionRequest
from osa.actions.pipeline import (
    ActionConfirmationHandler,
    ActionDecision,
    ActionPolicy,
    ActionPolicyDecision,
)
from osa.core.modes import AgentMode, AgentModePolicy
from osa.permissions import (
    ConfirmationHandler,
    ConfirmationRequest,
    PermissionLevel,
    PermissionPolicy,
)


class AgentModeActionPolicy(ActionPolicy):
    """Adapt the legacy AgentModePolicy to the action policy protocol."""

    def __init__(
        self,
        mode_policy: AgentModePolicy,
        mode_getter: Callable[[], AgentMode],
    ) -> None:
        self._mode_policy = mode_policy
        self._mode_getter = mode_getter

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        """Evaluate whether the current mode supports the action."""
        if request.kind is not ActionKind.TOOL:
            return ActionPolicyDecision(
                ActionDecision.DENY,
                "Agent mode policy only permits TOOL actions in this "
                "compatibility pipeline.",
            )

        mode = self._mode_getter()

        if self._mode_policy.supports_tool(
            mode,
            request.name,
        ):
            return ActionPolicyDecision(
                ActionDecision.ALLOW,
                "",
            )

        return ActionPolicyDecision(
            ActionDecision.DENY,
            (
                f"Tool '{request.name}' is not available in agent "
                f"mode '{mode.value}'."
            ),
        )


class LegacyPermissionActionPolicy(ActionPolicy):
    """Adapt the legacy tool-name permission policy."""

    def __init__(
        self,
        permission_policy: PermissionPolicy,
        logger: Callable[..., None] | None = None,
    ) -> None:
        self._permission_policy = permission_policy
        self._logger = logger

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        """Map a legacy PermissionDecision to ActionPolicyDecision."""
        if request.kind is not ActionKind.TOOL:
            return ActionPolicyDecision(
                ActionDecision.DENY,
                "Legacy tool permission policy only supports TOOL actions.",
            )

        decision = self._permission_policy.decide(request.name)

        if self._logger is not None:
            self._logger(
                "permission.decision",
                tool=request.name,
                level=decision.level.value,
                reason=decision.reason,
            )

        if decision.level is PermissionLevel.ALLOW:
            return ActionPolicyDecision(
                ActionDecision.ALLOW,
                "",
            )

        if decision.level is PermissionLevel.CONFIRM:
            return ActionPolicyDecision(
                ActionDecision.CONFIRM,
                decision.reason
                or (
                    f"Confirmation is required for tool "
                    f"'{request.name}'."
                ),
            )

        return ActionPolicyDecision(
            ActionDecision.DENY,
            (
                f"Permission denied for tool '{request.name}': "
                f"{decision.reason}"
            ),
        )


class LegacyConfirmationActionHandler(ActionConfirmationHandler):
    """Adapt the legacy confirmation handler to unified actions."""

    def __init__(
        self,
        confirmation_handler: ConfirmationHandler,
    ) -> None:
        self._confirmation_handler = confirmation_handler

    def confirm(
        self,
        request: ActionRequest,
        reason: str,
    ) -> bool:
        """Request confirmation using the legacy handler contract."""
        if request.kind is not ActionKind.TOOL:
            return False

        return self._confirmation_handler.request(
            ConfirmationRequest(
                tool_name=request.name,
                description=(
                    f"OSA wants to execute tool '{request.name}'. "
                    "Do you allow this action?"
                ),
            )
        )
