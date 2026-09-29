"""Safety pipeline for unified OSA actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.actions.router import ActionRouter


class ActionSafetyError(RuntimeError):
    """Raised when the action safety pipeline is misconfigured."""


class ActionDecision(str, Enum):
    """Decision returned by an action policy stage."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ActionPolicyDecision:
    """Normalized result of one policy stage."""

    decision: ActionDecision
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ActionDecision):
            try:
                object.__setattr__(
                    self,
                    "decision",
                    ActionDecision(self.decision),
                )
            except (TypeError, ValueError) as exc:
                raise ActionSafetyError(
                    "decision must be a valid ActionDecision."
                ) from exc

        if not isinstance(self.reason, str):
            raise ActionSafetyError(
                "reason must be a string."
            )

        if self.decision is not ActionDecision.ALLOW and not self.reason.strip():
            raise ActionSafetyError(
                "deny and confirm decisions require a reason."
            )


class ActionPolicy(Protocol):
    """Protocol implemented by mode, permission, and safety policies."""

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        """Evaluate one action request."""
        ...


class ActionConfirmationHandler(Protocol):
    """Protocol for the final user confirmation step."""

    def confirm(
        self,
        request: ActionRequest,
        reason: str,
    ) -> bool:
        """Return whether the action is confirmed."""
        ...


@dataclass(frozen=True, slots=True)
class AllowAllActionPolicy:
    """Policy that permits all action requests."""

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        return ActionPolicyDecision(ActionDecision.ALLOW)


@dataclass(frozen=True, slots=True)
class MappingActionModePolicy:
    """Mode policy backed by an explicit mode-to-kind mapping."""

    allowed_kinds_by_mode: Mapping[str, frozenset[ActionKind]]
    current_mode: str

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        allowed = self.allowed_kinds_by_mode.get(
            self.current_mode,
            frozenset(),
        )

        if request.kind in allowed:
            return ActionPolicyDecision(ActionDecision.ALLOW)

        return ActionPolicyDecision(
            ActionDecision.DENY,
            (
                f"Action kind '{request.kind.value}' is not available "
                f"in mode '{self.current_mode}'."
            ),
        )


@dataclass(frozen=True, slots=True)
class RuleActionSafetyPolicy:
    """Configurable safety policy for unified actions."""

    blocked_action_names: frozenset[str] = frozenset()
    confirmation_action_names: frozenset[str] = frozenset()
    blocked_argument_keys: frozenset[str] = frozenset()
    confirmation_argument_keys: frozenset[str] = frozenset()

    @staticmethod
    def _normalize_names(values: frozenset[str]) -> frozenset[str]:
        return frozenset(
            value.strip().lower()
            for value in values
            if isinstance(value, str) and value.strip()
        )

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        action_name = request.name.lower()
        blocked_names = self._normalize_names(
            self.blocked_action_names
        )
        confirm_names = self._normalize_names(
            self.confirmation_action_names
        )
        blocked_keys = self._normalize_names(
            self.blocked_argument_keys
        )
        confirm_keys = self._normalize_names(
            self.confirmation_argument_keys
        )

        if action_name in blocked_names:
            return ActionPolicyDecision(
                ActionDecision.DENY,
                f"Action '{request.name}' is blocked by safety policy.",
            )

        argument_keys = {
            key.strip().lower()
            for key in request.arguments
        }

        matched_blocked_keys = sorted(
            argument_keys & blocked_keys
        )

        if matched_blocked_keys:
            return ActionPolicyDecision(
                ActionDecision.DENY,
                (
                    "Action contains blocked argument keys: "
                    + ", ".join(matched_blocked_keys)
                    + "."
                ),
            )

        if action_name in confirm_names:
            return ActionPolicyDecision(
                ActionDecision.CONFIRM,
                f"Action '{request.name}' requires confirmation.",
            )

        matched_confirm_keys = sorted(
            argument_keys & confirm_keys
        )

        if matched_confirm_keys:
            return ActionPolicyDecision(
                ActionDecision.CONFIRM,
                (
                    "Action contains sensitive argument keys: "
                    + ", ".join(matched_confirm_keys)
                    + "."
                ),
            )

        return ActionPolicyDecision(ActionDecision.ALLOW)


class CallbackActionPolicy:
    """Adapt a callable into an ActionPolicy."""

    def __init__(
        self,
        callback: Callable[
            [ActionRequest],
            ActionPolicyDecision,
        ],
    ) -> None:
        if not callable(callback):
            raise ActionSafetyError(
                "Action policy callback must be callable."
            )

        self._callback = callback

    def evaluate(
        self,
        request: ActionRequest,
    ) -> ActionPolicyDecision:
        result = self._callback(request)

        if not isinstance(result, ActionPolicyDecision):
            raise ActionSafetyError(
                "Action policy callback returned an invalid decision."
            )

        return result


class CallbackActionConfirmationHandler:
    """Adapt a callable into an action confirmation handler."""

    def __init__(
        self,
        callback: Callable[
            [ActionRequest, str],
            bool,
        ],
    ) -> None:
        if not callable(callback):
            raise ActionSafetyError(
                "Confirmation callback must be callable."
            )

        self._callback = callback

    def confirm(
        self,
        request: ActionRequest,
        reason: str,
    ) -> bool:
        result = self._callback(request, reason)

        if not isinstance(result, bool):
            raise ActionSafetyError(
                "Confirmation callback must return a boolean."
            )

        return result


class ActionSafetyPipeline:
    """Evaluate an action through ordered policy and confirmation stages."""

    def __init__(
        self,
        router: ActionRouter,
        *,
        mode_policy: ActionPolicy | None = None,
        permission_policy: ActionPolicy | None = None,
        safety_policy: ActionPolicy | None = None,
        confirmation_handler: ActionConfirmationHandler | None = None,
    ) -> None:
        self._router = router
        self._mode_policy = mode_policy or AllowAllActionPolicy()
        self._permission_policy = (
            permission_policy or AllowAllActionPolicy()
        )
        self._safety_policy = (
            safety_policy or AllowAllActionPolicy()
        )
        self._confirmation_handler = confirmation_handler

    def dispatch(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """Run the complete safety pipeline and then route the action."""
        if not isinstance(request, ActionRequest):
            raise ActionSafetyError(
                "request must be an ActionRequest."
            )

        stages = (
            ("mode", self._mode_policy),
            ("permission", self._permission_policy),
            ("safety", self._safety_policy),
        )

        for stage_name, policy in stages:
            try:
                decision = policy.evaluate(request)
            except Exception as exc:
                return self._stage_failure(
                    request,
                    stage_name,
                    "policy_exception",
                    f"{stage_name} policy failed: {exc}",
                )

            if not isinstance(decision, ActionPolicyDecision):
                return self._stage_failure(
                    request,
                    stage_name,
                    "invalid_policy_result",
                    f"{stage_name} policy returned an invalid decision.",
                )

            if decision.decision is ActionDecision.DENY:
                return self._stage_failure(
                    request,
                    stage_name,
                    "denied",
                    decision.reason,
                )

            if decision.decision is ActionDecision.CONFIRM:
                if self._confirmation_handler is None:
                    return self._stage_failure(
                        request,
                        stage_name,
                        "confirmation_not_configured",
                        (
                            f"{stage_name} policy requires confirmation, "
                            "but no confirmation handler is configured."
                        ),
                    )

                try:
                    confirmed = self._confirmation_handler.confirm(
                        request,
                        decision.reason,
                    )
                except Exception as exc:
                    return self._stage_failure(
                        request,
                        "confirmation",
                        "confirmation_exception",
                        f"Confirmation failed: {exc}",
                    )

                if not confirmed:
                    return self._stage_failure(
                        request,
                        "confirmation",
                        "not_confirmed",
                        "Action confirmation was not granted.",
                    )

        return self._router.dispatch(request)

    @staticmethod
    def _stage_failure(
        request: ActionRequest,
        stage: str,
        code: str,
        error: str,
    ) -> ActionResult:
        return ActionResult.failed(
            request.request_id,
            error,
            metadata={
                "pipeline_stage": stage,
                "pipeline_error": code,
                "action_kind": request.kind.value,
                "action_name": request.name,
            },
        )
