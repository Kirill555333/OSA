"""Unified action infrastructure for OSA."""

from osa.actions.browser import (
    BROWSER_ACTION_NAMES,
    BrowserActionAdapter,
    BrowserActionAdapterError,
    BrowserActionTool,
)
from osa.actions.contracts import (
    ActionContractError,
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.actions.desktop import (
    DESKTOP_ACTION_NAMES,
    DesktopActionAdapter,
    DesktopActionAdapterError,
)
from osa.actions.pipeline import (
    ActionConfirmationHandler,
    ActionDecision,
    ActionPolicy,
    ActionPolicyDecision,
    ActionSafetyError,
    ActionSafetyPipeline,
    AllowAllActionPolicy,
    CallbackActionConfirmationHandler,
    CallbackActionPolicy,
    MappingActionModePolicy,
    RuleActionSafetyPolicy,
)
from osa.actions.router import (
    ActionHandler,
    ActionRouter,
    ActionRouterError,
)

__all__ = [
    "ActionContractError",
    "ActionKind",
    "ActionRequest",
    "ActionResult",
    "ActionConfirmationHandler",
    "ActionDecision",
    "ActionPolicy",
    "ActionPolicyDecision",
    "ActionSafetyError",
    "ActionSafetyPipeline",
    "AllowAllActionPolicy",
    "CallbackActionConfirmationHandler",
    "CallbackActionPolicy",
    "MappingActionModePolicy",
    "RuleActionSafetyPolicy",
    "ActionHandler",
    "ActionRouter",
    "ActionRouterError",
    "BROWSER_ACTION_NAMES",
    "BrowserActionAdapter",
    "BrowserActionAdapterError",
    "BrowserActionTool",
    "DESKTOP_ACTION_NAMES",
    "DesktopActionAdapter",
    "DesktopActionAdapterError",
]
