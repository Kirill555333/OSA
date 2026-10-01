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
from osa.actions.desktop_safety import (
    CompositeActionPolicy,
    DesktopActionSafetyPolicy,
    DesktopSafetyConfig,
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
    "ActionConfirmationHandler",
    "ActionContractError",
    "ActionDecision",
    "ActionHandler",
    "ActionKind",
    "ActionPolicy",
    "ActionPolicyDecision",
    "ActionRequest",
    "ActionResult",
    "ActionRouter",
    "ActionRouterError",
    "ActionSafetyError",
    "ActionSafetyPipeline",
    "AllowAllActionPolicy",
    "BROWSER_ACTION_NAMES",
    "BrowserActionAdapter",
    "BrowserActionAdapterError",
    "BrowserActionTool",
    "CallbackActionConfirmationHandler",
    "CallbackActionPolicy",
    "CompositeActionPolicy",
    "DESKTOP_ACTION_NAMES",
    "DesktopActionAdapter",
    "DesktopActionAdapterError",
    "DesktopActionSafetyPolicy",
    "DesktopSafetyConfig",
    "MappingActionModePolicy",
    "RuleActionSafetyPolicy",
]
