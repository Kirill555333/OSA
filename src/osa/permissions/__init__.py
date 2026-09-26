"""Permission components for OSA."""

from osa.permissions.confirmation import (
    ConfirmationHandler,
    ConfirmationRequest,
)
from osa.permissions.policy import (
    PermissionDecision,
    PermissionLevel,
    PermissionPolicy,
)

__all__ = [
    "ConfirmationHandler",
    "ConfirmationRequest",
    "PermissionDecision",
    "PermissionLevel",
    "PermissionPolicy",
]
