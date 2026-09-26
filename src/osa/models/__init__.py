"""Model layer for OSA."""

from osa.models.interface import (
    ChatMessage,
    ModelConnectionError,
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ToolCall,
    ToolDefinition,
)
from osa.models.local import LlamaCppConfig, LlamaCppModel
from osa.models.registry import ModelRegistry

__all__ = [
    "ChatMessage",
    "LlamaCppConfig",
    "LlamaCppModel",
    "ModelConnectionError",
    "ModelError",
    "ModelInterface",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseError",
    "ToolCall",
    "ToolDefinition",
]
