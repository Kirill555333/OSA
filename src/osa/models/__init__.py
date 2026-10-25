"""Model layer for OSA."""

from osa.models.config import ModelConfig
from osa.models.interface import (
    ChatMessage,
    ModelConnectionError,
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ToolCall,
    ToolDefinition,
)
from osa.models.local import LlamaCppConfig, LlamaCppModel
from osa.models.registry import ModelRegistry

__all__ = [
    "ChatMessage",
    "LlamaCppConfig",
    "LlamaCppModel",
    "ModelConfig",
    "ModelConnectionError",
    "ModelError",
    "ModelInterface",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseError",
    "ModelStreamEvent",
    "ToolCall",
    "ToolDefinition",
]
