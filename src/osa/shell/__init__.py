"""Interactive shell and console tools for OSA."""

from osa.shell.console import (
    ConsoleColors,
    InteractiveShell,
    create_interactive_confirmation_handler,
    terminal_confirmation_prompt,
)

__all__ = [
    "ConsoleColors",
    "InteractiveShell",
    "create_interactive_confirmation_handler",
    "terminal_confirmation_prompt",
]
