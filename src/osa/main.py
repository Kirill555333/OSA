"""Main entry point for OSA."""

from __future__ import annotations

from pathlib import Path
import sys

from osa.browser import HttpBrowser
from osa.core import Agent, AgentError
from osa.memory import (
    AutomaticMemory,
    LongTermMemory,
)
from osa.memory.integration import MemoryIntegration
from osa.memory.retrieval import MemoryRetriever
from osa.models import (
    LlamaCppConfig,
    LlamaCppModel,
    ModelConnectionError,
)
from osa.permissions import (
    PermissionLevel,
    PermissionPolicy,
)
from osa.recovery import ErrorRecovery
from osa.shell import (
    InteractiveShell,
    create_interactive_confirmation_handler,
)
from osa.system import create_system_provider
from osa.tools import (
    BrowserFetchTool,
    CalculatorTool,
    FileExistsTool,
    FindFilesTool,
    ForgetTool,
    ListDirectoryTool,
    PatchFileTool,
    ReadFileTool,
    RecallTool,
    RememberTool,
    SafeFilesystem,
    SafeShell,
    SystemInfoTool,
    TerminalExecuteTool,
    ToolRegistry,
    WebResearchTool,
    WriteFileTool,
)
from osa.utils.logging import EventLogger

SYSTEM_PROMPT = (
    "You are OSA, a personal AI agent inspired by JARVIS. "
    "You communicate clearly, think carefully before taking actions, "
    "and use tools safely to help the user manage their computer, tasks, and knowledge."
)


def create_agent() -> Agent:
    """Build and configure the main OSA agent."""
    model = LlamaCppModel(
        LlamaCppConfig(
            base_url="http://127.0.0.1:8080",
            timeout=120.0,
        )
    )

    workspace_dir = Path.cwd()
    data_dir = workspace_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tool_registry = ToolRegistry()

    # Core tools
    tool_registry.register(CalculatorTool())
    tool_registry.register(SystemInfoTool(create_system_provider()))

    # Filesystem and Shell tools
    filesystem = SafeFilesystem(workspace_dir)
    tool_registry.register(ListDirectoryTool(filesystem))
    tool_registry.register(FindFilesTool(filesystem))
    tool_registry.register(ReadFileTool(filesystem))
    tool_registry.register(WriteFileTool(filesystem))
    tool_registry.register(PatchFileTool(filesystem))
    tool_registry.register(FileExistsTool(filesystem))

    safe_shell = SafeShell(workspace_dir)
    tool_registry.register(TerminalExecuteTool(safe_shell))

    # Browser tools
    browser = HttpBrowser()
    tool_registry.register(BrowserFetchTool(browser))
    tool_registry.register(WebResearchTool(browser))

    # Long-term Memory tools
    memory = LongTermMemory(data_dir / "osa-memory.db")
    automatic_memory = AutomaticMemory(memory)

    tool_registry.register(RememberTool(memory))
    tool_registry.register(RecallTool(memory))
    tool_registry.register(ForgetTool(memory))

    memory_retriever = MemoryRetriever(memory)
    memory_integration = MemoryIntegration(
        memory_retriever,
        limit=3,
        max_query_terms=8,
        max_prompt_characters=3000,
    )

    permission_policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
            "browser_fetch": PermissionLevel.ALLOW,
            "web_research": PermissionLevel.ALLOW,
            "list_directory": PermissionLevel.ALLOW,
            "find_files": PermissionLevel.ALLOW,
            "read_file": PermissionLevel.ALLOW,
            "file_exists": PermissionLevel.ALLOW,
            "system_info": PermissionLevel.ALLOW,
            "recall": PermissionLevel.ALLOW,
            "remember": PermissionLevel.ALLOW,
            "forget": PermissionLevel.CONFIRM,
            "write_file": PermissionLevel.CONFIRM,
            "patch_file": PermissionLevel.CONFIRM,
            "terminal_execute": PermissionLevel.CONFIRM,
        }
    )

    confirmation_handler = create_interactive_confirmation_handler()

    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    event_logger = EventLogger(logs_dir / "osa-events.jsonl")

    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.2,
        automatic_memory=automatic_memory,
        max_tokens=512,
        tool_registry=tool_registry,
        event_logger=event_logger,
        permission_policy=permission_policy,
        confirmation_handler=confirmation_handler,
        recovery=ErrorRecovery(),
        memory_integration=memory_integration,
    )


def main() -> None:
    """Start OSA interactive shell."""
    try:
        agent = create_agent()
        shell = InteractiveShell(agent)
        shell.run()
    except ModelConnectionError as exc:
        print(f"\nOSA startup error: {exc}")
        print("Please ensure the llama.cpp server is running at http://127.0.0.1:8080.\n")
    except KeyboardInterrupt:
        print("\nOSA stopped.")


if __name__ == "__main__":
    main()
