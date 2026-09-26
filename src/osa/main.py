"""Command-line entry point for OSA."""

from __future__ import annotations

from pathlib import Path

from osa.core import Agent, AgentError
from osa.models import LlamaCppConfig, LlamaCppModel, ModelConnectionError
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry
from osa.utils import EventLogger


SYSTEM_PROMPT = """You are OSA, a personal AI assistant.
Be helpful, clear, concise, and honest.
Answer in the same language as the user unless the user asks otherwise.

You have access to tools.
When a tool is appropriate for the task, use the available tool instead of
performing the operation yourself.
"""


def create_agent() -> Agent:
    """Create the OSA agent with its local model and tools."""
    model = LlamaCppModel(
        LlamaCppConfig(
            base_url="http://127.0.0.1:8080",
            timeout=120.0,
        )
    )

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    permission_policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    event_logger = EventLogger(
        Path.cwd() / "logs" / "osa-events.jsonl"
    )

    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=512,
        tool_registry=tool_registry,
        event_logger=event_logger,
        permission_policy=permission_policy,
    )


def print_tools(agent: Agent) -> None:
    """Print the tools currently available to OSA."""
    print("Available tools:")

    for tool in agent.tools.describe():
        print(
            f"- {tool['name']}: "
            f"{tool['description']}"
        )

    print()


def run_chat() -> None:
    """Run the interactive OSA chat."""
    agent = create_agent()

    if not agent.model.health_check():
        raise ModelConnectionError(
            "The local llama.cpp server is not available at "
            "http://127.0.0.1:8080."
        )

    print("OSA v0.1.7")
    print(f"Local model: {agent.model.model_name}")
    print_tools(agent)

    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOSA stopped.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("OSA stopped.")
            break

        try:
            response = agent.chat(user_input)
        except AgentError as exc:
            print(f"OSA error: {exc}")
            print()
            continue

        print(f"OSA: {response.content}")
        print()


def main() -> None:
    """Start OSA."""
    try:
        run_chat()
    except ModelConnectionError as exc:
        print(f"OSA startup error: {exc}")


if __name__ == "__main__":
    main()
