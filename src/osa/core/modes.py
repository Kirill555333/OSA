from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentMode(StrEnum):
    """Execution mode selected for the current agent interaction."""

    CHAT = "chat"
    TASK = "task"
    RESEARCH = "research"
    AUTONOMOUS = "autonomous"


@dataclass(frozen=True, slots=True)
class AgentModeProfile:
    """Behavioral constraints associated with an agent mode."""

    mode: AgentMode
    instruction: str
    allows_tools: bool
    allows_research: bool
    autonomous: bool
    max_tool_rounds: int

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty.")

        if self.max_tool_rounds <= 0:
            raise ValueError(
                "max_tool_rounds must be greater than zero."
            )

        if self.autonomous and not self.allows_tools:
            raise ValueError(
                "autonomous mode requires tool access."
            )

        if self.allows_research and not self.allows_tools:
            raise ValueError(
                "research mode requires tool access."
            )


_DEFAULT_PROFILES: dict[AgentMode, AgentModeProfile] = {
    AgentMode.CHAT: AgentModeProfile(
        mode=AgentMode.CHAT,
        instruction=(
            "Answer the user directly. Use tools only when they are "
            "needed to answer the request."
        ),
        allows_tools=True,
        allows_research=False,
        autonomous=False,
        max_tool_rounds=4,
    ),
    AgentMode.TASK: AgentModeProfile(
        mode=AgentMode.TASK,
        instruction=(
            "Work toward the requested task using available tools. "
            "Verify important results before reporting completion."
        ),
        allows_tools=True,
        allows_research=False,
        autonomous=False,
        max_tool_rounds=8,
    ),
    AgentMode.RESEARCH: AgentModeProfile(
        mode=AgentMode.RESEARCH,
        instruction=(
            "Gather external information through research tools, "
            "keep source context bounded, and distinguish retrieved "
            "facts from generated conclusions."
        ),
        allows_tools=True,
        allows_research=True,
        autonomous=False,
        max_tool_rounds=8,
    ),
    AgentMode.AUTONOMOUS: AgentModeProfile(
        mode=AgentMode.AUTONOMOUS,
        instruction=(
            "Execute the requested objective as an agent. Plan work, "
            "use available tools, verify results, recover from safe "
            "transient failures, and stop when the objective is complete."
        ),
        allows_tools=True,
        allows_research=True,
        autonomous=True,
        max_tool_rounds=12,
    ),
}


class AgentModePolicy:
    """Resolve validated profiles for agent execution modes."""

    def __init__(
        self,
        profiles: dict[AgentMode, AgentModeProfile] | None = None,
    ) -> None:
        self._profiles = dict(
            profiles or _DEFAULT_PROFILES
        )

        for mode in AgentMode:
            if mode not in self._profiles:
                raise ValueError(
                    f"Missing profile for agent mode: {mode.value}"
                )

    def profile(
        self,
        mode: AgentMode | str,
    ) -> AgentModeProfile:
        if isinstance(mode, str):
            try:
                mode = AgentMode(mode.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unknown agent mode: {mode!r}"
                ) from exc

        try:
            return self._profiles[mode]
        except KeyError as exc:
            raise ValueError(
                f"Unknown agent mode: {mode!r}"
            ) from exc

    def supports_tool(
        self,
        mode: AgentMode | str,
        tool_name: str,
    ) -> bool:
        profile = self.profile(mode)

        if not profile.allows_tools:
            return False

        if (
            tool_name == "web_research"
            and not profile.allows_research
        ):
            return False

        return True
