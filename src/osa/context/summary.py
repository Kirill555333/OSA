"""Structured conversation summary contract and builder for OSA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Mapping, Sequence

from osa.models import ChatMessage, ModelError, ModelInterface, ModelRequest


class SummaryError(RuntimeError):
    """Raised when conversation summary operations fail."""


_SENSITIVE_PATTERN = re.compile(
    r"\b(?:"
    r"парол\w*|"
    r"password\w*|"
    r"api[\s_-]*key\w*|"
    r"token\w*|"
    r"секрет\w*|"
    r"secret\w*|"
    r"private[\s_-]*key\w*|"
    r"ssh[\s_-]*key\w*|"
    r"seed[\s_-]*phrase\w*|"
    r"recovery[\s_-]*phrase\w*"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """Structured representation of prior conversation state."""

    goal: str = ""
    current_state: str = ""
    decisions: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    messages_summarized_count: int = 0
    updated_at: str = ""

    @property
    def is_empty(self) -> bool:
        """Return True if the summary contains no meaningful content."""
        return not (
            self.goal
            or self.current_state
            or self.decisions
            or self.facts
            or self.constraints
            or self.unresolved
        )

    @classmethod
    def empty(cls) -> ConversationSummary:
        """Return a blank conversation summary."""
        return cls(updated_at=datetime.now(timezone.utc).isoformat())

    def format_for_prompt(self) -> str:
        """Format structured summary as a concise reference context block."""
        if self.is_empty:
            return ""

        lines = ["[Prior Conversation Summary]"]

        if self.goal:
            lines.append(f"Goal: {self.goal}")

        if self.current_state:
            lines.append(f"Current State: {self.current_state}")

        if self.decisions:
            lines.append("Key Decisions:")
            for item in self.decisions:
                lines.append(f"- {item}")

        if self.facts:
            lines.append("Relevant Facts:")
            for item in self.facts:
                lines.append(f"- {item}")

        if self.constraints:
            lines.append("Constraints:")
            for item in self.constraints:
                lines.append(f"- {item}")

        if self.unresolved:
            lines.append("Unresolved Items:")
            for item in self.unresolved:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary state into a plain dictionary."""
        return {
            "goal": self.goal,
            "current_state": self.current_state,
            "decisions": list(self.decisions),
            "facts": list(self.facts),
            "constraints": list(self.constraints),
            "unresolved": list(self.unresolved),
            "messages_summarized_count": self.messages_summarized_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConversationSummary:
        """Restore ConversationSummary from serialized dictionary."""
        return cls(
            goal=str(data.get("goal", "")),
            current_state=str(data.get("current_state", "")),
            decisions=tuple(str(x) for x in data.get("decisions", [])),
            facts=tuple(str(x) for x in data.get("facts", [])),
            constraints=tuple(str(x) for x in data.get("constraints", [])),
            unresolved=tuple(str(x) for x in data.get("unresolved", [])),
            messages_summarized_count=int(data.get("messages_summarized_count", 0)),
            updated_at=str(data.get("updated_at", "")),
        )


class SummaryBuilder:
    """Build and update structured conversation summaries with privacy protection."""

    def __init__(
        self,
        model: ModelInterface | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def build_summary(
        self,
        messages: Sequence[ChatMessage],
        previous_summary: ConversationSummary | None = None,
    ) -> ConversationSummary:
        """Create or update a structured summary over a sequence of messages."""
        if not messages:
            return previous_summary or ConversationSummary.empty()

        if self._model is not None:
            try:
                return self._build_with_model(messages, previous_summary)
            except Exception:
                # Resilient fallback to deterministic extraction on model error
                pass

        return self._build_fallback(messages, previous_summary)

    def _build_with_model(
        self,
        messages: Sequence[ChatMessage],
        previous_summary: ConversationSummary | None,
    ) -> ConversationSummary:
        """Ask model to generate structured summary in JSON format."""
        prompt_lines = [
            "You are OSA's conversation summarizer.",
            "Analyze the conversation transcript and output a JSON object with these exact keys:",
            '- "goal": main topic or goal of conversation',
            '- "current_state": where the conversation or task left off',
            '- "decisions": list of key agreed conclusions or decisions (strings)',
            '- "facts": list of important surfaced user/system facts (strings)',
            '- "constraints": list of constraints or user preferences mentioned (strings)',
            '- "unresolved": list of pending questions or tasks (strings)',
            "Do not include credentials, passwords, or tokens. Return JSON only.",
        ]

        if previous_summary and not previous_summary.is_empty:
            prompt_lines.append(
                f"\nExisting summary to incorporate:\n{previous_summary.format_for_prompt()}"
            )

        transcript = "\n".join(
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages
            if msg.role != "system" and msg.content
        )

        prompt_lines.append(f"\nNew messages to summarize:\n{transcript}")

        req = ModelRequest(
            messages=(
                ChatMessage(
                    role="user",
                    content="\n".join(prompt_lines),
                ),
            ),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        assert self._model is not None
        response = self._model.generate(req)
        parsed = self._parse_json(response.content)

        prev_count = previous_summary.messages_summarized_count if previous_summary else 0
        total_count = prev_count + len(messages)

        raw_goal = str(parsed.get("goal", "")).strip()
        goal = "[Redacted sensitive goal]" if self._is_sensitive(raw_goal) else raw_goal

        raw_state = str(parsed.get("current_state", "")).strip()
        current_state = "[Redacted sensitive state]" if self._is_sensitive(raw_state) else raw_state

        decisions = tuple(
            str(x).strip()
            for x in parsed.get("decisions", [])
            if str(x).strip() and not self._is_sensitive(str(x))
        )
        facts = tuple(
            str(x).strip()
            for x in parsed.get("facts", [])
            if str(x).strip() and not self._is_sensitive(str(x))
        )
        constraints = tuple(
            str(x).strip()
            for x in parsed.get("constraints", [])
            if str(x).strip() and not self._is_sensitive(str(x))
        )
        unresolved = tuple(
            str(x).strip()
            for x in parsed.get("unresolved", [])
            if str(x).strip() and not self._is_sensitive(str(x))
        )

        return ConversationSummary(
            goal=goal,
            current_state=current_state,
            decisions=decisions,
            facts=facts,
            constraints=constraints,
            unresolved=unresolved,
            messages_summarized_count=total_count,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _build_fallback(
        self,
        messages: Sequence[ChatMessage],
        previous_summary: ConversationSummary | None,
    ) -> ConversationSummary:
        """Deterministic heuristic extraction with sensitive data redaction."""
        prev_goal = previous_summary.goal if previous_summary else ""
        prev_state = previous_summary.current_state if previous_summary else ""
        decisions = list(previous_summary.decisions) if previous_summary else []
        facts = list(previous_summary.facts) if previous_summary else []
        constraints = list(previous_summary.constraints) if previous_summary else []
        unresolved = list(previous_summary.unresolved) if previous_summary else []

        goal = prev_goal
        last_assistant_reply = ""

        for msg in messages:
            content = msg.content.strip()
            if not content:
                continue

            if msg.role == "user":
                if not goal and not self._is_sensitive(content):
                    goal = content.splitlines()[0][:120]
                if any(word in content.lower() for word in ("не ", "must", "only", "только")):
                    if not self._is_sensitive(content):
                        constraints.append(content[:100])
                if "?" in content and not self._is_sensitive(content):
                    unresolved.append(content[:100])

            elif msg.role == "assistant":
                if not self._is_sensitive(content):
                    last_assistant_reply = content.splitlines()[0][:120]
                if any(kw in content.lower() for kw in ("решено", "выполнено", "agreed", "done")):
                    if not self._is_sensitive(content):
                        decisions.append(content[:100])

        current_state = last_assistant_reply or prev_state or "In progress"

        prev_count = previous_summary.messages_summarized_count if previous_summary else 0
        total_count = prev_count + len(messages)

        return ConversationSummary(
            goal=goal,
            current_state=current_state,
            decisions=tuple(dict.fromkeys(decisions))[:10],
            facts=tuple(dict.fromkeys(facts))[:10],
            constraints=tuple(dict.fromkeys(constraints))[:10],
            unresolved=tuple(dict.fromkeys(unresolved))[:10],
            messages_summarized_count=total_count,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _is_sensitive(text: str) -> bool:
        """Check whether text contains sensitive credentials or keys."""
        return bool(_SENSITIVE_PATTERN.search(text))

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Extract and parse JSON object from model response."""
        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1]).strip()

        first_brace = content.find("{")
        last_brace = content.rfind("}")

        if first_brace >= 0 and last_brace > first_brace:
            content = content[first_brace : last_brace + 1]

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise SummaryError(f"Invalid JSON returned: {exc}") from exc

        raise SummaryError("Summary output was not a JSON object.")
