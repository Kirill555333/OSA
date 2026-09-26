"""Task planning primitives for OSA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from osa.models import (
    ChatMessage,
    ModelError,
    ModelInterface,
    ModelRequest,
    ToolDefinition,
)


class PlannerError(RuntimeError):
    """Base exception raised by the planner."""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One step in a generated plan."""

    step_id: str
    description: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Plan:
    """A validated multi-step plan."""

    goal: str
    steps: tuple[PlanStep, ...]


class Planner:
    """Generate validated structured plans without executing them."""

    _PLAN_TOOL_NAME = "submit_plan"

    def __init__(
        self,
        model: ModelInterface,
        *,
        max_steps: int = 8,
        temperature: float = 0.1,
        max_tokens: int = 768,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> ModelInterface:
        """Return the model used for planning."""
        return self._model

    def create_plan(
        self,
        goal: str,
        *,
        context: str | None = None,
    ) -> Plan:
        """Generate and validate a plan for a user goal."""
        normalized_goal = goal.strip()

        if not normalized_goal:
            raise ValueError(
                "goal cannot be empty."
            )

        messages = self._build_messages(
            normalized_goal,
            context=context,
        )

        request = ModelRequest(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tools=(self._plan_tool_definition(),),
        )

        try:
            response = self._model.generate(request)
        except ModelError as exc:
            raise PlannerError(
                f"Planning model request failed: {exc}"
            ) from exc

        payload = self._extract_payload(response)

        return self._plan_from_payload(
            payload,
            fallback_goal=normalized_goal,
        )

    def _build_messages(
        self,
        goal: str,
        *,
        context: str | None,
    ) -> tuple[ChatMessage, ...]:
        """Build the isolated planning prompt."""
        system_content = (
            "You are OSA's planning module. "
            "Create a practical, ordered plan for the user's goal. "
            "Do not execute actions. "
            "Do not claim that anything was completed. "
            f"Use between 1 and {self._max_steps} steps. "
            "Each step must have a unique short id, a clear description, "
            "and optional dependencies on earlier step ids. "
            "Return the plan only by calling the submit_plan tool."
        )

        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=system_content,
            )
        ]

        if context:
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "Additional context for planning:\n"
                        f"{context.strip()}"
                    ),
                )
            )

        messages.append(
            ChatMessage(
                role="user",
                content=goal,
            )
        )

        return tuple(messages)

    def _extract_payload(
        self,
        response: Any,
    ) -> Mapping[str, Any]:
        """Extract a plan payload from a tool call or JSON content."""
        for tool_call in response.tool_calls:
            if tool_call.name != self._PLAN_TOOL_NAME:
                continue

            return tool_call.arguments

        content = response.content.strip()

        if not content:
            raise PlannerError(
                "Planning model returned neither a plan tool call "
                "nor JSON content."
            )

        parsed = self._parse_json_content(
            content
        )

        if not isinstance(parsed, Mapping):
            raise PlannerError(
                "Planning model returned invalid plan data."
            )

        return parsed

    @staticmethod
    def _parse_json_content(
        content: str,
    ) -> Any:
        """Parse JSON from plain or fenced model output."""
        candidates = [content]

        if content.startswith("```"):
            lines = content.splitlines()

            if (
                len(lines) >= 3
                and lines[-1].strip() == "```"
            ):
                candidates.append(
                    "\n".join(lines[1:-1]).strip()
                )

        first_brace = content.find("{")
        last_brace = content.rfind("}")

        if (
            first_brace >= 0
            and last_brace > first_brace
        ):
            candidates.append(
                content[
                    first_brace : last_brace + 1
                ]
            )

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise PlannerError(
            "Planning model returned invalid JSON."
        )

    def _plan_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_goal: str,
    ) -> Plan:
        """Validate and convert model data into a Plan."""
        goal = payload.get(
            "goal",
            fallback_goal,
        )

        steps_data = payload.get(
            "steps"
        )

        if not isinstance(goal, str):
            raise PlannerError(
                "Plan field 'goal' must be a string."
            )

        goal = goal.strip()

        if not goal:
            raise PlannerError(
                "Plan goal cannot be empty."
            )

        if not isinstance(steps_data, list):
            raise PlannerError(
                "Plan field 'steps' must be a list."
            )

        if not steps_data:
            raise PlannerError(
                "Plan must contain at least one step."
            )

        if len(steps_data) > self._max_steps:
            raise PlannerError(
                f"Plan cannot contain more than "
                f"{self._max_steps} steps."
            )

        steps: list[PlanStep] = []
        step_ids: set[str] = set()

        for raw_step in steps_data:
            if not isinstance(raw_step, Mapping):
                raise PlannerError(
                    "Each plan step must be an object."
                )

            step_id = raw_step.get(
                "id"
            )

            description = raw_step.get(
                "description"
            )

            depends_on = raw_step.get(
                "depends_on",
                [],
            )

            if not isinstance(step_id, str):
                raise PlannerError(
                    "Plan step id must be a string."
                )

            if not isinstance(description, str):
                raise PlannerError(
                    "Plan step description must be a string."
                )

            step_id = step_id.strip()
            description = description.strip()

            if not step_id:
                raise PlannerError(
                    "Plan step id cannot be empty."
                )

            if not description:
                raise PlannerError(
                    f"Plan step '{step_id}' "
                    "description cannot be empty."
                )

            if step_id in step_ids:
                raise PlannerError(
                    f"Duplicate plan step id: '{step_id}'."
                )

            if not isinstance(depends_on, list):
                raise PlannerError(
                    f"Dependencies for step '{step_id}' "
                    "must be a list."
                )

            normalized_dependencies = tuple(
                dependency.strip()
                for dependency in depends_on
                if isinstance(dependency, str)
                and dependency.strip()
            )

            if len(normalized_dependencies) != len(
                depends_on
            ):
                raise PlannerError(
                    f"Invalid dependency list for "
                    f"step '{step_id}'."
                )

            if step_id in normalized_dependencies:
                raise PlannerError(
                    f"Step '{step_id}' cannot depend on itself."
                )

            step_ids.add(step_id)

            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    depends_on=normalized_dependencies,
                )
            )

        for step in steps:
            unknown_dependencies = tuple(
                dependency
                for dependency in step.depends_on
                if dependency not in step_ids
            )

            if unknown_dependencies:
                raise PlannerError(
                    f"Step '{step.step_id}' depends on "
                    f"unknown step(s): "
                    f"{', '.join(unknown_dependencies)}."
                )

        self._validate_acyclic(
            tuple(steps)
        )

        return Plan(
            goal=goal,
            steps=tuple(steps),
        )

    @staticmethod
    def _validate_acyclic(
        steps: Sequence[PlanStep],
    ) -> None:
        """Reject dependency cycles."""
        dependencies = {
            step.step_id: set(step.depends_on)
            for step in steps
        }

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return

            if step_id in visiting:
                raise PlannerError(
                    "Plan contains a dependency cycle."
                )

            visiting.add(step_id)

            for dependency in dependencies[step_id]:
                visit(dependency)

            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in dependencies:
            visit(step_id)

    def _plan_tool_definition(self) -> ToolDefinition:
        """Return the synthetic tool used to submit a plan."""
        return ToolDefinition(
            name=self._PLAN_TOOL_NAME,
            description=(
                "Submit a structured execution plan. "
                "Do not execute the plan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The goal of the plan.",
                    },
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": self._max_steps,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                },
                                "description": {
                                    "type": "string",
                                },
                                "depends_on": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                    },
                                },
                            },
                            "required": [
                                "id",
                                "description",
                                "depends_on",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "goal",
                    "steps",
                ],
                "additionalProperties": False,
            },
        )
