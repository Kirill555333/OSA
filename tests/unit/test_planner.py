from typing import Any

import pytest

from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.planning import (
    Plan,
    PlanStep,
    Planner,
    PlannerError,
)


class FakePlanningModel(ModelInterface):
    """Fake model for planner tests."""

    def __init__(
        self,
        response: ModelResponse,
    ) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "planner-test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.requests.append(request)
        return self.response

    def health_check(self) -> bool:
        return True


def make_tool_call_response(
    arguments: dict[str, Any],
) -> ModelResponse:
    return ModelResponse(
        content="",
        model_name="planner-test-model",
        finish_reason="tool_calls",
        usage={},
        tool_calls=(
            ToolCall(
                id="plan-call-1",
                name="submit_plan",
                arguments=arguments,
            ),
        ),
    )


def test_planner_generates_structured_plan() -> None:
    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Prepare OSA release",
                "steps": [
                    {
                        "id": "inspect",
                        "description": "Inspect the repository.",
                        "depends_on": [],
                    },
                    {
                        "id": "test",
                        "description": "Run the test suite.",
                        "depends_on": ["inspect"],
                    },
                ],
            }
        )
    )

    planner = Planner(model)

    plan = planner.create_plan(
        "Prepare OSA release"
    )

    assert isinstance(plan, Plan)
    assert plan.goal == "Prepare OSA release"
    assert plan.steps == (
        PlanStep(
            step_id="inspect",
            description="Inspect the repository.",
            depends_on=(),
        ),
        PlanStep(
            step_id="test",
            description="Run the test suite.",
            depends_on=("inspect",),
        ),
    )


def test_planner_sends_isolated_planning_request() -> None:
    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Test OSA",
                "steps": [
                    {
                        "id": "run",
                        "description": "Run tests.",
                        "depends_on": [],
                    },
                ],
            }
        )
    )

    planner = Planner(model)

    planner.create_plan(
        "Test OSA",
        context="OSA is a Python project.",
    )

    assert len(model.requests) == 1

    request = model.requests[0]

    assert len(request.tools) == 1
    assert request.tools[0].name == "submit_plan"

    assert request.messages[0].role == "system"
    assert "planning module" in (
        request.messages[0].content.lower()
    )

    assert request.messages[-1].role == "user"
    assert request.messages[-1].content == "Test OSA"


def test_planner_accepts_json_fallback() -> None:
    model = FakePlanningModel(
        ModelResponse(
            content=(
                '{"goal":"Test OSA","steps":['
                '{"id":"run",'
                '"description":"Run tests.",'
                '"depends_on":[]}'
                ']}'
            ),
            model_name="planner-test-model",
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )
    )

    planner = Planner(model)

    plan = planner.create_plan(
        "Test OSA"
    )

    assert plan.goal == "Test OSA"
    assert len(plan.steps) == 1


def test_planner_rejects_duplicate_step_ids() -> None:
    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Test",
                "steps": [
                    {
                        "id": "step",
                        "description": "First.",
                        "depends_on": [],
                    },
                    {
                        "id": "step",
                        "description": "Second.",
                        "depends_on": [],
                    },
                ],
            }
        )
    )

    planner = Planner(model)

    with pytest.raises(
        PlannerError,
        match="Duplicate plan step id",
    ):
        planner.create_plan("Test")


def test_planner_rejects_unknown_dependencies() -> None:
    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Test",
                "steps": [
                    {
                        "id": "step",
                        "description": "Do something.",
                        "depends_on": ["missing"],
                    },
                ],
            }
        )
    )

    planner = Planner(model)

    with pytest.raises(
        PlannerError,
        match="unknown step",
    ):
        planner.create_plan("Test")


def test_planner_rejects_dependency_cycles() -> None:
    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Test",
                "steps": [
                    {
                        "id": "one",
                        "description": "First.",
                        "depends_on": ["two"],
                    },
                    {
                        "id": "two",
                        "description": "Second.",
                        "depends_on": ["one"],
                    },
                ],
            }
        )
    )

    planner = Planner(model)

    with pytest.raises(
        PlannerError,
        match="dependency cycle",
    ):
        planner.create_plan("Test")


def test_planner_rejects_too_many_steps() -> None:
    steps = [
        {
            "id": str(index),
            "description": f"Step {index}.",
            "depends_on": [],
        }
        for index in range(1, 10)
    ]

    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Test",
                "steps": steps,
            }
        )
    )

    planner = Planner(
        model,
        max_steps=8,
    )

    with pytest.raises(
        PlannerError,
        match="more than 8 steps",
    ):
        planner.create_plan("Test")


def test_planner_rejects_empty_goal() -> None:
    model = FakePlanningModel(
        make_tool_call_response(
            {
                "goal": "Test",
                "steps": [
                    {
                        "id": "step",
                        "description": "Do something.",
                        "depends_on": [],
                    },
                ],
            }
        )
    )

    planner = Planner(model)

    with pytest.raises(ValueError):
        planner.create_plan("   ")
