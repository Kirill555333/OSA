from __future__ import annotations

from osa.planning import Plan, PlanStep
from osa.tasks.verification import (
    TaskVerifier,
    VerificationResult,
)
from osa.models import ModelInterface
from osa.tasks import (
    create_default_planned_task_runner,
    ActionResolver,
    PlannedTaskRunner,
    TaskAction,
    TaskExecutor,
    TaskManager,
)
from osa.tools import ToolResult


class FakePlanner:
    """Deterministic planner for runner tests."""

    def create_plan(
        self,
        goal: str,
        *,
        context: str | None = None,
    ) -> Plan:
        assert goal == "Build report"

        return Plan(
            goal=goal,
            steps=(
                PlanStep(
                    step_id="collect",
                    description="Collect data",
                ),
                PlanStep(
                    step_id="write",
                    description="Write report",
                    depends_on=("collect",),
                ),
            ),
        )


class FakeResolver(ActionResolver):
    """Resolve tasks to deterministic fake tools."""

    def resolve(
        self,
        task,
        outputs,
    ) -> TaskAction:
        if task.task_id == "collect":
            return TaskAction(
                tool_name="collect",
                arguments={},
            )

        assert outputs["collect"] == "collected"

        return TaskAction(
            tool_name="write",
            arguments={
                "content": outputs["collect"],
            },
        )



def verify_success(
    task,
    action,
    result,
    outputs,
) -> VerificationResult:
    return VerificationResult(
        verified=True,
        message=f"Verified {task.task_id}.",
    )


def fake_tool_executor(
    tool_name: str,
    arguments,
) -> ToolResult:
    if tool_name == "collect":
        return ToolResult(
            success=True,
            output="collected",
        )

    if tool_name == "write":
        return ToolResult(
            success=True,
            output="report written",
        )

    return ToolResult(
        success=False,
        error=f"Unknown tool: {tool_name}",
    )


def test_planned_task_runner_connects_pipeline() -> None:
    manager = TaskManager()
    executor = TaskExecutor(
        manager
    )

    runner = PlannedTaskRunner(
        planner=FakePlanner(),
        resolver=FakeResolver(),
        manager=manager,
        executor=executor,
        tool_executor=fake_tool_executor,
        verifier=TaskVerifier({
            "collect": verify_success,
            "write": verify_success,
        }),
    )

    report = runner.run(
        "Build report",
        context="Use project data.",
    )

    assert report.success is True
    assert report.completed == (
        "collect",
        "write",
    )
    assert report.outputs == {
        "collect": "collected",
        "write": "report written",
    }


def test_planned_task_runner_propagates_tool_failure() -> None:
    class FailingResolver(ActionResolver):
        def resolve(
            self,
            task,
            outputs,
        ) -> TaskAction:
            return TaskAction(
                tool_name="collect",
                arguments={},
            )

    def fail_tool(
        tool_name: str,
        arguments,
    ) -> ToolResult:
        return ToolResult(
            success=False,
            error="Permission refused.",
        )

    manager = TaskManager()
    executor = TaskExecutor(
        manager
    )

    runner = PlannedTaskRunner(
        planner=FakePlanner(),
        resolver=FailingResolver(),
        manager=manager,
        executor=executor,
        tool_executor=fail_tool,
        verifier=TaskVerifier({
            "collect": verify_success,
        }),
    )

    report = runner.run(
        "Build report"
    )

    assert report.success is False
    assert report.failed == (
        "collect",
    )
    assert report.errors == {
        "collect": "Permission refused.",
    }
    assert report.blocked == (
        "write",
    )


def test_default_planned_task_runner_includes_verification(
    tmp_path,
) -> None:
    class DummyModel(ModelInterface):
        @property
        def model_name(self) -> str:
            return "dummy"

        def generate(
            self,
            request,
        ):
            raise AssertionError(
                "Model should not be called in this construction test."
            )

        def health_check(self) -> bool:
            return True

    filesystem = tmp_path / "workspace"
    filesystem.mkdir()

    tool_registry = __import__(
        "osa.tools",
        fromlist=["ToolRegistry"],
    ).ToolRegistry()

    runner = create_default_planned_task_runner(
        DummyModel(),
        tool_registry,
        lambda tool_name, arguments: None,
        filesystem=__import__(
            "osa.tools",
            fromlist=["SafeFilesystem"],
        ).SafeFilesystem(filesystem),
    )

    assert runner.verifier is not None
