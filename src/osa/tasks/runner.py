"""Plan-to-execution orchestration for OSA."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from osa.planning import Plan, Planner
from osa.tasks.action import (
    TaskAction,
    TaskActionResolver,
)
from osa.tasks.executor import (
    TaskExecutionReport,
    TaskExecutor,
)
from osa.tasks.verification import TaskVerifier
from osa.tasks.manager import (
    Task,
    TaskManager,
)
from osa.tools import ToolResult


class PlanCreator(Protocol):
    """Protocol for components that create validated plans."""

    def create_plan(
        self,
        goal: str,
        *,
        context: str | None = None,
    ) -> Plan:
        """Create a plan for a user goal."""
        ...


class ActionResolver(Protocol):
    """Protocol for resolving a task into a tool action."""

    def resolve(
        self,
        task: Task,
        outputs: Mapping[str, str],
    ) -> TaskAction:
        """Resolve a task into one tool action."""
        ...


ToolExecutor = Callable[
    [str, Mapping[str, object]],
    ToolResult,
]


class TaskRunnerError(RuntimeError):
    """Raised when planned task execution cannot continue."""


class PlannedTaskRunner:
    """Connect planning, task lifecycle, action resolution, and tools."""

    def __init__(
        self,
        planner: PlanCreator,
        resolver: ActionResolver,
        manager: TaskManager,
        executor: TaskExecutor,
        tool_executor: ToolExecutor,
        verifier: TaskVerifier,
    ) -> None:
        self._planner = planner
        self._resolver = resolver
        self._manager = manager
        self._executor = executor
        self._tool_executor = tool_executor
        self._verifier = verifier

    @property
    def planner(self) -> PlanCreator:
        """Return the plan creator."""
        return self._planner

    @property
    def resolver(self) -> ActionResolver:
        """Return the task action resolver."""
        return self._resolver

    @property
    def manager(self) -> TaskManager:
        """Return the task manager."""
        return self._manager

    @property
    def verifier(self) -> TaskVerifier:
        """Return the task verifier."""
        return self._verifier

    @property
    def executor(self) -> TaskExecutor:
        """Return the task executor."""
        return self._executor

    def run(
        self,
        goal: str,
        *,
        context: str | None = None,
    ) -> TaskExecutionReport:
        """Create and execute a complete plan for the supplied goal."""
        normalized_goal = goal.strip()

        if not normalized_goal:
            raise ValueError(
                "goal cannot be empty."
            )

        try:
            plan = self._planner.create_plan(
                normalized_goal,
                context=context,
            )
        except Exception as exc:
            raise TaskRunnerError(
                f"Could not create task plan: {exc}"
            ) from exc

        run = self._manager.create_run(
            plan
        )

        def handle_task(
            task: Task,
            outputs: Mapping[str, str],
        ) -> str:
            try:
                action = self._resolver.resolve(
                    task,
                    outputs,
                )

                result = self._tool_executor(
                    action.tool_name,
                    action.arguments,
                )

            except Exception:
                raise

            if not result.success:
                raise TaskRunnerError(
                    result.error
                    or (
                        f"Tool '{action.tool_name}' "
                        "reported failure."
                    )
                )

            return result.output

        return self._executor.execute(
            run.run_id,
            handle_task,
        )



def create_default_planned_task_runner(
    model,
    tool_registry,
    tool_executor,
    filesystem=None,
    max_steps: int = 8,
):
    from osa.tasks.recovery import TaskRecoveryPolicy
    from osa.tasks.verification_rules import create_default_task_verifier

    planner = Planner(model, max_steps=max_steps)
    resolver = TaskActionResolver(model, tool_registry)
    manager = TaskManager()
    recovery = TaskRecoveryPolicy(max_retries=2)
    executor = TaskExecutor(manager, recovery=recovery)
    verifier = create_default_task_verifier(filesystem=filesystem)

    return PlannedTaskRunner(
        planner=planner,
        resolver=resolver,
        manager=manager,
        executor=executor,
        tool_executor=tool_executor,
        verifier=verifier,
    )
