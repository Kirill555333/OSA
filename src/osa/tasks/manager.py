"""Task management for OSA plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping
from uuid import uuid4

from osa.planning import Plan


class TaskManagerError(RuntimeError):
    """Base exception raised by the task manager."""


class TaskStatus(StrEnum):
    """Lifecycle state of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Task:
    """One executable unit derived from a plan step."""

    task_id: str
    description: str
    depends_on: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.PENDING
    error: str | None = None


@dataclass(slots=True)
class TaskRun:
    """An in-memory execution state for one plan."""

    run_id: str
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)


class TaskManager:
    """Track task lifecycle and dependency readiness."""

    def __init__(self) -> None:
        self._runs: dict[str, TaskRun] = {}

    def create_run(
        self,
        plan: Plan,
    ) -> TaskRun:
        """Create an in-memory task run from a validated plan."""
        run_id = str(uuid4())

        tasks = {
            step.step_id: Task(
                task_id=step.step_id,
                description=step.description,
                depends_on=step.depends_on,
            )
            for step in plan.steps
        }

        run = TaskRun(
            run_id=run_id,
            goal=plan.goal,
            tasks=tasks,
        )

        self._runs[run_id] = run

        return run

    def get_run(
        self,
        run_id: str,
    ) -> TaskRun:
        """Return a task run by ID."""
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise TaskManagerError(
                f"Task run '{run_id}' does not exist."
            ) from exc

    def ready_tasks(
        self,
        run_id: str,
    ) -> tuple[Task, ...]:
        """Return pending tasks whose dependencies are completed."""
        run = self.get_run(run_id)

        completed = {
            task.task_id
            for task in run.tasks.values()
            if task.status == TaskStatus.COMPLETED
        }

        ready = [
            task
            for task in run.tasks.values()
            if task.status == TaskStatus.PENDING
            and all(
                dependency in completed
                for dependency in task.depends_on
            )
        ]

        return tuple(ready)

    def blocked_tasks(
        self,
        run_id: str,
    ) -> tuple[Task, ...]:
        """Return pending tasks whose dependencies are not complete."""
        run = self.get_run(run_id)

        completed = {
            task.task_id
            for task in run.tasks.values()
            if task.status == TaskStatus.COMPLETED
        }

        blocked = [
            task
            for task in run.tasks.values()
            if task.status == TaskStatus.PENDING
            and not all(
                dependency in completed
                for dependency in task.depends_on
            )
        ]

        return tuple(blocked)

    def start_task(
        self,
        run_id: str,
        task_id: str,
    ) -> Task:
        """Mark a ready task as running."""
        task = self._get_task(
            run_id,
            task_id,
        )

        if task.status != TaskStatus.PENDING:
            raise TaskManagerError(
                f"Task '{task_id}' cannot start from "
                f"status '{task.status.value}'."
            )

        if not self._dependencies_completed(
            run_id,
            task,
        ):
            raise TaskManagerError(
                f"Task '{task_id}' has incomplete dependencies."
            )

        task.status = TaskStatus.RUNNING
        task.error = None

        return task

    def complete_task(
        self,
        run_id: str,
        task_id: str,
    ) -> Task:
        """Mark a running task as completed."""
        task = self._get_task(
            run_id,
            task_id,
        )

        if task.status != TaskStatus.RUNNING:
            raise TaskManagerError(
                f"Task '{task_id}' cannot complete from "
                f"status '{task.status.value}'."
            )

        task.status = TaskStatus.COMPLETED
        task.error = None

        return task

    def fail_task(
        self,
        run_id: str,
        task_id: str,
        error: str,
    ) -> Task:
        """Mark a running task as failed."""
        task = self._get_task(
            run_id,
            task_id,
        )

        normalized_error = error.strip()

        if not normalized_error:
            raise ValueError(
                "error cannot be empty."
            )

        if task.status != TaskStatus.RUNNING:
            raise TaskManagerError(
                f"Task '{task_id}' cannot fail from "
                f"status '{task.status.value}'."
            )

        task.status = TaskStatus.FAILED
        task.error = normalized_error

        return task

    def retry_task(
        self,
        run_id: str,
        task_id: str,
    ) -> Task:
        """Reset a failed task to pending."""
        task = self._get_task(
            run_id,
            task_id,
        )

        if task.status != TaskStatus.FAILED:
            raise TaskManagerError(
                f"Task '{task_id}' cannot be retried from "
                f"status '{task.status.value}'."
            )

        task.status = TaskStatus.PENDING
        task.error = None

        return task

    def cancel_task(
        self,
        run_id: str,
        task_id: str,
    ) -> Task:
        """Cancel a pending or running task."""
        task = self._get_task(
            run_id,
            task_id,
        )

        if task.status not in {
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
        }:
            raise TaskManagerError(
                f"Task '{task_id}' cannot be cancelled from "
                f"status '{task.status.value}'."
            )

        task.status = TaskStatus.CANCELLED
        task.error = None

        return task

    def is_complete(
        self,
        run_id: str,
    ) -> bool:
        """Return whether every task completed successfully."""
        run = self.get_run(run_id)

        return all(
            task.status == TaskStatus.COMPLETED
            for task in run.tasks.values()
        )

    def has_failed(
        self,
        run_id: str,
    ) -> bool:
        """Return whether any task has failed."""
        run = self.get_run(run_id)

        return any(
            task.status == TaskStatus.FAILED
            for task in run.tasks.values()
        )

    def progress(
        self,
        run_id: str,
    ) -> tuple[int, int]:
        """Return completed-task count and total-task count."""
        run = self.get_run(run_id)

        completed = sum(
            task.status == TaskStatus.COMPLETED
            for task in run.tasks.values()
        )

        return (
            completed,
            len(run.tasks),
        )

    def snapshot(
        self,
        run_id: str,
    ) -> Mapping[str, Task]:
        """Return a read-only-style mapping view of current tasks."""
        run = self.get_run(run_id)

        return dict(run.tasks)

    def _get_task(
        self,
        run_id: str,
        task_id: str,
    ) -> Task:
        """Return a task or raise a clear manager error."""
        run = self.get_run(run_id)

        try:
            return run.tasks[task_id]
        except KeyError as exc:
            raise TaskManagerError(
                f"Task '{task_id}' does not exist in run "
                f"'{run_id}'."
            ) from exc

    def _dependencies_completed(
        self,
        run_id: str,
        task: Task,
    ) -> bool:
        """Check whether all dependencies are completed."""
        run = self.get_run(run_id)

        return all(
            run.tasks[dependency].status
            == TaskStatus.COMPLETED
            for dependency in task.depends_on
        )
