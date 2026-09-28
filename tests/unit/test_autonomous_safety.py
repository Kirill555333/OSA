from __future__ import annotations

from dataclasses import dataclass

from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tasks.autonomous import (
    AutonomousLoop,
    AutonomousTaskResult,
)
from osa.tasks.safety import AutonomousSafetyGate


@dataclass
class Decision:
    level: PermissionLevel


class FakeBackend:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def create_run(self, goal: str) -> str:
        return "run-1"

    def ready_task_ids(self, run_id: str):
        if not self.executed:
            return ("task-1",)
        return ()

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.executed.append(task_id)

        return AutonomousTaskResult(
            task_id=task_id,
            status="completed",
            output="done",
        )

    def is_complete(self, run_id: str) -> bool:
        return bool(self.executed)

    def has_failed(self, run_id: str) -> bool:
        return False


def test_allow_permission_reaches_executor() -> None:
    backend = FakeBackend()

    permissions = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    gate = AutonomousSafetyGate(
        permissions,
        resolver=lambda run_id, task_id: "calculator",
    )

    loop = AutonomousLoop(
        backend,
        authorizer=gate,
    )

    report = loop.run("calculate something")

    assert report.success is True
    assert backend.executed == ["task-1"]


def test_deny_permission_blocks_execution() -> None:
    backend = FakeBackend()

    permissions = PermissionPolicy(
        {
            "shell": PermissionLevel.DENY,
        }
    )

    gate = AutonomousSafetyGate(
        permissions,
        resolver=lambda run_id, task_id: "shell",
    )

    loop = AutonomousLoop(
        backend,
        authorizer=gate,
    )

    report = loop.run("run shell command")

    assert report.success is False
    assert backend.executed == []
    assert report.failed_task_ids == ("task-1",)
    assert report.cycles[0].results[0].error == "permission_denied"


def test_confirmation_without_callback_fails_closed() -> None:
    backend = FakeBackend()

    permissions = PermissionPolicy(
        {
            "write_file": PermissionLevel.CONFIRM,
        }
    )

    gate = AutonomousSafetyGate(
        permissions,
        resolver=lambda run_id, task_id: "write_file",
    )

    loop = AutonomousLoop(
        backend,
        authorizer=gate,
    )

    report = loop.run("write a file")

    assert report.success is False
    assert backend.executed == []
    assert report.cycles[0].results[0].error == "confirmation_required"


def test_confirmation_callback_can_authorize() -> None:
    backend = FakeBackend()
    confirmations: list[tuple[str, str, str]] = []

    permissions = PermissionPolicy(
        {
            "write_file": PermissionLevel.CONFIRM,
        }
    )

    def confirm(
        run_id: str,
        task_id: str,
        tool_name: str,
    ) -> bool:
        confirmations.append((run_id, task_id, tool_name))
        return True

    gate = AutonomousSafetyGate(
        permissions,
        resolver=lambda run_id, task_id: "write_file",
        confirmation=confirm,
    )

    loop = AutonomousLoop(
        backend,
        authorizer=gate,
    )

    report = loop.run("write a file")

    assert report.success is True
    assert backend.executed == ["task-1"]
    assert confirmations == [
        ("run-1", "task-1", "write_file")
    ]


def test_unknown_tool_fails_closed() -> None:
    backend = FakeBackend()

    permissions = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
        }
    )

    gate = AutonomousSafetyGate(
        permissions,
        resolver=lambda run_id, task_id: None,
    )

    loop = AutonomousLoop(
        backend,
        authorizer=gate,
    )

    report = loop.run("do something")

    assert report.success is False
    assert backend.executed == []
    assert report.cycles[0].results[0].error == "tool_resolution_failed"


def test_additional_autonomous_deny_list() -> None:
    backend = FakeBackend()

    permissions = PermissionPolicy(
        {
            "browser_fetch": PermissionLevel.ALLOW,
        }
    )

    gate = AutonomousSafetyGate(
        permissions,
        resolver=lambda run_id, task_id: "browser_fetch",
        additional_denied_tools={"browser_fetch"},
    )

    loop = AutonomousLoop(
        backend,
        authorizer=gate,
    )

    report = loop.run("fetch a page")

    assert report.success is False
    assert backend.executed == []
    assert (
        report.cycles[0].results[0].error
        == "tool_blocked_by_autonomous_policy"
    )
