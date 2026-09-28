from __future__ import annotations

from pathlib import Path

from osa.planning import Plan, PlanStep
from osa.tasks import (
    TaskAction,
    TaskManager,
    VerificationResult,
    create_default_task_verifier,
)
from osa.tools import SafeFilesystem, ToolResult


def create_task():
    manager = TaskManager()

    run = manager.create_run(
        Plan(
            goal="Verify tool",
            steps=(
                PlanStep(
                    step_id="one",
                    description="Execute one tool",
                ),
            ),
        )
    )

    return run.tasks["one"]


def test_default_verifier_checks_browser_result() -> None:
    verifier = create_default_task_verifier()
    task = create_task()

    result = verifier.verify(
        task,
        TaskAction(
            tool_name="browser_fetch",
            arguments={
                "url": "https://example.com",
            },
        ),
        ToolResult(
            success=True,
            output="Example page",
            metadata={
                "status_code": 200,
                "title": "Example",
            },
        ),
        {},
    )

    assert result.verified is True


def test_default_verifier_rejects_bad_browser_status() -> None:
    verifier = create_default_task_verifier()
    task = create_task()

    result = verifier.verify(
        task,
        TaskAction(
            tool_name="browser_fetch",
            arguments={
                "url": "https://example.com",
            },
        ),
        ToolResult(
            success=True,
            output="Not found",
            metadata={
                "status_code": 404,
            },
        ),
        {},
    )

    assert result.verified is False
    assert "HTTP 404" in result.message


def test_default_verifier_checks_written_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    filesystem = SafeFilesystem(
        workspace
    )

    filesystem.write_file(
        "report.txt",
        "OSA report",
    )

    verifier = create_default_task_verifier(
        filesystem
    )

    result = verifier.verify(
        create_task(),
        TaskAction(
            tool_name="write_file",
            arguments={
                "path": "report.txt",
                "content": "OSA report",
            },
        ),
        ToolResult(
            success=True,
            output="File written.",
        ),
        {},
    )

    assert result.verified is True


def test_default_verifier_rejects_wrong_file_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    filesystem = SafeFilesystem(
        workspace
    )

    filesystem.write_file(
        "report.txt",
        "different",
    )

    verifier = create_default_task_verifier(
        filesystem
    )

    result = verifier.verify(
        create_task(),
        TaskAction(
            tool_name="write_file",
            arguments={
                "path": "report.txt",
                "content": "OSA report",
            },
        ),
        ToolResult(
            success=True,
            output="File written.",
        ),
        {},
    )

    assert result.verified is False
    assert "do not match" in result.message


def test_default_verifier_checks_read_only_tool_output() -> None:
    verifier = create_default_task_verifier()

    result = verifier.verify(
        create_task(),
        TaskAction(
            tool_name="calculator",
            arguments={
                "expression": "2 + 2",
            },
        ),
        ToolResult(
            success=True,
            output="4",
        ),
        {},
    )

    assert result.verified is True


def test_default_verifier_rejects_empty_read_only_output() -> None:
    verifier = create_default_task_verifier()

    result = verifier.verify(
        create_task(),
        TaskAction(
            tool_name="calculator",
            arguments={},
        ),
        ToolResult(
            success=True,
            output="",
        ),
        {},
    )

    assert result.verified is False
    assert "no output" in result.message
