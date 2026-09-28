"""Built-in verification rules for OSA tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osa.tasks.action import TaskAction
from osa.tasks.manager import Task
from osa.tasks.verification import (
    TaskVerifier,
    VerificationResult,
)
from osa.tools import SafeFilesystem, ToolResult


def _require_output(
    tool_name: str,
):
    """Build a verifier requiring a successful non-empty result."""

    def verify(
        task: Task,
        action: TaskAction,
        result: ToolResult,
        outputs: Mapping[str, str],
    ) -> VerificationResult:
        if not result.success:
            return VerificationResult(
                verified=False,
                message=(
                    result.error
                    or f"Tool '{tool_name}' reported failure."
                ),
            )

        if not result.output.strip():
            return VerificationResult(
                verified=False,
                message=(
                    f"Tool '{tool_name}' returned no output."
                ),
            )

        return VerificationResult(
            verified=True,
            message=(
                f"Tool '{tool_name}' returned a valid result."
            ),
        )

    return verify


def create_default_task_verifier(
    filesystem: SafeFilesystem | None = None,
) -> TaskVerifier:
    """Create a verifier containing OSA's built-in safe-tool rules."""
    verifier = TaskVerifier()

    verifier.register(
        "calculator",
        _require_output("calculator"),
    )
    verifier.register(
        "list_directory",
        _require_output("list_directory"),
    )
    verifier.register(
        "read_file",
        _require_output("read_file"),
    )
    verifier.register(
        "file_exists",
        _require_output("file_exists"),
    )
    verifier.register(
        "system_info",
        _require_output("system_info"),
    )
    verifier.register(
        "remember",
        _require_output("remember"),
    )
    verifier.register(
        "recall",
        _require_output("recall"),
    )

    verifier.register(
        "browser_fetch",
        _verify_browser_fetch,
    )

    if filesystem is not None:
        verifier.register(
            "write_file",
            _make_write_file_verifier(filesystem),
        )

    return verifier


def _verify_browser_fetch(
    task: Task,
    action: TaskAction,
    result: ToolResult,
    outputs: Mapping[str, str],
) -> VerificationResult:
    """Verify that browser_fetch returned a successful HTTP result."""
    if not result.success:
        return VerificationResult(
            verified=False,
            message=(
                result.error
                or "browser_fetch reported failure."
            ),
        )

    status_code = result.metadata.get(
        "status_code"
    )

    if not isinstance(status_code, int):
        return VerificationResult(
            verified=False,
            message=(
                "browser_fetch did not provide a valid HTTP status."
            ),
        )

    if not 200 <= status_code < 300:
        return VerificationResult(
            verified=False,
            message=(
                f"browser_fetch returned HTTP {status_code}."
            ),
        )

    if not result.output.strip():
        return VerificationResult(
            verified=False,
            message=(
                "browser_fetch returned no page content."
            ),
        )

    return VerificationResult(
        verified=True,
        message=(
            f"Web page verified with HTTP {status_code}."
        ),
    )


def _make_write_file_verifier(
    filesystem: SafeFilesystem,
):
    """Create a verifier that checks the written file contents."""

    def verify(
        task: Task,
        action: TaskAction,
        result: ToolResult,
        outputs: Mapping[str, str],
    ) -> VerificationResult:
        if not result.success:
            return VerificationResult(
                verified=False,
                message=(
                    result.error
                    or "write_file reported failure."
                ),
            )

        path = action.arguments.get(
            "path"
        )
        expected_content = action.arguments.get(
            "content"
        )

        if not isinstance(path, str):
            return VerificationResult(
                verified=False,
                message=(
                    "write_file action did not provide a valid path."
                ),
            )

        if not isinstance(expected_content, str):
            return VerificationResult(
                verified=False,
                message=(
                    "write_file action did not provide valid content."
                ),
            )

        try:
            actual_content = filesystem.read_file(
                path,
                max_characters=max(
                    1,
                    len(expected_content) + 1,
                ),
            )
        except Exception as exc:
            return VerificationResult(
                verified=False,
                message=(
                    f"Written file could not be verified: {exc}"
                ),
            )

        if actual_content != expected_content:
            return VerificationResult(
                verified=False,
                message=(
                    "Written file contents do not match "
                    "the requested content."
                ),
            )

        return VerificationResult(
            verified=True,
            message=(
                f"File '{path}' was written and verified."
            ),
        )

    return verify
