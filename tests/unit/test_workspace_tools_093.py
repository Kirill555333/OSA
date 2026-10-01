"""Unit tests for SafeShell, TerminalExecuteTool, and advanced filesystem tools (0.9.3)."""

from __future__ import annotations

from pathlib import Path
import pytest

from osa.tools import (
    FindFilesTool,
    PatchFileTool,
    SafeFilesystem,
    SafeShell,
    SafeShellError,
    TerminalExecuteTool,
)


def test_safe_shell_execution_and_stdout(tmp_path: Path) -> None:
    shell = SafeShell(tmp_path)
    res = shell.execute("echo 'OSA Terminal Live'")

    assert res.returncode == 0
    assert "OSA Terminal Live" in res.stdout
    assert res.stderr == ""
    assert res.duration_ms >= 0


def test_safe_shell_non_zero_exit_code(tmp_path: Path) -> None:
    shell = SafeShell(tmp_path)
    res = shell.execute("exit 42")

    assert res.returncode == 42


def test_safe_shell_blocked_dangerous_commands(tmp_path: Path) -> None:
    shell = SafeShell(tmp_path)

    dangerous = (
        "rm -rf /",
        "rm -fr /",
        "rm -rf ~",
        "mkfs.ext4 /dev/sda",
        ":(){ :|:& };:",
        "shutdown -h now",
        "reboot",
    )

    for cmd in dangerous:
        with pytest.raises(SafeShellError, match="Command blocked for system safety"):
            shell.execute(cmd)


def test_safe_shell_cwd_resolution_and_escape_protection(tmp_path: Path) -> None:
    sub = tmp_path / "subdir"
    sub.mkdir()

    shell = SafeShell(tmp_path)
    res = shell.execute("pwd", working_directory="subdir")
    assert "subdir" in res.stdout

    # Escape attempt
    with pytest.raises(SafeShellError, match="outside workspace root"):
        shell.execute("pwd", working_directory="../")


def test_terminal_execute_tool(tmp_path: Path) -> None:
    shell = SafeShell(tmp_path)
    tool = TerminalExecuteTool(shell)

    res_ok = tool.execute({"command": "echo 'Hello from tool'"})
    assert res_ok.success is True
    assert "Hello from tool" in res_ok.output
    assert res_ok.metadata["returncode"] == 0

    res_blocked = tool.execute({"command": "rm -rf /"})
    assert res_blocked.success is False
    assert "Command blocked" in (res_blocked.error or "")


def test_find_files_tool(tmp_path: Path) -> None:
    fs = SafeFilesystem(tmp_path)
    tool = FindFilesTool(fs)

    (tmp_path / "module_a.py").write_text("print(1)")
    (tmp_path / "module_b.py").write_text("print(2)")
    (tmp_path / "notes.txt").write_text("hello")

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("print(3)")

    res_py = tool.execute({"pattern": "*.py"})
    assert res_py.success is True
    assert "module_a.py" in res_py.output
    assert "module_b.py" in res_py.output
    assert "notes.txt" not in res_py.output

    res_none = tool.execute({"pattern": "*.json"})
    assert res_none.success is True
    assert "No matching files" in res_none.output


def test_patch_file_tool(tmp_path: Path) -> None:
    fs = SafeFilesystem(tmp_path)
    tool = PatchFileTool(fs)

    file_p = tmp_path / "config.ini"
    file_p.write_text("port=8080\nhost=localhost\n")

    res = tool.execute({
        "path": "config.ini",
        "target": "port=8080",
        "replacement": "port=9090",
    })

    assert res.success is True
    assert "successfully patched" in res.output
    assert file_p.read_text() == "port=9090\nhost=localhost\n"

    # Missing target
    res_miss = tool.execute({
        "path": "config.ini",
        "target": "port=1234",
        "replacement": "port=9999",
    })
    assert res_miss.success is False
    assert "Target text was not found" in (res_miss.error or "")
