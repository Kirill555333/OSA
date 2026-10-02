"""Tool infrastructure for OSA."""

from osa.tools.browser import BrowserFetchTool
from osa.tools.calculator import CalculatorTool
from osa.tools.desktop import (
    ClickMouseTool,
    CloseApplicationTool,
    HotkeyTool,
    InspectScreenTool,
    LaunchApplicationTool,
    OpenUrlTool,
    PressKeyTool,
    ScrollMouseTool,
    SetSystemVolumeTool,
    TakeScreenshotTool,
    TypeTextTool,
)
from osa.tools.filesystem import (
    FileExistsTool,
    FindFilesTool,
    ListDirectoryTool,
    PatchFileTool,
    ReadFileTool,
    SafeFilesystem,
    WriteFileTool,
)
from osa.tools.memory import (
    ForgetTool,
    RecallTool,
    RememberTool,
)
from osa.tools.registry import (
    ToolError,
    ToolInterface,
    ToolRegistry,
    ToolResult,
)
from osa.tools.research import WebResearchTool
from osa.tools.shell import (
    SafeShell,
    SafeShellError,
    TerminalExecuteTool,
)
from osa.tools.system import SystemInfoTool

__all__ = [
    "BrowserFetchTool",
    "CalculatorTool",
    "ClickMouseTool",
    "CloseApplicationTool",
    "FileExistsTool",
    "FindFilesTool",
    "ForgetTool",
    "HotkeyTool",
    "InspectScreenTool",
    "LaunchApplicationTool",
    "ListDirectoryTool",
    "OpenUrlTool",
    "PatchFileTool",
    "PressKeyTool",
    "ReadFileTool",
    "RecallTool",
    "RememberTool",
    "SafeFilesystem",
    "SafeShell",
    "SafeShellError",
    "ScrollMouseTool",
    "SetSystemVolumeTool",
    "SystemInfoTool",
    "TakeScreenshotTool",
    "TerminalExecuteTool",
    "ToolError",
    "ToolInterface",
    "ToolRegistry",
    "ToolResult",
    "TypeTextTool",
    "WebResearchTool",
    "WriteFileTool",
]
