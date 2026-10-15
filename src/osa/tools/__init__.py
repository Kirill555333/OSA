"""Tool infrastructure for OSA."""

from osa.tools.browser import BrowserFetchTool
from osa.tools.calculator import CalculatorTool
from osa.tools.desktop import (
    ClickMouseTool,
    HotkeyTool,
    InspectScreenTool,
    OpenUrlTool,
    PressKeyTool,
    ScrollMouseTool,
    SetSystemVolumeTool,
    TakeScreenshotTool,
    TypeTextTool,
)
from osa.tools.desktop_apps import (
    CloseApplicationTool,
    FocusWindowTool,
    LaunchApplicationTool,
    ListApplicationsTool,
    ListWindowsTool,
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
    "FocusWindowTool",
    "ForgetTool",
    "HotkeyTool",
    "InspectScreenTool",
    "LaunchApplicationTool",
    "ListApplicationsTool",
    "ListDirectoryTool",
    "ListWindowsTool",
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
