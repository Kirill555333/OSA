"""GUI entrypoint for OSA. Run with: python -m osa.gui"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from osa.gui.window import JarvisMainWindow
    from osa.main import create_agent
    from osa.memory import LongTermMemory

    workspace_dir = Path.cwd()
    data_dir = workspace_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    memory_storage = LongTermMemory(data_dir / "osa-memory.db")
    agent = create_agent(memory_storage)

    app = QApplication(sys.argv)
    window = JarvisMainWindow(agent)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
