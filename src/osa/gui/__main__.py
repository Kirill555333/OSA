"""GUI entrypoint for OSA. Run with: python -m osa.gui"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from osa.gui.window import JarvisMainWindow
    from osa.main import create_agent
    from osa.memory import LongTermMemory
    from osa.utils.paths import resolve_osa_home

    osa_home = resolve_osa_home()
    data_dir = osa_home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    memory_storage = LongTermMemory(data_dir / "osa-memory.db")
    agent = create_agent(memory_storage)

    app = QApplication(sys.argv)
    window = JarvisMainWindow(agent)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
