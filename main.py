from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.main_window import AtmosphereMainWindow


def load_stylesheet(app: QApplication) -> None:
    resource_root = Path(__file__).resolve().parent / "resources"
    path = resource_root / "styles" / "fluent.qss"
    if path.exists():
        stylesheet = path.read_text(encoding="utf-8")
        stylesheet = stylesheet.replace(
            "__ICON_DIR__", (resource_root / "icons").as_posix()
        )
        app.setStyleSheet(stylesheet)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ARTE Atmosphere")
    app.setOrganizationName("ARTE Solver")
    load_stylesheet(app)
    window = AtmosphereMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
