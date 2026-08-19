from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog


def configure_resizable_dialog(dialog: QDialog) -> None:
    """为自定义弹窗统一启用最小化、最大化和F11全屏操作。"""

    dialog.setWindowFlag(Qt.WindowSystemMenuHint, True)
    dialog.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
    dialog.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
    dialog.setSizeGripEnabled(True)

    fullscreen_shortcut = QShortcut(QKeySequence("F11"), dialog)
    fullscreen_shortcut.activated.connect(
        lambda: dialog.showNormal() if dialog.isFullScreen() else dialog.showFullScreen()
    )
    # QShortcut已以dialog为父对象；保留引用也便于界面测试检查此能力。
    dialog._fullscreen_shortcut = fullscreen_shortcut  # type: ignore[attr-defined]
