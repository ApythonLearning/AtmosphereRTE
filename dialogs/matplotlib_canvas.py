from __future__ import annotations

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer


class DebouncedFigureCanvas(FigureCanvasQTAgg):
    """拖动窗口期间合并Matplotlib的连续重绘请求。"""

    def __init__(self, figure: Figure, resize_delay_ms: int = 160) -> None:
        super().__init__(figure)
        self._inside_resize_event = False
        self._resize_redraw_timer = QTimer(self)
        self._resize_redraw_timer.setSingleShot(True)
        self._resize_redraw_timer.setInterval(resize_delay_ms)
        self._resize_redraw_timer.timeout.connect(self._draw_after_resize)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        self._inside_resize_event = True
        try:
            super().resizeEvent(event)
        finally:
            self._inside_resize_event = False
        self._resize_redraw_timer.start()

    def draw_idle(self) -> None:
        if self._inside_resize_event:
            self._resize_redraw_timer.start()
            return
        super().draw_idle()

    def _draw_after_resize(self) -> None:
        super().draw_idle()
