from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.atmospheric_radiation_manager import LayeredAtmosphereSolver
from dialogs.window_controls import configure_resizable_dialog


class OpticalDepthCorrectionDialog(QDialog):
    """编辑按波长区间施加的总气体光学厚度倍率。"""

    DEFAULT_BANDS = (
        {"wavelength_min_um": 9.0, "wavelength_max_um": 10.0, "factor": 1.0},
        {"wavelength_min_um": 14.0, "wavelength_max_um": 15.2, "factor": 1.0},
    )

    def __init__(self, corrections: list[dict[str, Any]] | None = None, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("分波段总光学厚度修正")
        configure_resizable_dialog(self)
        self.resize(690, 430)
        self.setMinimumSize(560, 340)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "修正倍率逐层作用于所选波长区间内的气体总光学厚度：倍率 > 1 增强吸收，"
            "倍率 < 1 减弱吸收。原始光学厚度文件不会被覆盖。"
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            "起始波长 (μm)", "结束波长 (μm)", "光学厚度倍率",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        row_buttons = QHBoxLayout()
        add_button = QPushButton("新增波段")
        add_button.clicked.connect(lambda: self._append_row(3.0, 5.0, 1.0))
        remove_button = QPushButton("删除选中波段")
        remove_button.clicked.connect(self._remove_selected_row)
        row_buttons.addWidget(add_button)
        row_buttons.addWidget(remove_button)
        row_buttons.addStretch(1)
        layout.addLayout(row_buttons)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用并保存到项目")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        initial = corrections if corrections else list(self.DEFAULT_BANDS)
        for correction in initial:
            self._append_row(
                float(correction.get("wavelength_min_um", 3.0)),
                float(correction.get("wavelength_max_um", 5.0)),
                float(correction.get("factor", 1.0)),
            )

    def _append_row(self, wavelength_min: float, wavelength_max: float, factor: float) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((wavelength_min, wavelength_max, factor)):
            self.table.setItem(row, column, QTableWidgetItem(f"{value:.8g}"))

    def _remove_selected_row(self) -> None:
        row = self.table.currentRow()
        if row < 0 and self.table.rowCount():
            row = self.table.rowCount() - 1
        if row >= 0:
            self.table.removeRow(row)

    def corrections(self) -> list[dict[str, float]]:
        values: list[dict[str, float]] = []
        for row in range(self.table.rowCount()):
            try:
                wavelength_min = float(self.table.item(row, 0).text())
                wavelength_max = float(self.table.item(row, 1).text())
                factor = float(self.table.item(row, 2).text())
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"第{row + 1}行包含无效数值。") from exc
            values.append({
                "wavelength_min_um": wavelength_min,
                "wavelength_max_um": wavelength_max,
                "factor": factor,
            })
        if not values:
            raise ValueError("请至少保留一个光学厚度修正波段。")
        return LayeredAtmosphereSolver.normalize_optical_depth_corrections(values)

    def accept(self) -> None:
        try:
            self.corrections()
        except ValueError as exc:
            QMessageBox.warning(self, "修正参数无效", str(exc))
            return
        super().accept()
