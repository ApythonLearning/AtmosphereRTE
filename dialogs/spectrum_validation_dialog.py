from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.spectrum_validation_manager import SpectrumValidationManager
from dialogs.matplotlib_canvas import DebouncedFigureCanvas
from dialogs.window_controls import configure_resizable_dialog


PLOT_BACKGROUND = "#eef0f3"
TEXT_COLOR = "#26313d"
SPINE_COLOR = "#697886"


AXIS_KINDS = {
    "自动识别": "auto",
    "波长（μm）": "wavelength_um",
    "波长（nm）": "wavelength_nm",
    "波数（cm⁻¹）": "wavenumber_cm",
}

SCALING_MODES = {
    "绝对值对比（同单位）": "absolute",
    "最小二乘幅值匹配": "least_squares",
    "峰值归一化": "peak",
    "积分面积归一化": "area",
}

METRIC_LABELS = {
    "sample_count": "共同采样点数",
    "overlap_min_um": "重叠波段下限（μm）",
    "overlap_max_um": "重叠波段上限（μm）",
    "scale_factor": "仿真谱幅值系数",
    "rmse": "均方根误差 RMSE",
    "nrmse": "归一化 RMSE",
    "mae": "平均绝对误差 MAE",
    "bias": "平均偏差 Bias",
    "correlation": "皮尔逊相关系数",
    "spectral_angle_deg": "光谱夹角（deg）",
}


class SpectrumValidationDialog(QDialog):
    """仿真探测器光谱与卫星产品光谱的验证工作台。"""

    def __init__(
        self,
        atmospheric_result: dict[str, Any],
        parent: QWidget | None = None,
        correction_handler: Callable[[], bool] | None = None,
        automatic_correction_handler: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.embedded = bool(embedded)
        self.setWindowTitle("大气辐射传输光谱验证")
        if self.embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
            self.setMinimumSize(0, 0)
        else:
            configure_resizable_dialog(self)
            self.resize(1420, 880)
            self.setMinimumSize(1040, 700)
        self.manager = SpectrumValidationManager()
        self.atmospheric_result = dict(atmospheric_result)
        self._correction_handler = correction_handler
        self._automatic_correction_handler = automatic_correction_handler
        self._satellite_series: dict[str, np.ndarray] = {}
        self._comparison_result: dict[str, Any] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._build_source_group())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_plot_widget())
        splitter.addWidget(self._build_metric_widget())
        splitter.setSizes([1060, 300])
        layout.addWidget(splitter, 1)

        command_row = QHBoxLayout()
        self.compare_button = QPushButton("执行光谱验证")
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._compare)
        self.export_button = QPushButton("导出对比 CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        self.correction_button = QPushButton("手动配置光学厚度修正…")
        self.correction_button.setEnabled(False)
        self.correction_button.clicked.connect(self._configure_optical_depth_correction)
        self.automatic_correction_button = QPushButton("一键自动矫正总光学厚度")
        self.automatic_correction_button.setEnabled(False)
        self.automatic_correction_button.clicked.connect(
            self._automatically_correct_optical_depth
        )
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setVisible(not self.embedded)
        command_row.addWidget(self.compare_button)
        command_row.addWidget(self.export_button)
        command_row.addWidget(self.correction_button)
        command_row.addWidget(self.automatic_correction_button)
        command_row.addStretch(1)
        command_row.addWidget(self.close_button)
        layout.addLayout(command_row)

        self._update_simulation_summary()

    def set_atmospheric_result(self, atmospheric_result: dict[str, Any] | None) -> None:
        self.atmospheric_result = dict(atmospheric_result or {})
        self._comparison_result = None
        self.export_button.setEnabled(False)
        self.correction_button.setEnabled(False)
        self.automatic_correction_button.setEnabled(False)
        for row in range(self.metric_table.rowCount()):
            self.metric_table.item(row, 1).setText("-")
        self._draw_placeholder()
        self._update_simulation_summary()

    def _update_simulation_summary(self) -> None:
        try:
            simulated = self.manager.from_atmospheric_detector_result(self.atmospheric_result)
            self.simulation_summary.setText(
                f"探测器接收的大气传输总谱：{len(simulated['wavelength_um'])} 点，"
                f"{simulated['wavelength_um'][0]:.6g}–{simulated['wavelength_um'][-1]:.6g} μm，"
                f"{simulated['value_unit']}"
            )
        except ValueError as exc:
            self.simulation_summary.setText(str(exc))

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("验证数据源与匹配设置")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.simulation_summary = QLabel("正在读取当前仿真光谱…")
        self.simulation_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("仿真光谱", self.simulation_summary)

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.satellite_path = QLineEdit()
        self.satellite_path.setReadOnly(True)
        browse_button = QPushButton("选择产品…")
        browse_button.clicked.connect(self._browse_satellite_product)
        file_layout.addWidget(self.satellite_path, 1)
        file_layout.addWidget(browse_button)
        form.addRow("卫星产品光谱", file_row)

        mapping_row = QWidget()
        mapping_layout = QHBoxLayout(mapping_row)
        mapping_layout.setContentsMargins(0, 0, 0, 0)
        self.axis_combo = QComboBox()
        self.value_combo = QComboBox()
        self.axis_kind_combo = QComboBox()
        self.axis_kind_combo.addItems(AXIS_KINDS)
        mapping_layout.addWidget(QLabel("坐标"))
        mapping_layout.addWidget(self.axis_combo, 2)
        mapping_layout.addWidget(QLabel("光谱值"))
        mapping_layout.addWidget(self.value_combo, 2)
        mapping_layout.addWidget(QLabel("坐标类型"))
        mapping_layout.addWidget(self.axis_kind_combo, 1)
        form.addRow("变量映射", mapping_row)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(SCALING_MODES)
        form.addRow("幅值处理", self.scaling_combo)
        note = QLabel(
            "支持 CSV/TXT/DAT、NPZ、NetCDF 与 HDF5 的一维光谱变量。绝对值对比要求两组光谱值采用相同物理单位；"
            "归一化模式用于只验证谱形。与卫星谱辐亮度产品对比时，请先使用高分辨率大气光谱模式。"
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        form.addRow("说明", note)
        return group

    def _build_plot_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Microsoft YaHei", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        })
        self.figure = Figure(figsize=(10.6, 6.6), dpi=100)
        self.figure.patch.set_facecolor(PLOT_BACKGROUND)
        self.canvas = DebouncedFigureCanvas(self.figure)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setObjectName("plotNavigationToolbar")
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self._draw_placeholder()
        return widget

    def _build_metric_widget(self) -> QWidget:
        group = QGroupBox("定量验证指标")
        layout = QVBoxLayout(group)
        self.metric_table = QTableWidget(len(METRIC_LABELS), 2)
        self.metric_table.setHorizontalHeaderLabels(["指标", "结果"])
        self.metric_table.verticalHeader().setVisible(False)
        self.metric_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metric_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.metric_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for row, label in enumerate(METRIC_LABELS.values()):
            self.metric_table.setItem(row, 0, QTableWidgetItem(label))
            self.metric_table.setItem(row, 1, QTableWidgetItem("-"))
        layout.addWidget(self.metric_table)
        return group

    def _browse_satellite_product(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择真实卫星产品光谱",
            "",
            "光谱产品 (*.csv *.txt *.dat *.npz *.nc *.nc4 *.h5 *.hdf5);;所有文件 (*)",
        )
        if not path:
            return
        try:
            series = self.manager.read_numeric_series(path)
            axis_name, value_name = self.manager.suggest_series(series)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "卫星产品读取失败", str(exc))
            return
        self._satellite_series = series
        self.satellite_path.setText(str(Path(path).resolve()))
        self.axis_combo.clear()
        self.value_combo.clear()
        self.axis_combo.addItems(series)
        self.value_combo.addItems(series)
        self.axis_combo.setCurrentText(axis_name)
        self.value_combo.setCurrentText(value_name)
        self.compare_button.setEnabled(True)

    def _compare(self) -> None:
        try:
            simulated = self.manager.from_atmospheric_detector_result(self.atmospheric_result)
            observed = self.manager.build_spectrum(
                self._satellite_series,
                self.axis_combo.currentText(),
                self.value_combo.currentText(),
                AXIS_KINDS[self.axis_kind_combo.currentText()],
            )
            result = self.manager.compare(
                simulated,
                observed,
                SCALING_MODES[self.scaling_combo.currentText()],
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "光谱验证失败", str(exc))
            return
        self._comparison_result = result
        self._draw_result(result)
        self._update_metrics(result["metrics"])
        self.export_button.setEnabled(True)
        self.correction_button.setEnabled(self._correction_handler is not None)
        self.automatic_correction_button.setEnabled(
            self._automatic_correction_handler is not None
            and SCALING_MODES[self.scaling_combo.currentText()] == "absolute"
        )

    def _configure_optical_depth_correction(self) -> None:
        if self._correction_handler is not None:
            self._correction_handler()

    def _automatically_correct_optical_depth(self) -> None:
        if self._comparison_result is None or self._automatic_correction_handler is None:
            return
        if SCALING_MODES[self.scaling_combo.currentText()] != "absolute":
            QMessageBox.information(
                self,
                "一键自动矫正总光学厚度",
                "自动光学厚度矫正需要使用“绝对值对比（同单位）”。",
            )
            return
        payload = {
            "wavelength_um": np.asarray(
                self._comparison_result["wavelength_um"], dtype=float
            ).copy(),
            "observed": np.asarray(
                self._comparison_result["observed"], dtype=float
            ).copy(),
            "simulated": np.asarray(
                self._comparison_result["simulated"], dtype=float
            ).copy(),
        }
        corrected_result = self._automatic_correction_handler(payload)
        if not corrected_result:
            return
        self.atmospheric_result = dict(corrected_result)
        self._compare()
        corrections = corrected_result.get("optical_depth_corrections", [])
        diagnostics = corrected_result.get("optical_depth_auto_correction", {})
        temperature_offset = (
            float(diagnostics.get("upper_atmosphere_temperature_offset_k", 0.0))
            if isinstance(diagnostics, dict) else 0.0
        )
        rmse_before = diagnostics.get("rmse_before", "-") if isinstance(diagnostics, dict) else "-"
        rmse_after = diagnostics.get("rmse_after", "-") if isinstance(diagnostics, dict) else "-"
        QMessageBox.information(
            self,
            "自动矫正完成",
            "逐波数总光学厚度和高层温度偏移已保存到当前项目。\n"
            f"光学厚度修正单元：{len(corrections)}\n"
            f"10 km以上温度偏移：{temperature_offset:+.1f} K\n"
            f"RMSE：{rmse_before} → {rmse_after}",
        )

    def _draw_placeholder(self) -> None:
        self.figure.clear()
        axis = self.figure.subplots(1, 1)
        self._style_axis(axis)
        axis.text(
            0.5, 0.5, "导入卫星产品并执行光谱验证",
            transform=axis.transAxes, ha="center", va="center", color=SPINE_COLOR, fontsize=14,
            fontfamily="Microsoft YaHei",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        self.canvas.draw_idle()

    def _draw_result(self, result: dict[str, Any]) -> None:
        self.figure.clear()
        overlay, residual = self.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
        wavelength = result["wavelength_um"]
        overlay.plot(wavelength, result["observed"], color="#1672b8", linewidth=1.2, label="Satellite product")
        overlay.plot(wavelength, result["simulated"], color="#d97706", linewidth=1.1, label="ARTE Atmosphere simulation")
        quantity = str(self.atmospheric_result.get("spectral_quantity", "irradiance"))
        overlay.set_ylabel(
            r"Spectral radiance (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)"
            if quantity == "radiance"
            else r"Spectral irradiance (W m$^{-2}$ $\mu$m$^{-1}$)"
        )
        overlay.set_title("Detector-received atmospheric radiative-transfer spectrum validation")
        legend = overlay.legend(loc="best", frameon=False)
        for label in legend.get_texts():
            label.set_color(TEXT_COLOR)
        residual.axhline(0.0, color=SPINE_COLOR, linewidth=0.8)
        residual.plot(wavelength, result["residual"], color="#c43c64", linewidth=0.9)
        residual.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        residual.set_ylabel("Sim − Obs")
        for axis in (overlay, residual):
            self._style_axis(axis)
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

    def _style_axis(self, axis: Any) -> None:
        axis.set_facecolor(PLOT_BACKGROUND)
        axis.tick_params(which="both", direction="in", top=True, right=True, colors=TEXT_COLOR)
        axis.xaxis.label.set_color(TEXT_COLOR)
        axis.yaxis.label.set_color(TEXT_COLOR)
        axis.title.set_color(TEXT_COLOR)
        for spine in axis.spines.values():
            spine.set_color(SPINE_COLOR)

    def _update_metrics(self, metrics: dict[str, Any]) -> None:
        for row, key in enumerate(METRIC_LABELS):
            value = metrics[key]
            text = str(value) if isinstance(value, int) else (f"{float(value):.8g}" if np.isfinite(value) else "N/A")
            self.metric_table.item(row, 1).setText(text)

    def _export(self) -> None:
        if self._comparison_result is None:
            return
        default_path = str(Path(self.satellite_path.text()).with_name("spectrum_validation.csv"))
        path, _ = QFileDialog.getSaveFileName(self, "导出光谱验证结果", default_path, "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            exported = self.manager.export_csv(self._comparison_result, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", f"光谱验证结果已导出：\n{exported}")
