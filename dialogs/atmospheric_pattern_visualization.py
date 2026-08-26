from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.atmospheric_pattern_manager import AtmosphericPatternManager
from dialogs.matplotlib_canvas import DebouncedFigureCanvas


PLOT_BACKGROUND = "#eef0f3"
AXIS_BACKGROUND = "#f7f8fa"
TEXT_COLOR = "#26313d"
SPINE_COLOR = "#697886"


def configure_scientific_plot_style() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
        }
    )


def apply_axis_style(axis: Any, *, grid: bool = False) -> None:
    axis.set_facecolor(AXIS_BACKGROUND)
    axis.tick_params(
        which="both", direction="in", top=True, right=True, colors=TEXT_COLOR
    )
    axis.minorticks_on()
    if grid:
        axis.grid(True, color=SPINE_COLOR, alpha=0.16, linewidth=0.6)
    else:
        axis.grid(False)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.title.set_color(TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(SPINE_COLOR)


def add_panel_label(axis: Any, label: str) -> None:
    axis.text(
        0.03,
        0.96,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontstyle="normal",
        fontweight="bold",
        va="top",
        ha="left",
        color=TEXT_COLOR,
    )


class AtmosphericPatternVisualization(QWidget):
    """Lightweight views of learned atmospheric regimes without raw-field caching."""

    representativeRequested = Signal(int)

    def __init__(
        self, manager: AtmosphericPatternManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        configure_scientific_plot_style()
        self.manager = manager
        self._build_ui()
        self.refresh()

    @staticmethod
    def _figure(width: float, height: float) -> Figure:
        figure = Figure(
            figsize=(width, height), facecolor=PLOT_BACKGROUND, constrained_layout=True
        )
        return figure

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_overview_tab(), "训练概览")
        self.tabs.addTab(self._build_map_tab(), "全球模式分布")
        self.tabs.addTab(self._build_profile_tab(), "代表性廓线")
        self.tabs.addTab(self._build_feature_tab(), "特征空间")
        self.tabs.setMinimumHeight(540)
        layout.addWidget(self.tabs)

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.overview_text = QLabel("尚无可视化训练结果。")
        self.overview_text.setWordWrap(True)
        self.overview_text.setObjectName("secondaryText")
        layout.addWidget(self.overview_text)
        self.overview_figure = self._figure(10.0, 4.2)
        self.overview_canvas = DebouncedFigureCanvas(self.overview_figure)
        self.overview_canvas.setMinimumHeight(390)
        layout.addWidget(self.overview_canvas, 1)
        return page

    def _build_map_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("时间"))
        self.map_time = QComboBox()
        self.map_time.currentIndexChanged.connect(self._draw_map)
        controls.addWidget(self.map_time)
        controls.addWidget(QLabel("模式"))
        self.map_pattern = QComboBox()
        self.map_pattern.currentIndexChanged.connect(self._draw_map)
        controls.addWidget(self.map_pattern)
        controls.addStretch(1)
        self.map_status = QLabel()
        controls.addWidget(self.map_status)
        layout.addLayout(controls)
        self.map_figure = self._figure(10.0, 4.8)
        self.map_canvas = DebouncedFigureCanvas(self.map_figure)
        self.map_canvas.setMinimumHeight(430)
        layout.addWidget(self.map_canvas, 1)
        return page

    def _build_profile_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "可按住 Ctrl 选择多个模式叠加。实线为代表廓线，半透明区域为该模式内样本的10%—90%区间。"
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        layout.addWidget(note)
        splitter = QSplitter()
        selector = QWidget()
        selector_layout = QVBoxLayout(selector)
        selector_layout.setContentsMargins(0, 0, 6, 0)
        selector_layout.addWidget(QLabel("最终代表模式"))
        self.profile_modes = QListWidget()
        self.profile_modes.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.profile_modes.itemSelectionChanged.connect(self._draw_profiles)
        selector_layout.addWidget(self.profile_modes, 1)
        self.profile_hapi_button = QPushButton("在HAPI中打开选中模式")
        self.profile_hapi_button.setObjectName("primaryButton")
        self.profile_hapi_button.clicked.connect(self._request_selected_profile)
        selector_layout.addWidget(self.profile_hapi_button)
        splitter.addWidget(selector)
        chart = QWidget()
        chart_layout = QVBoxLayout(chart)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_figure = self._figure(10.5, 4.8)
        self.profile_canvas = DebouncedFigureCanvas(self.profile_figure)
        chart_layout.addWidget(self.profile_canvas)
        splitter.addWidget(chart)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 900])
        layout.addWidget(splitter, 1)
        return page

    def _build_feature_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.feature_text = QLabel(
            "散点表示训练样本，星形表示最终代表模式。颜色与全球模式分布图一致。"
        )
        self.feature_text.setWordWrap(True)
        self.feature_text.setObjectName("secondaryText")
        layout.addWidget(self.feature_text)
        self.feature_figure = self._figure(10.0, 4.8)
        self.feature_canvas = DebouncedFigureCanvas(self.feature_figure)
        self.feature_canvas.setMinimumHeight(430)
        layout.addWidget(self.feature_canvas, 1)
        return page

    def refresh(self) -> None:
        self._refresh_controls()
        self._draw_overview()
        self._draw_map()
        self._draw_profiles()
        self._draw_feature_space()

    def set_selected_mode(self, mode_index: int) -> None:
        if 0 <= mode_index < self.profile_modes.count():
            self.profile_modes.clearSelection()
            self.profile_modes.item(mode_index).setSelected(True)

    def _refresh_controls(self) -> None:
        self.map_time.blockSignals(True)
        self.map_time.clear()
        self.map_time.addItem("全部时段", "")
        if self.manager.training_times_utc.size:
            months = sorted(
                {
                    str(value)[:7]
                    for value in self.manager.training_times_utc
                    if len(str(value)) >= 7
                }
            )
            for month in months:
                self.map_time.addItem(month, month)
        self.map_time.blockSignals(False)

        self.map_pattern.blockSignals(True)
        self.map_pattern.clear()
        self.map_pattern.addItem("全部模式", -1)
        for index in range(len(self.manager.representative_metadata)):
            self.map_pattern.addItem(f"模式 {index + 1}", index)
        self.map_pattern.blockSignals(False)

        self.profile_modes.blockSignals(True)
        self.profile_modes.clear()
        counts = self.manager.cluster_counts
        for index in range(len(self.manager.representative_metadata)):
            count = int(counts[index]) if index < counts.size else 0
            item = QListWidgetItem(f"模式 {index + 1}  ·  {count:,} 条样本")
            item.setData(256, index)
            self.profile_modes.addItem(item)
        if self.profile_modes.count():
            self.profile_modes.item(0).setSelected(True)
        self.profile_modes.blockSignals(False)
        self.profile_hapi_button.setEnabled(self.profile_modes.count() > 0)

    def _draw_overview(self) -> None:
        figure = self.overview_figure
        figure.clear()
        if not self.manager.is_fitted:
            self.overview_text.setText("尚未训练或读取模型。")
            self._empty_figure(figure, self.overview_canvas, "No trained model")
            return
        summary = self.manager.summary()
        method = "Autoencoder + K-means" if self.manager.method == "autoencoder" else "PCA/EOF + K-means"
        self.overview_text.setText(
            f"{method}；样本 {summary['training_sample_count']:,} 条；"
            f"最终模式 {summary['representative_count']} 个；35层特征 {summary['feature_count']} 组；"
            f"重建RMSE {summary['reconstruction_rmse']:.4g}；"
            f"含经纬度样本 {summary['geolocated_sample_count']:,} 条。训练阶段未调用HAPI。"
        )
        if not self.manager.has_visualization_data:
            self._empty_figure(
                figure,
                self.overview_canvas,
                "This legacy model has no saved visualization data",
            )
            return
        axes = figure.subplots(1, 2)
        pattern_indices = np.arange(1, self.manager.cluster_counts.size + 1)
        axes[0].bar(
            pattern_indices,
            self.manager.cluster_counts,
            color=colormaps["turbo"].resampled(max(len(pattern_indices), 2))(
                np.arange(len(pattern_indices))
            ),
            width=0.82,
        )
        axes[0].set_xlabel("Pattern index")
        axes[0].set_ylabel("Sample count")
        axes[0].set_title("Pattern population")
        if pattern_indices.size <= 24:
            axes[0].set_xticks(pattern_indices)
        add_panel_label(axes[0], "(a)")
        apply_axis_style(axes[0])

        if self.manager.training_loss_history.size:
            iterations = np.arange(1, self.manager.training_loss_history.size + 1)
            axes[1].semilogy(
                iterations, self.manager.training_loss_history, color="#1f6f8b", lw=1.5
            )
            axes[1].set_xlabel("Training epoch")
            axes[1].set_ylabel("Mean squared loss")
            axes[1].set_title("Autoencoder convergence")
        else:
            ratio = self.manager.pca_explained_variance_ratio
            count = min(20, ratio.size)
            axes[1].plot(
                np.arange(1, count + 1),
                np.cumsum(ratio[:count]),
                color="#1f6f8b",
                marker="o",
                ms=3.5,
                lw=1.4,
            )
            axes[1].set_ylim(0.0, 1.02)
            axes[1].set_xlabel("Number of principal components")
            axes[1].set_ylabel("Cumulative explained variance")
            axes[1].set_title("PCA information retention")
        add_panel_label(axes[1], "(b)")
        apply_axis_style(axes[1])
        self.overview_canvas.draw_idle()

    def _draw_map(self) -> None:
        figure = self.map_figure
        figure.clear()
        if not self.manager.has_visualization_data:
            self.map_status.setText("无可用空间样本")
            self._empty_figure(figure, self.map_canvas, "No geolocated training samples")
            return
        latitude = self.manager.training_latitudes
        longitude = self.manager.training_longitudes
        labels = self.manager.training_labels
        mask = np.isfinite(latitude) & np.isfinite(longitude)
        month = str(self.map_time.currentData() or "")
        if month and self.manager.training_times_utc.size == mask.size:
            mask &= np.char.startswith(self.manager.training_times_utc.astype(str), month)
        selected_pattern = self.map_pattern.currentData()
        selected_pattern = int(selected_pattern) if selected_pattern is not None else -1
        if selected_pattern >= 0:
            mask &= labels == selected_pattern
        axis = figure.subplots()
        count = int(np.count_nonzero(mask))
        self.map_status.setText(f"显示 {count:,} / {labels.size:,} 条样本")
        if count:
            pattern_count = max(len(self.manager.representative_metadata), 2)
            scatter = axis.scatter(
                longitude[mask],
                latitude[mask],
                c=labels[mask],
                cmap=colormaps["turbo"].resampled(pattern_count),
                vmin=-0.5,
                vmax=pattern_count - 0.5,
                s=10,
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
            if selected_pattern < 0:
                colorbar = figure.colorbar(scatter, ax=axis, pad=0.015, shrink=0.88)
                colorbar.set_label("Pattern index")
                if pattern_count <= 16:
                    colorbar.set_ticks(np.arange(pattern_count))
                    colorbar.set_ticklabels([str(i + 1) for i in range(pattern_count)])
            for index, metadata in enumerate(self.manager.representative_metadata):
                if selected_pattern >= 0 and index != selected_pattern:
                    continue
                try:
                    rep_latitude = float(metadata.get("latitude_deg"))
                    rep_longitude = float(metadata.get("longitude_deg"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(rep_latitude) and np.isfinite(rep_longitude):
                    axis.scatter(
                        rep_longitude,
                        rep_latitude,
                        marker="*",
                        s=90,
                        color="white",
                        edgecolor="#111820",
                        linewidth=0.8,
                        zorder=3,
                    )
        else:
            axis.text(0.5, 0.5, "No samples match the filter", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlim(-180.0, 180.0)
        axis.set_ylim(-90.0, 90.0)
        axis.set_xticks(np.arange(-180.0, 181.0, 60.0))
        axis.set_yticks(np.arange(-90.0, 91.0, 30.0))
        axis.set_xlabel("Longitude (deg)")
        axis.set_ylabel("Latitude (deg)")
        axis.set_title("Global distribution of learned atmospheric patterns")
        apply_axis_style(axis, grid=True)
        self.map_canvas.draw_idle()

    def _selected_profile_modes(self) -> list[int]:
        return sorted(
            int(item.data(256))
            for item in self.profile_modes.selectedItems()
            if item.data(256) is not None
        )

    def _draw_profiles(self) -> None:
        figure = self.profile_figure
        figure.clear()
        selected = self._selected_profile_modes()
        if not selected or not self.manager.representative_values.size:
            self._empty_figure(figure, self.profile_canvas, "Select one or more patterns")
            return
        columns = self.manager.representative_columns
        column_indices = {name: index for index, name in enumerate(columns)}
        specifications = [
            ("temperature(K)", "Temperature", "$T$ (K)", False),
            ("column_H2O(molec_cm-2)", "Water vapour", "$N_{\\mathrm{H_2O}}$ (molec cm$^{-2}$)", True),
            ("column_O3(molec_cm-2)", "Ozone", "$N_{\\mathrm{O_3}}$ (molec cm$^{-2}$)", True),
        ]
        specifications = [item for item in specifications if item[0] in column_indices]
        pressure_index = column_indices.get("pressure(hPa)")
        if pressure_index is None or not specifications:
            self._empty_figure(figure, self.profile_canvas, "Profile variables unavailable")
            return
        axes = figure.subplots(1, len(specifications), sharey=True, squeeze=False)[0]
        color_map = colormaps["turbo"].resampled(max(len(self.manager.representative_metadata), 2))
        for panel_index, (axis, specification) in enumerate(zip(axes, specifications)):
            column, title, x_label, logarithmic = specification
            value_index = column_indices[column]
            for mode_index in selected:
                color = color_map(mode_index)
                representative = self.manager.representative_values[mode_index]
                pressure = np.maximum(representative[:, pressure_index], 1.0e-8)
                values = representative[:, value_index]
                if logarithmic:
                    values = np.maximum(values, 1.0e-30)
                axis.plot(
                    values,
                    pressure,
                    color=color,
                    lw=1.6,
                    label=f"Pattern {mode_index + 1}",
                )
                if self.manager.cluster_profile_p10.shape == self.manager.cluster_profile_p90.shape and self.manager.cluster_profile_p10.ndim == 3:
                    lower = self.manager.cluster_profile_p10[mode_index, :, value_index]
                    upper = self.manager.cluster_profile_p90[mode_index, :, value_index]
                    if logarithmic:
                        lower = np.maximum(lower, 1.0e-30)
                        upper = np.maximum(upper, 1.0e-30)
                    axis.fill_betweenx(
                        pressure,
                        np.minimum(lower, upper),
                        np.maximum(lower, upper),
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )
            if logarithmic:
                axis.set_xscale("log")
            axis.set_yscale("log")
            axis.invert_yaxis()
            axis.set_xlabel(x_label)
            axis.set_title(title)
            add_panel_label(axis, f"({chr(ord('a') + panel_index)})")
            apply_axis_style(axis)
        axes[0].set_ylabel("Pressure (hPa)")
        axes[-1].legend(loc="best")
        self.profile_canvas.draw_idle()

    def _draw_feature_space(self) -> None:
        figure = self.feature_figure
        figure.clear()
        if not self.manager.has_visualization_data:
            self._empty_figure(figure, self.feature_canvas, "No saved latent-space samples")
            return
        scores = self.manager.training_scores
        x_values = scores[:, 0]
        y_values = scores[:, 1] if scores.shape[1] > 1 else np.zeros(scores.shape[0])
        pattern_count = max(len(self.manager.representative_metadata), 2)
        axis = figure.subplots()
        scatter = axis.scatter(
            x_values,
            y_values,
            c=self.manager.training_labels,
            cmap=colormaps["turbo"].resampled(pattern_count),
            vmin=-0.5,
            vmax=pattern_count - 0.5,
            s=10,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        representative = self.manager.representative_scores
        representative_y = representative[:, 1] if representative.shape[1] > 1 else np.zeros(representative.shape[0])
        axis.scatter(
            representative[:, 0],
            representative_y,
            marker="*",
            s=125,
            c=np.arange(representative.shape[0]),
            cmap=colormaps["turbo"].resampled(pattern_count),
            vmin=-0.5,
            vmax=pattern_count - 0.5,
            edgecolor="#111820",
            linewidth=0.8,
            label="Representative pattern",
        )
        prefix = "PC" if self.manager.method == "pca" else "$z_{}$"
        if self.manager.method == "pca":
            ratios = self.manager.pca_explained_variance_ratio
            x_label = f"PC 1 ({ratios[0]:.1%})" if ratios.size else "PC 1"
            y_label = f"PC 2 ({ratios[1]:.1%})" if ratios.size > 1 else "PC 2"
        else:
            x_label = prefix.format(1)
            y_label = prefix.format(2)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title("Learned feature space and representative patterns")
        axis.legend(loc="best")
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.015, shrink=0.88)
        colorbar.set_label("Pattern index")
        if pattern_count <= 16:
            colorbar.set_ticks(np.arange(pattern_count))
            colorbar.set_ticklabels([str(i + 1) for i in range(pattern_count)])
        apply_axis_style(axis)
        self.feature_canvas.draw_idle()

    def _request_selected_profile(self) -> None:
        selected = self._selected_profile_modes()
        if selected:
            self.representativeRequested.emit(selected[0])

    @staticmethod
    def _empty_figure(
        figure: Figure, canvas: DebouncedFigureCanvas, message: str
    ) -> None:
        axis = figure.subplots()
        axis.set_facecolor(AXIS_BACKGROUND)
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
            color=TEXT_COLOR,
        )
        axis.set_axis_off()
        canvas.draw_idle()
