from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QVBoxLayout

from dialogs.matplotlib_canvas import DebouncedFigureCanvas
from dialogs.window_controls import configure_resizable_dialog


class AbsorptionPreviewDialog(QDialog):
    """总气体分子光学厚度的光谱与分层质量检查窗口。"""

    MAX_SPECTRAL_POINTS = 6000
    MAX_HEATMAP_COLUMNS = 3000

    def __init__(self, visualization_data: dict[str, Any], parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("总气体分子光学厚度预览")
        configure_resizable_dialog(self)
        self.resize(1450, 850)
        self.setMinimumSize(900, 620)

        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "SimSun", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        })
        # 固定布局并延迟缩放重绘，避免多幅子图和色标持续重新布局。
        figure = Figure(figsize=(14.5, 8.0), dpi=100, constrained_layout=False)
        figure.patch.set_facecolor("#253242")
        canvas = DebouncedFigureCanvas(figure)
        self._canvas = canvas
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.setObjectName("plotNavigationToolbar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        self._draw(figure, visualization_data)
        figure.subplots_adjust(
            left=0.050, right=0.980, bottom=0.075, top=0.955,
            wspace=0.32, hspace=0.30,
        )
        canvas.draw()

    def _draw(self, figure: Figure, data: dict[str, Any]) -> None:
        wavenumber = np.asarray(data["wavenumber_cm"], dtype=float)
        tau_layers = np.asarray(data["total_tau_layers"], dtype=float)
        solar_irradiance = np.asarray(
            data["solar_spectral_irradiance_w_m2_per_cm"], dtype=float
        )
        solar_source = str(data.get("solar_spectrum_source", "未记录"))
        sources = data.get("sources", [])
        corrections = data.get("optical_depth_corrections", [])
        grid = figure.add_gridspec(2, 3)
        axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[0, 2]),
            figure.add_subplot(grid[1, 0:2]),
            figure.add_subplot(grid[1, 2]),
        ]

        spectral_indices = self._sample_indices(wavenumber.size, self.MAX_SPECTRAL_POINTS)
        column_tau = np.sum(tau_layers, axis=0)
        axes[0].semilogy(wavenumber[spectral_indices], np.maximum(column_tau[spectral_indices], 1e-30),
                         color="#ffcc66", linewidth=1.0)
        axes[0].set_title("总柱气体光学厚度谱", fontfamily="SimSun")
        axes[0].set_xlabel(r"Wavenumber $\tilde{\nu}$ (cm$^{-1}$)")
        axes[0].set_ylabel(r"Column optical depth $\tau$")

        valid_solar = np.isfinite(solar_irradiance) & (solar_irradiance >= 0.0)
        solar_indices = spectral_indices[valid_solar[spectral_indices]]
        if solar_indices.size:
            axes[1].plot(
                wavenumber[solar_indices], solar_irradiance[solar_indices],
                color="#f6b73c", linewidth=1.05,
            )
        axes[1].set_title("大气层顶入射太阳辐射光谱", fontfamily="SimSun")
        axes[1].set_xlabel(r"Wavenumber $\tilde{\nu}$ (cm$^{-1}$)")
        axes[1].set_ylabel(
            r"$E_{\tilde{\nu}}^{\rm TOA}$ (W m$^{-2}$ (cm$^{-1}$)$^{-1}$)"
        )

        colors = mpl.colormaps["viridis"]
        representative_layers = np.unique(np.linspace(0, tau_layers.shape[0] - 1, min(8, tau_layers.shape[0]), dtype=int))
        for curve_index, layer_index in enumerate(representative_layers):
            axes[2].semilogy(
                wavenumber[spectral_indices], np.maximum(tau_layers[layer_index, spectral_indices], 1e-30),
                color=colors(curve_index / max(len(representative_layers) - 1, 1)),
                linewidth=0.9, label=f"Layer {layer_index + 1}",
            )
        axes[2].set_title("代表性大气层光学厚度谱", fontfamily="SimSun")
        axes[2].set_xlabel(r"Wavenumber $\tilde{\nu}$ (cm$^{-1}$)")
        axes[2].set_ylabel(r"Layer optical depth $\tau_l$")
        axes[2].legend(fontsize=7, ncol=2, loc="best")

        heatmap_indices = self._sample_indices(wavenumber.size, self.MAX_HEATMAP_COLUMNS)
        log_tau = np.log10(np.maximum(tau_layers[:, heatmap_indices], 1e-30))
        image = axes[3].imshow(
            log_tau,
            origin="lower",
            extent=(float(wavenumber[heatmap_indices[0]]), float(wavenumber[heatmap_indices[-1]]), 0.5, tau_layers.shape[0] + 0.5),
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
        )
        axes[3].set_title("分层光学厚度分布", fontfamily="SimSun")
        axes[3].set_xlabel(r"Wavenumber $\tilde{\nu}$ (cm$^{-1}$)")
        axes[3].set_ylabel("Atmospheric layer")
        colorbar = figure.colorbar(image, ax=axes[3], fraction=0.025, pad=0.020)
        colorbar.set_label(r"$\log_{10}(\tau)$", color="#e9f0f7")
        colorbar.ax.tick_params(colors="#d7e2ec", direction="in", labelsize=8)
        colorbar.outline.set_edgecolor("#71869a")

        layer_integral = np.trapezoid(tau_layers, wavenumber, axis=1)
        layer_number = np.arange(1, tau_layers.shape[0] + 1)
        axes[4].barh(layer_number, layer_integral, color="#70a9d6", edgecolor="#b9d6ec", linewidth=0.3)
        axes[4].set_title("各层波数积分光学厚度", fontfamily="SimSun")
        axes[4].set_xlabel(r"Integrated optical depth (cm$^{-1}$)")
        axes[4].set_ylabel("Atmospheric layer")
        axes[4].text(
            0.98, 0.04,
            f"Input CSV: {len(sources)}\nSpectral points: {wavenumber.size:,}\n"
            f"Layers: {tau_layers.shape[0]}\nCorrections: {len(corrections)}\nSolar: {solar_source}",
            transform=axes[4].transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#e3edf5", bbox={"facecolor": "#253242", "edgecolor": "#53687d", "alpha": 0.88},
        )

        for index, axis in enumerate(axes):
            self._style_axis(axis)
            axis.text(
                0.025, 0.96, f"({chr(ord('a') + index)})", transform=axis.transAxes,
                fontsize=11, fontweight="bold", fontstyle="normal", va="top", ha="left",
                color="#ffffff", bbox={"facecolor": "#253242", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            )

    def _style_axis(self, axis: Any) -> None:
        axis.set_facecolor("#2d3b4b")
        axis.tick_params(which="both", direction="in", top=True, right=True, colors="#d7e2ec", labelsize=8)
        axis.minorticks_on()
        axis.grid(False)
        axis.xaxis.label.set_color("#e9f0f7")
        axis.yaxis.label.set_color("#e9f0f7")
        axis.title.set_color("#f3f8fc")
        for spine in axis.spines.values():
            spine.set_color("#71869a")

    def _sample_indices(self, size: int, maximum: int) -> np.ndarray:
        if size <= maximum:
            return np.arange(size, dtype=int)
        return np.unique(np.linspace(0, size - 1, maximum, dtype=int))
