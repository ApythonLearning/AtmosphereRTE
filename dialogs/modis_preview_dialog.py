from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QVBoxLayout

from core.atmospheric_radiation_manager import (
    EarthEnvironmentGrid,
    Merra2AerosolField,
    ModisDataManager,
)
from dialogs.matplotlib_canvas import DebouncedFigureCanvas
from dialogs.window_controls import configure_resizable_dialog


class ModisPreviewDialog(QDialog):
    """MODIS地表/云场与MERRA-2气溶胶场的质量检查窗口。"""

    def __init__(
        self,
        grid: EarthEnvironmentGrid,
        aerosol_field: Merra2AerosolField | None = None,
        aerosol_time_index: int = 0,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("地球环境数据预览")
        configure_resizable_dialog(self)
        self.resize(1380, 960)
        self.setMinimumSize(980, 700)

        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Microsoft YaHei", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
        })
        # constrained_layout会在每次窗口尺寸变化时重新求解全部地图和色标，
        # 对交互缩放开销很大；这里使用一次性固定边距布局。
        figure = Figure(figsize=(13.8, 9.2), dpi=100, constrained_layout=False)
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

        self._draw(figure, grid, aerosol_field, aerosol_time_index)
        figure.subplots_adjust(
            left=0.052, right=0.982, bottom=0.062, top=0.955,
            wspace=0.36, hspace=0.54,
        )
        canvas.draw()

    def _draw(
        self,
        figure: Figure,
        grid: EarthEnvironmentGrid,
        aerosol_field: Merra2AerosolField | None,
        aerosol_time_index: int,
    ) -> None:
        axes = figure.subplots(3, 3)
        panels = [
            (grid.surface_temperature_k, "(a)", "地表温度", "Temperature (K)", "inferno", "bilinear", "upper", (-180.0, 180.0, -90.0, 90.0)),
            (grid.surface_type, "(b)", "地表类型", "MODIS class", "tab20", "nearest", "upper", (-180.0, 180.0, -90.0, 90.0)),
            (grid.cloud_fraction * 100.0, "(c)", "云量", "Cloud fraction (%)", "Blues", "bilinear", "upper", (-180.0, 180.0, -90.0, 90.0)),
            (grid.cloud_top_temperature_k, "(d)", "云顶温度", "Temperature (K)", "magma", "bilinear", "upper", (-180.0, 180.0, -90.0, 90.0)),
            (grid.cloud_top_height_m / 1000.0, "(e)", "云顶高度", "Height (km)", "viridis", "bilinear", "upper", (-180.0, 180.0, -90.0, 90.0)),
        ]
        cloud_lwp = (
            np.asarray(grid.cloud_liquid_water_path_g_m2, dtype=float)
            if grid.cloud_liquid_water_path_g_m2 is not None
            else np.full(grid.cloud_fraction.shape, np.nan, dtype=float)
        )
        # MOD08的液态水路径只对液态云反演；晴空处即使存在回退数组值也不显示。
        cloud_lwp = np.where(
            (np.asarray(grid.cloud_fraction, dtype=float) > 0.0)
            & np.isfinite(cloud_lwp)
            & (cloud_lwp > 0.0),
            cloud_lwp,
            np.nan,
        )
        lwp_source = str(grid.metadata.get("cloud_liquid_water_path_source", "未记录"))
        lwp_source_labels = {
            "modis_with_optical_thickness_and_empirical_fallback": "MODIS/缺失回退",
            "modis_optical_thickness_with_empirical_fallback": "光学厚度/缺失回退",
            "empirical": "经验估计",
        }
        lwp_title = "液态水路径"
        if lwp_source in lwp_source_labels:
            lwp_title += f"（{lwp_source_labels[lwp_source]}）"
        panels.append((
            cloud_lwp, "(f)", lwp_title, r"LWP (g m$^{-2}$)",
            "YlGnBu", "bilinear", "upper", (-180.0, 180.0, -90.0, 90.0),
        ))
        opac_type_order = tuple(ModisDataManager.OPAC_AEROSOL_TYPE_LABELS)
        opac_types = ModisDataManager.opac_aerosol_type_grid(
            grid.surface_type, grid.latitude
        )
        opac_type_codes = np.zeros(opac_types.shape, dtype=int)
        for type_code, type_name in enumerate(opac_type_order):
            opac_type_codes[opac_types == type_name] = type_code
        panels.append((
            opac_type_codes, "(g)", "下垫面OPAC气溶胶类型", "OPAC type code",
            "tab10", "nearest", "upper", (-180.0, 180.0, -90.0, 90.0),
        ))
        selected_time_index: int | None = None
        if aerosol_field is not None:
            selected_time_index = int(np.clip(
                aerosol_time_index, 0, aerosol_field.time_seconds.size - 1
            ))
            aerosol_extent = (
                float(np.min(aerosol_field.longitude)),
                float(np.max(aerosol_field.longitude)),
                float(np.min(aerosol_field.latitude)),
                float(np.max(aerosol_field.latitude)),
            )
            panels.append((
                aerosol_field.aerosol_optical_depth_550[selected_time_index],
                "(h)", "气溶胶光学厚度", r"AOD$_{550}$", "YlOrRd", "bilinear", "lower", aerosol_extent,
            ))
            if aerosol_field.angstrom_exponent is not None:
                panels.append((
                    aerosol_field.angstrom_exponent[selected_time_index],
                    "(i)", "Ångström指数", "470–870 nm", "cividis", "bilinear", "lower", aerosol_extent,
                ))
        for axis, (values, panel_label, title, colorbar_label, cmap, interpolation, origin, extent) in zip(axes.ravel(), panels):
            image = axis.imshow(
                np.asarray(values),
                origin=origin,
                extent=extent,
                aspect="auto",
                cmap=cmap,
                interpolation=interpolation,
            )
            is_opac_type_panel = title == "下垫面OPAC气溶胶类型"
            if is_opac_type_panel:
                image.set_clim(-0.5, len(opac_type_order) - 0.5)
            axis.set_title(title, fontsize=11, color="#f3f8fc", fontfamily="SimSun")
            axis.set_xlabel("Longitude (deg)", fontsize=9)
            axis.set_ylabel("Latitude (deg)", fontsize=9)
            axis.set_xticks([-180, -120, -60, 0, 60, 120, 180])
            axis.set_yticks([-90, -60, -30, 0, 30, 60, 90])
            self._style_axis(axis)
            axis.text(
                0.025, 0.96, panel_label, transform=axis.transAxes,
                fontsize=11, fontweight="bold", fontstyle="normal",
                va="top", ha="left", color="#ffffff",
                bbox={"facecolor": "#253242", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
            colorbar.set_label(colorbar_label, fontsize=9, color="#e9f0f7")
            colorbar.ax.tick_params(colors="#d7e2ec", direction="in", labelsize=8)
            colorbar.outline.set_edgecolor("#71869a")
            if is_opac_type_panel:
                colorbar.set_ticks(np.arange(len(opac_type_order)))

        for axis in axes.ravel()[len(panels):]:
            axis.set_visible(False)

    def _style_axis(self, axis: Any) -> None:
        axis.set_facecolor("#2d3b4b")
        axis.tick_params(which="both", direction="in", top=True, right=True,
                         colors="#d7e2ec", labelsize=8)
        axis.minorticks_on()
        axis.grid(False)
        axis.xaxis.label.set_color("#e9f0f7")
        axis.yaxis.label.set_color("#e9f0f7")
        for spine in axis.spines.values():
            spine.set_color("#71869a")

    @classmethod
    def statistics_lines(
        cls,
        grid: EarthEnvironmentGrid,
        aerosol_field: Merra2AerosolField | None,
        aerosol_time_index: int | None,
    ) -> list[str]:
        valid = np.asarray(grid.valid_mask, dtype=bool)

        def stats(values: np.ndarray) -> tuple[float, float, float]:
            array = np.asarray(values, dtype=float)
            selected = array[valid & np.isfinite(array)]
            if selected.size == 0:
                return float("nan"), float("nan"), float("nan")
            return float(np.min(selected)), float(np.mean(selected)), float(np.max(selected))

        surface = stats(grid.surface_temperature_k)
        cloud_fraction = stats(grid.cloud_fraction * 100.0)
        cloud_temperature = stats(grid.cloud_top_temperature_k)
        cloud_height = stats(grid.cloud_top_height_m / 1000.0)
        cloud_lwp_values = (
            np.asarray(grid.cloud_liquid_water_path_g_m2, dtype=float)
            if grid.cloud_liquid_water_path_g_m2 is not None
            else np.full(valid.shape, np.nan, dtype=float)
        )
        cloud_lwp_values = np.where(
            np.asarray(grid.cloud_fraction, dtype=float) > 0.0,
            cloud_lwp_values,
            np.nan,
        )
        cloud_lwp = stats(cloud_lwp_values)
        cloud_lwp_source = str(
            grid.metadata.get("cloud_liquid_water_path_source", "未记录")
        )
        resolution = float(grid.metadata.get("resolution_deg", 0.0))
        valid_ratio = 100.0 * float(np.count_nonzero(valid)) / max(valid.size, 1)
        lines = [
            f"网格规模={valid.shape[0]}×{valid.shape[1]}，空间分辨率={resolution:.3g} deg，有效数据比例={valid_ratio:.2f}%",
            f"地表温度(K)：min={surface[0]:.2f}，mean={surface[1]:.2f}，max={surface[2]:.2f}",
            f"云量(%)：min={cloud_fraction[0]:.2f}，mean={cloud_fraction[1]:.2f}，max={cloud_fraction[2]:.2f}",
            f"云顶温度(K)：min={cloud_temperature[0]:.2f}，mean={cloud_temperature[1]:.2f}，max={cloud_temperature[2]:.2f}",
            f"云顶高度(km)：min={cloud_height[0]:.2f}，mean={cloud_height[1]:.2f}，max={cloud_height[2]:.2f}",
            f"液态水路径LWP(g/m²)：min={cloud_lwp[0]:.2f}，mean={cloud_lwp[1]:.2f}，max={cloud_lwp[2]:.2f}，来源={cloud_lwp_source}",
        ]
        if aerosol_field is not None and aerosol_time_index is not None:
            aod = cls._finite_stats(aerosol_field.aerosol_optical_depth_550[aerosol_time_index])
            lines.append(f"MERRA-2匹配时刻={aerosol_field.time_labels[aerosol_time_index]}")
            lines.append(
                f"AOD550：min={aod[0]:.4f}，mean={aod[1]:.4f}，max={aod[2]:.4f}"
            )
            if aerosol_field.angstrom_exponent is not None:
                angstrom = cls._finite_stats(aerosol_field.angstrom_exponent[aerosol_time_index])
                lines.append(
                    f"Ångström指数：min={angstrom[0]:.3f}，mean={angstrom[1]:.3f}，max={angstrom[2]:.3f}"
                )
        else:
            lines.append("MERRA-2气溶胶数据未加载，AOD使用能见度回退")
        opac_types = ModisDataManager.opac_aerosol_type_grid(
            grid.surface_type, grid.latitude
        )
        unique_types, counts = np.unique(opac_types, return_counts=True)
        type_summary = "、".join(
            f"{ModisDataManager.OPAC_AEROSOL_TYPE_LABELS.get(str(type_name), str(type_name))}={int(count)}"
            for type_name, count in zip(unique_types, counts)
        )
        lines.append(f"OPAC网格类型统计：{type_summary}")
        return lines

    @staticmethod
    def _finite_stats(values: np.ndarray) -> tuple[float, float, float]:
        selected = np.asarray(values, dtype=float)
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.min(selected)), float(np.mean(selected)), float(np.max(selected))
