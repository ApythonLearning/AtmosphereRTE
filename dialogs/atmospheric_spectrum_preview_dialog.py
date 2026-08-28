from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from PySide6.QtWidgets import QDialog, QVBoxLayout

from dialogs.matplotlib_canvas import DebouncedFigureCanvas
from dialogs.window_controls import configure_resizable_dialog


class AtmosphericSpectrumPreviewDialog(QDialog):
    """指定时刻大气辐射传输求解结果窗口。"""

    def __init__(self, spectrum: dict[str, Any], parent: Any = None) -> None:
        super().__init__(parent)
        preview_mode = spectrum.get("preview_mode")
        high_resolution = preview_mode == "high_resolution"
        if high_resolution:
            self.setWindowTitle("指定时刻大气辐射高分辨率目标单柱结果")
        else:
            self.setWindowTitle("指定时刻大气辐射传输求解结果")
        configure_resizable_dialog(self)
        self.resize(1280, 820)
        self.setMinimumSize(900, 620)

        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Microsoft YaHei", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        })
        # 固定布局避免缩放时反复求解四幅子图、图例和长文本的位置。
        self.figure = Figure(figsize=(13.0, 7.8), dpi=100, constrained_layout=False)
        self.figure.patch.set_facecolor("#253242")
        self.canvas = DebouncedFigureCanvas(self.figure)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setObjectName("plotNavigationToolbar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self._draw(spectrum)
        self.figure.subplots_adjust(
            left=0.055, right=0.985, bottom=0.075, top=0.955,
            wspace=0.22, hspace=0.28,
        )
        self.canvas.draw()

    def _draw(self, spectrum: dict[str, Any]) -> None:
        preview_mode = spectrum.get("preview_mode")
        high_resolution = preview_mode == "high_resolution"
        if high_resolution:
            horizontal = np.asarray(spectrum["wavenumber_cm"], dtype=float)
            thermal = np.asarray(spectrum["earth_thermal_spectral_wavenumber"], dtype=float)
            reflected = np.asarray(spectrum["earth_reflected_spectral_wavenumber"], dtype=float)
            single_scattering = np.asarray(spectrum["solar_single_scattering_spectral_wavenumber"], dtype=float)
            multiple_scattering = np.asarray(spectrum["solar_multiple_scattering_spectral_wavenumber"], dtype=float)
            surface_reflection = np.asarray(spectrum["solar_surface_reflection_spectral_wavenumber"], dtype=float)
            horizontal_label = r"Wavenumber $\tilde{\nu}$ (cm$^{-1}$)"
            spectral_label = r"Spectral radiance (W m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)"
            integrated_label = r"Integrated radiance (W m$^{-2}$ sr$^{-1}$)"
        else:
            horizontal = np.asarray(spectrum["wavelength_um"], dtype=float)
            thermal = np.asarray(spectrum["earth_thermal_spectral_irradiance"], dtype=float)
            reflected = np.asarray(spectrum["earth_reflected_spectral_irradiance"], dtype=float)
            single_scattering = np.asarray(spectrum["solar_single_scattering_spectral_irradiance"], dtype=float)
            multiple_scattering = np.asarray(spectrum["solar_multiple_scattering_spectral_irradiance"], dtype=float)
            surface_reflection = np.asarray(spectrum["solar_surface_reflection_spectral_irradiance"], dtype=float)
            horizontal_label = r"Wavelength $\lambda$ ($\mu$m)"
            spectral_label = r"Spectral irradiance (W m$^{-2}$ $\mu$m$^{-1}$)"
            integrated_label = r"Integrated irradiance (W m$^{-2}$)"
        thermal_integral = float(spectrum["earth_thermal_irradiance"])
        reflected_integral = float(spectrum["earth_reflected_irradiance"])
        single_integral = float(spectrum["solar_single_scattering_irradiance"])
        multiple_integral = float(spectrum["solar_multiple_scattering_irradiance"])
        surface_integral = float(spectrum["solar_surface_reflection_irradiance"])
        axes = self.figure.subplots(2, 2)

        axes[0, 0].plot(horizontal, thermal, color="#ffb347", linewidth=1.0)
        axes[0, 0].set_title(
            "目标单柱地球热辐射光谱"
            if high_resolution
            else "地球热辐射到达目标的光谱",
            fontfamily="Microsoft YaHei",
        )
        axes[0, 0].set_xlabel(horizontal_label)
        axes[0, 0].set_ylabel(spectral_label)

        axes[0, 1].plot(
            horizontal, reflected, color="#e8eef5", linewidth=1.05, alpha=0.80,
            label=f"Reflected solar total (I={reflected_integral:.4g})", zorder=1,
        )
        axes[0, 1].plot(
            horizontal, surface_reflection, color="#d8a4ff", linewidth=0.8,
            label=f"Atmosphere-attenuated surface reflection (I={surface_integral:.4g})", zorder=2,
        )
        axes[0, 1].plot(
            horizontal, single_scattering, color="#62b6ff", linewidth=0.8,
            label=f"Single scattering (I={single_integral:.4g})", zorder=3,
        )
        axes[0, 1].plot(
            horizontal, multiple_scattering, color="#8bd17c", linewidth=1.05,
            label=f"Multiple scattering (I={multiple_integral:.4g})", zorder=4,
        )
        axes[0, 1].set_title(
            "目标单柱反射太阳辐射三分量"
            if high_resolution
            else "地球反射太阳辐射到达目标的光谱",
            fontfamily="Microsoft YaHei",
        )
        axes[0, 1].set_xlabel(horizontal_label)
        axes[0, 1].set_ylabel(spectral_label)
        axes[0, 1].legend(fontsize=7.5, loc="best")

        common_scale = max(
            float(np.max(single_scattering)) if single_scattering.size else 0.0,
            float(np.max(multiple_scattering)) if multiple_scattering.size else 0.0,
            float(np.max(surface_reflection)) if surface_reflection.size else 0.0,
        )
        axes[1, 0].plot(
            horizontal,
            single_scattering / common_scale if common_scale > 0.0 else np.zeros_like(single_scattering),
            color="#62b6ff",
            linewidth=1.1,
            label=f"Single scattering (I={single_integral:.4g})",
        )
        axes[1, 0].plot(
            horizontal,
            multiple_scattering / common_scale if common_scale > 0.0 else np.zeros_like(multiple_scattering),
            color="#8bd17c",
            linewidth=1.25,
            label=f"Multiple scattering (I={multiple_integral:.4g})",
        )
        axes[1, 0].plot(
            horizontal,
            surface_reflection / common_scale if common_scale > 0.0 else np.zeros_like(surface_reflection),
            color="#d8a4ff",
            linewidth=1.1,
            label=f"Atmosphere-attenuated reflection (I={surface_integral:.4g})",
        )
        axes[1, 0].set_title("反射太阳辐射三分量统一尺度对比", fontfamily="Microsoft YaHei")
        axes[1, 0].set_xlabel(horizontal_label)
        axes[1, 0].set_ylabel("Normalized spectrum")
        axes[1, 0].set_ylim(-0.02, 1.08)
        axes[1, 0].legend(fontsize=8, loc="best")

        integrated = np.asarray([
            thermal_integral,
            single_integral,
            multiple_integral,
            surface_integral,
            reflected_integral,
        ])
        bars = axes[1, 1].bar(
            np.arange(5), integrated,
            color=["#ffb347", "#62b6ff", "#8bd17c", "#d8a4ff", "#a9c7dc"],
            edgecolor="#dce8f1", linewidth=0.5,
        )
        axes[1, 1].set_xticks(
            np.arange(5), ["Thermal", "Single", "Multiple", "Atten. surface", "Solar total"],
            rotation=12,
        )
        axes[1, 1].set_ylabel(integrated_label)
        axes[1, 1].set_title("光谱积分与目标信息", fontfamily="Microsoft YaHei")
        axes[1, 1].bar_label(bars, labels=[f"{value:.4g}" for value in integrated], padding=3, color="#e9f0f7", fontsize=8)
        integrated_maximum = float(np.max(integrated)) if integrated.size else 0.0
        axes[1, 1].set_ylim(0.0, 2.0 * integrated_maximum if integrated_maximum > 0.0 else 1.0)
        axes[1, 1].text(
            0.98, 0.96,
            self._summary_text(spectrum),
            transform=axes[1, 1].transAxes,
            ha="right", va="top", fontsize=8.5, linespacing=1.4,
            color="#e3edf5", fontfamily="Microsoft YaHei",
            bbox={"facecolor": "#253242", "edgecolor": "#53687d", "alpha": 0.90, "pad": 5.0},
        )

        for index, axis in enumerate(axes.ravel()):
            self._style_axis(axis)
            if index < 3 and not high_resolution:
                axis.set_xscale("log")
            if index < 3:
                axis.set_xlim(float(np.min(horizontal)), float(np.max(horizontal)))
            axis.text(
                0.025, 0.96, f"({chr(ord('a') + index)})",
                transform=axis.transAxes, fontsize=11, fontweight="bold",
                fontstyle="normal", va="top", ha="left", color="#ffffff",
                bbox={"facecolor": "#253242", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            )

    def _style_axis(self, axis: Any) -> None:
        axis.set_facecolor("#2d3b4b")
        axis.tick_params(which="both", direction="in", top=True, right=True,
                         colors="#d7e2ec", labelsize=8)
        axis.minorticks_on()
        axis.grid(False)
        axis.xaxis.label.set_color("#e9f0f7")
        axis.yaxis.label.set_color("#e9f0f7")
        axis.title.set_color("#f3f8fc")
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axis.yaxis.get_offset_text().set_color("#d7e2ec")
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#e9f0f7")
        for spine in axis.spines.values():
            spine.set_color("#71869a")

    def _summary_text(self, spectrum: dict[str, Any]) -> str:
        optical_model = "Imported total optical depth" if spectrum.get("uses_total_optical_depth") else "Built-in gas model"
        environment_text = self._environment_text(spectrum)
        if spectrum.get("preview_mode") == "high_resolution":
            wavenumber = np.asarray(spectrum["wavenumber_cm"], dtype=float)
            step = float(np.median(np.diff(wavenumber))) if wavenumber.size > 1 else 0.0
            return (
                "Mode: High-resolution single column\n"
                f"Column: {float(spectrum['representative_latitude_deg']):.3f}°, "
                f"{float(spectrum['representative_longitude_deg']):.3f}°\n"
                f"Cloud: f={float(spectrum['representative_cloud_fraction']):.3f}, "
                f"r_eff={float(spectrum['representative_cloud_effective_radius_um']):.2f} um, "
                f"LWP={float(spectrum['representative_cloud_liquid_water_path_g_m2']):.2f} g/m^2\n"
                f"Cloud fraction source: {spectrum.get('cloud_fraction_source', 'Not recorded')}\n"
                f"View target altitude: {float(spectrum['target_altitude_m']) / 1000.0:.3f} km\n"
                f"Satellite zenith/azimuth: "
                f"{float(spectrum.get('satellite_zenith_deg', 0.0)):.3f}°/"
                f"{float(spectrum.get('satellite_azimuth_deg', 0.0)):.3f}°\n"
                f"View air-mass factor: "
                f"{float(spectrum.get('view_air_mass_factor', 1.0)):.5f}\n"
                f"Solar zenith/azimuth: "
                f"{float(spectrum.get('solar_zenith_deg', 0.0)):.3f}°/"
                f"{float(spectrum.get('solar_azimuth_deg', 0.0)):.3f}°\n"
                f"Range: {float(spectrum['wavenumber_min_cm']):.1f}–"
                f"{float(spectrum['wavenumber_max_cm']):.1f} cm^-1\n"
                f"Step: {step:.3g} cm^-1\n"
                f"Spectral points: {int(spectrum['spectral_point_count']):,}\n"
                f"Solar spectrum: {spectrum.get('solar_spectrum_source', 'Not recorded')}\n"
                f"Multiple/single integral: {float(spectrum['solar_multiple_scattering_irradiance']) / max(float(spectrum['solar_single_scattering_irradiance']), 1e-30):.4g}\n"
                f"Gas absorption: {optical_model}\n"
                f"Temperature profile: {spectrum.get('temperature_profile_source', 'Not recorded')}\n"
                f"{environment_text}"
            )
        return (
            "Mode: Fast Earth-disk integration\n"
            f"Time: {float(spectrum.get('target_time', 0.0)):.6g}\n"
            f"Position: {float(spectrum['target_latitude_deg']):.3f}°, "
            f"{float(spectrum['target_longitude_deg']):.3f}°\n"
            f"Altitude: {float(spectrum['target_altitude_m']) / 1000.0:.3f} km\n"
            f"Weighted cloud: f={float(spectrum['representative_cloud_fraction']):.3f}, "
            f"r_eff={float(spectrum['representative_cloud_effective_radius_um']):.2f} um, "
            f"LWP={float(spectrum['representative_cloud_liquid_water_path_g_m2']):.2f} g/m^2\n"
            f"Visible cells: {int(spectrum['visible_cell_count']):,}\n"
            f"Evaluated cells: {int(spectrum.get('evaluated_cell_count', spectrum['visible_cell_count'])):,}\n"
            f"Spectral points: {int(spectrum['spectral_point_count']):,}\n"
            f"Solar spectrum: {spectrum.get('solar_spectrum_source', 'Not recorded')}\n"
            f"Gas absorption: {optical_model}\n"
            f"{environment_text}"
        )

    def _environment_text(self, spectrum: dict[str, Any]) -> str:
        requested = spectrum.get("requested_time")
        actual = float(spectrum.get("target_time", 0.0))
        requested_text = (
            f"Requested/actual time: {float(requested):.6g}/{actual:.6g} s\n"
            if requested is not None else ""
        )
        aerosol_source = str(spectrum.get("aerosol_optical_depth_source", "能见度估算"))
        merra_time = str(spectrum.get("merra2_time", "-"))
        merra_text = f"\nMERRA-2 time: {merra_time}" if merra_time not in {"", "-"} else ""
        return (
            f"{requested_text}"
            f"MODIS grid: {float(spectrum['environment_grid_latitude_deg']):.3f}°, "
            f"{float(spectrum['environment_grid_longitude_deg']):.3f}°\n"
            f"Surface: {int(spectrum['subpoint_surface_type_code'])} "
            f"{spectrum['subpoint_surface_type_name']}, "
            f"T={float(spectrum['subpoint_surface_temperature_k']):.2f} K, "
            f"a={float(spectrum['subpoint_surface_albedo']):.3f}, "
            f"e={float(spectrum['subpoint_surface_emissivity']):.3f}\n"
            f"Environment-grid cloud (reference): f={float(spectrum['subpoint_cloud_fraction']):.3f}, "
            f"h={float(spectrum['subpoint_cloud_top_height_m']) / 1000.0:.2f} km, "
            f"tau={float(spectrum['subpoint_cloud_optical_thickness']):.3g}\n"
            f"Visibility/AOD550: {float(spectrum['visibility_km']):.3g} km/"
            f"{float(spectrum['aerosol_optical_depth_550']):.4g}\n"
            f"AOD source: {aerosol_source}{merra_text}"
            f"\nOPAC type: {spectrum.get('subpoint_opac_aerosol_type_name', '-')}"
        )
