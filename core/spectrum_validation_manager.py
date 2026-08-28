from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_SPECTRUM_EXTENSIONS = {".csv", ".txt", ".dat", ".npz", ".nc", ".nc4", ".h5", ".hdf5"}
CLOUD_FIT_WINDOWS_UM = ((8.0, 9.2), (10.2, 12.5))
NUCAPS_CLOUD_PRIOR_RELATIVE_WEIGHT = 0.05

class SpectrumValidationManager:
    """读取异构光谱并在共同波长网格上进行定量验证。"""

    AXIS_ALIASES = (
        "wavelength_um", "wavelength", "wavelength_nm", "lambda", "wl",
        "wavenumber_cm", "wavenumber", "wn", "frequency",
    )
    VALUE_ALIASES = (
        "radiance_w_m2_sr_um",
        "radiance", "spectral_radiance", "irradiance", "spectral_irradiance",
        "signal", "value", "spectrum", "bt", "brightness_temperature",
    )

    @staticmethod
    def three_point_hamming(
        values: np.ndarray,
        coordinate: np.ndarray | None = None,
    ) -> np.ndarray:
        """执行CrIS SDR规定的三点Hamming切趾。

        NOAA CrIS SDR使用 ``[0.23, 0.54, 0.23]``。提供波数坐标时会在
        CrIS三个独立谱段的间隙处分段，防止LW/MW/SW边界相互卷积。
        """
        spectrum = np.asarray(values, dtype=float).reshape(-1)
        if spectrum.size < 3:
            return spectrum.copy()
        weights = np.asarray([0.23, 0.54, 0.23], dtype=float)
        if coordinate is None:
            padded = np.pad(spectrum, (1, 1), mode="edge")
            return np.convolve(padded, weights, mode="valid")
        spectral_coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
        if spectral_coordinate.size != spectrum.size:
            raise ValueError("Hamming切趾的光谱坐标和值长度不一致。")
        gaps = np.abs(np.diff(spectral_coordinate))
        finite_positive = gaps[np.isfinite(gaps) & (gaps > 0.0)]
        if finite_positive.size == 0:
            padded = np.pad(spectrum, (1, 1), mode="edge")
            return np.convolve(padded, weights, mode="valid")
        # 取较小90%的中位数估计通道间隔，避免两个谱段大间隙抬高阈值。
        upper_spacing = float(np.percentile(finite_positive, 90.0))
        nominal_spacing = float(np.median(finite_positive[finite_positive <= upper_spacing]))
        boundaries = np.flatnonzero(gaps > 3.0 * nominal_spacing) + 1
        result = spectrum.copy()
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, spectrum.size]
        for start, stop in zip(starts, stops):
            segment = spectrum[start:stop]
            if segment.size < 3:
                continue
            padded = np.pad(segment, (1, 1), mode="edge")
            result[start:stop] = np.convolve(padded, weights, mode="valid")
        return result

    @staticmethod
    def cris_sinc_resample(
        source_wavenumber_cm: np.ndarray,
        source_values: np.ndarray,
        target_wavenumber_cm: np.ndarray,
        channel_spacing_cm: float = 0.625,
        half_width_lobes: int = 8,
    ) -> np.ndarray:
        """用CrIS理想sinc仪器线型把过采样仿真卷积到FSR通道。"""
        source_x = np.asarray(source_wavenumber_cm, dtype=float).reshape(-1)
        source_y = np.asarray(source_values, dtype=float).reshape(-1)
        target_x = np.asarray(target_wavenumber_cm, dtype=float).reshape(-1)
        valid = np.isfinite(source_x) & np.isfinite(source_y)
        source_x, source_y = source_x[valid], source_y[valid]
        order = np.argsort(source_x)
        source_x, source_y = source_x[order], source_y[order]
        source_x, unique_indices = np.unique(source_x, return_index=True)
        source_y = source_y[unique_indices]
        if source_x.size < 3:
            raise ValueError("CrIS sinc卷积至少需要三个有效仿真波数点。")
        spacing = float(channel_spacing_cm)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("CrIS通道间隔必须为有限正值。")
        half_width = max(int(half_width_lobes), 1) * spacing
        output = np.full(target_x.shape, np.nan, dtype=float)
        for index, target in enumerate(target_x):
            left = int(np.searchsorted(source_x, target - half_width, side="left"))
            right = int(np.searchsorted(source_x, target + half_width, side="right"))
            local_x = source_x[left:right]
            local_y = source_y[left:right]
            if local_x.size < 3:
                continue
            response = np.sinc((local_x - target) / spacing)
            normalization = float(np.trapezoid(response, local_x))
            if abs(normalization) <= np.finfo(float).eps:
                continue
            output[index] = float(
                np.trapezoid(response * local_y, local_x) / normalization
            )
        return output

    @staticmethod
    def build_per_wavenumber_corrections(
        wavelength_um: np.ndarray,
        factors: np.ndarray,
        selected: np.ndarray,
    ) -> list[dict[str, float]]:
        """将逐通道倍率转换为互不重叠的波长单元，并合并相邻同倍率单元。"""
        wavelength = np.asarray(wavelength_um, dtype=float).reshape(-1)
        factor = np.asarray(factors, dtype=float).reshape(-1)
        mask = np.asarray(selected, dtype=bool).reshape(-1)
        if wavelength.size != factor.size or factor.size != mask.size:
            raise ValueError("逐波数修正的波长、倍率和掩码长度不一致。")
        if wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("逐波数修正要求严格递增且至少包含两个波长。")
        edges = np.empty(wavelength.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
        edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
        edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
        edges[0] = max(edges[0], np.finfo(float).eps)

        corrections: list[dict[str, float]] = []
        for index in np.flatnonzero(mask & np.isfinite(factor) & (factor > 0.0)):
            value = float(factor[index])
            if np.isclose(value, 1.0, rtol=0.0, atol=1.0e-12):
                continue
            lower = float(edges[index])
            upper = float(edges[index + 1])
            # 明确保护CO2中心和长波截止边界。
            if wavelength[index] < 14.8:
                upper = min(upper, 14.8)
            elif wavelength[index] > 15.1:
                lower = max(lower, 15.1)
                upper = min(upper, 15.35)
            if upper <= lower:
                continue
            if (
                corrections
                and np.isclose(corrections[-1]["factor"], value)
                and np.isclose(corrections[-1]["wavelength_max_um"], lower)
            ):
                corrections[-1]["wavelength_max_um"] = upper
            else:
                corrections.append({
                    "wavelength_min_um": lower,
                    "wavelength_max_um": upper,
                    "factor": value,
                })
        return corrections

    def read_numeric_series(self, file_path: str | Path) -> dict[str, np.ndarray]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"光谱文件不存在：{path}")
        extension = path.suffix.lower()
        if extension not in SUPPORTED_SPECTRUM_EXTENSIONS:
            raise ValueError(f"不支持的光谱文件格式：{extension}")
        if extension == ".npz":
            return self._read_npz(path)
        if extension in {".nc", ".nc4"}:
            return self._read_netcdf(path)
        if extension in {".h5", ".hdf5"}:
            return self._read_hdf5(path)
        return self._read_table(path)

    @staticmethod
    def read_sidecar_metadata(file_path: str | Path) -> dict[str, Any]:
        """读取与光谱同名的JSON元数据；没有侧车文件时返回空字典。"""
        path = Path(file_path)
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            return {}
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"光谱元数据必须是JSON对象：{metadata_path}")
        return payload

    def suggest_series(self, series: dict[str, np.ndarray]) -> tuple[str, str]:
        names = list(series)
        if len(names) < 2:
            raise ValueError("光谱文件至少需要两个一维数值序列（光谱坐标和光谱值）。")
        axis = self._match_alias(names, self.AXIS_ALIASES) or names[0]
        remaining = [name for name in names if name != axis]
        value = self._match_alias(remaining, self.VALUE_ALIASES) or remaining[0]
        return axis, value

    def build_spectrum(
        self,
        series: dict[str, np.ndarray],
        axis_name: str,
        value_name: str,
        axis_kind: str = "auto",
    ) -> dict[str, Any]:
        if axis_name == value_name:
            raise ValueError("光谱坐标与光谱值不能使用同一数据序列。")
        if axis_name not in series or value_name not in series:
            raise KeyError("选择的光谱序列不存在。")
        axis = np.asarray(series[axis_name], dtype=float).reshape(-1)
        values = np.asarray(series[value_name], dtype=float).reshape(-1)
        if axis.size != values.size:
            raise ValueError("光谱坐标和光谱值的长度不一致。")
        valid = np.isfinite(axis) & np.isfinite(values)
        axis = axis[valid]
        values = values[valid]
        if axis.size < 2:
            raise ValueError("光谱至少需要两个有效数据点。")

        resolved_kind = self._resolve_axis_kind(axis_name, axis_kind)
        if resolved_kind == "wavenumber_cm":
            positive = axis > 0.0
            wavelength_um = 1.0e4 / axis[positive]
            values = values[positive]
        elif resolved_kind == "wavelength_nm":
            wavelength_um = axis * 1.0e-3
        else:
            wavelength_um = axis

        valid = np.isfinite(wavelength_um) & (wavelength_um > 0.0)
        wavelength_um = wavelength_um[valid]
        values = values[valid]
        order = np.argsort(wavelength_um)
        wavelength_um = wavelength_um[order]
        values = values[order]
        wavelength_um, unique_indices = np.unique(wavelength_um, return_index=True)
        values = values[unique_indices]
        if wavelength_um.size < 2:
            raise ValueError("坐标转换后没有足够的唯一正波长点。")
        return {
            "wavelength_um": wavelength_um,
            "wavenumber_cm": 1.0e4 / wavelength_um,
            "values": values,
            "axis_name": axis_name,
            "value_name": value_name,
            "axis_kind": resolved_kind,
        }

    def from_atmospheric_detector_result(
        self,
        spectrum: dict[str, Any],
        effective_cloud_fraction: float | None = None,
    ) -> dict[str, Any]:
        if not spectrum:
            raise ValueError("当前没有探测器接收的大气辐射传输光谱结果。")
        try:
            wavelength = np.asarray(spectrum["wavelength_um"], dtype=float)
            if effective_cloud_fraction is None:
                values = np.asarray(
                    spectrum["earth_total_spectral_irradiance"], dtype=float
                )
            else:
                fraction = float(effective_cloud_fraction)
                if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                    raise ValueError("有效云量必须位于0～1之间。")
                clear = np.asarray(
                    spectrum["cloud_clear_total_spectral_irradiance"], dtype=float
                )
                overcast = np.asarray(
                    spectrum["cloud_overcast_total_spectral_irradiance"], dtype=float
                )
                values = clear + fraction * (overcast - clear)
        except KeyError as exc:
            raise ValueError("当前大气仿真结果缺少探测器接收的总大气传输波长谱。") from exc
        result = self.build_spectrum(
            {"wavelength_um": wavelength, "earth_total_spectral_irradiance": values},
            "wavelength_um",
            "earth_total_spectral_irradiance",
            "wavelength_um",
        )
        quantity = str(spectrum.get("spectral_quantity", "irradiance"))
        result["spectral_quantity"] = quantity
        result["value_unit"] = (
            "W·m⁻²·sr⁻¹·μm⁻¹" if quantity == "radiance" else "W·m⁻²·μm⁻¹"
        )
        result["spectral_processing"] = str(
            spectrum.get("spectral_processing", "monochromatic_simulation")
        )
        return result

    def fit_effective_cloud_fraction(
        self,
        atmospheric_result: dict[str, Any],
        observed: dict[str, Any],
    ) -> dict[str, Any]:
        """用大气窗口内的晴空/全云端元拟合同足迹有效云量。

        只采用对低层云敏感、受强气体吸收影响较小的8.0–9.2和
        10.2–12.5 μm窗口。存在同足迹NUCAPS云量时将其作为弱先验，
        而不是直接固定采用，避免用云量补偿O3、CO2或水汽带误差。
        """
        try:
            wavelength = np.asarray(
                atmospheric_result["wavelength_um"], dtype=float
            ).reshape(-1)
            clear = np.asarray(
                atmospheric_result["cloud_clear_total_spectral_irradiance"],
                dtype=float,
            ).reshape(-1)
            overcast = np.asarray(
                atmospheric_result["cloud_overcast_total_spectral_irradiance"],
                dtype=float,
            ).reshape(-1)
            observed_wavelength = np.asarray(
                observed["wavelength_um"], dtype=float
            ).reshape(-1)
            observed_values = np.asarray(observed["values"], dtype=float).reshape(-1)
        except KeyError as exc:
            raise ValueError("仿真结果缺少晴空/全云云量拟合端元。") from exc
        if not (
            wavelength.size == clear.size == overcast.size
            and observed_wavelength.size == observed_values.size
        ):
            raise ValueError("云量拟合端元或卫星光谱的数组长度不一致。")
        valid_model = (
            np.isfinite(wavelength)
            & np.isfinite(clear)
            & np.isfinite(overcast)
            & (wavelength > 0.0)
        )
        if np.count_nonzero(valid_model) < 3:
            raise ValueError("晴空/全云端元没有足够的有效光谱点。")
        wavelength = wavelength[valid_model]
        clear = clear[valid_model]
        overcast = overcast[valid_model]
        order = np.argsort(wavelength)
        wavelength, clear, overcast = (
            wavelength[order], clear[order], overcast[order]
        )
        lower = max(float(wavelength[0]), float(np.nanmin(observed_wavelength)))
        upper = min(float(wavelength[-1]), float(np.nanmax(observed_wavelength)))
        selected = (
            np.isfinite(observed_wavelength)
            & np.isfinite(observed_values)
            & (observed_values >= 0.0)
            & (observed_wavelength >= lower)
            & (observed_wavelength <= upper)
        )
        fit_wavelength = observed_wavelength[selected]
        measured = observed_values[selected]
        if fit_wavelength.size < 3:
            raise ValueError("卫星谱与云量端元没有足够的共同通道。")
        fit_wavenumber = 1.0e4 / fit_wavelength
        clear_common = self.three_point_hamming(
            np.interp(fit_wavelength, wavelength, clear), fit_wavenumber
        )
        overcast_common = self.three_point_hamming(
            np.interp(fit_wavelength, wavelength, overcast), fit_wavenumber
        )
        measured = self.three_point_hamming(measured, fit_wavenumber)
        sensitivity = overcast_common - clear_common
        window_mask = np.zeros(fit_wavelength.shape, dtype=bool)
        for window_lower, window_upper in CLOUD_FIT_WINDOWS_UM:
            inside_window = (
                (fit_wavelength >= window_lower)
                & (fit_wavelength <= window_upper)
            )
            # 三点Hamming会使用左右相邻通道。仅保留三个通道都位于同一
            # 大气窗口内的中心点，防止窗口外强吸收带残差从边界泄漏。
            interior = (
                inside_window
                & np.concatenate(([False], inside_window[:-1]))
                & np.concatenate((inside_window[1:], [False]))
            )
            window_mask |= interior
        finite = (
            np.isfinite(clear_common)
            & np.isfinite(overcast_common)
            & np.isfinite(measured)
            & np.isfinite(sensitivity)
            & window_mask
        )
        nonzero = np.abs(sensitivity[finite])
        if nonzero.size < 3 or float(np.max(nonzero)) <= np.finfo(float).eps:
            raise ValueError("晴空与全云端元差异过小，无法拟合有效云量。")
        threshold = max(float(np.percentile(nonzero, 20.0)), np.finfo(float).eps)
        fit_mask = finite & (np.abs(sensitivity) >= threshold)
        delta = sensitivity[fit_mask]
        target = measured[fit_mask] - clear_common[fit_mask]
        denominator = float(np.dot(delta, delta))
        if denominator <= np.finfo(float).eps:
            raise ValueError("有效云量拟合矩阵退化。")
        numerator = float(np.dot(delta, target))
        unconstrained = float(numerator / denominator)
        nucaps_cloud = atmospheric_result.get("nucaps_same_footprint_cloud")
        prior_fraction: float | None = None
        if isinstance(nucaps_cloud, dict):
            candidate = nucaps_cloud.get("fraction")
            if candidate is not None:
                candidate = float(candidate)
                if np.isfinite(candidate) and 0.0 <= candidate <= 1.0:
                    prior_fraction = candidate
        prior_relative_weight = (
            NUCAPS_CLOUD_PRIOR_RELATIVE_WEIGHT
            if prior_fraction is not None else 0.0
        )
        if prior_fraction is not None:
            regularization = prior_relative_weight * denominator
            regularized = float(
                (numerator + regularization * prior_fraction)
                / (denominator + regularization)
            )
        else:
            regularized = unconstrained
        fraction = float(np.clip(regularized, 0.0, 1.0))
        fitted_common = clear_common + fraction * sensitivity
        grid_fraction = float(
            atmospheric_result.get("representative_cloud_fraction", 0.0)
        )
        grid_common = clear_common + np.clip(grid_fraction, 0.0, 1.0) * sensitivity
        fitted_rmse = float(np.sqrt(np.mean(np.square(fitted_common[finite] - measured[finite]))))
        grid_rmse = float(np.sqrt(np.mean(np.square(grid_common[finite] - measured[finite]))))
        simulated_spectrum = self.from_atmospheric_detector_result(
            atmospheric_result, fraction
        )
        return {
            "effective_cloud_fraction": fraction,
            "unconstrained_cloud_fraction": unconstrained,
            "regularized_cloud_fraction": regularized,
            "prior_cloud_fraction": prior_fraction,
            "prior_relative_weight": prior_relative_weight,
            "fit_channel_count": int(np.count_nonzero(fit_mask)),
            "fit_windows_um": [list(window) for window in CLOUD_FIT_WINDOWS_UM],
            "overlap_min_um": lower,
            "overlap_max_um": upper,
            "rmse_fitted": fitted_rmse,
            "rmse_environment_grid_fraction": grid_rmse,
            "environment_grid_fraction": grid_fraction,
            "source": (
                "卫星同足迹大气窗口拟合（NUCAPS弱先验）"
                if prior_fraction is not None
                else "卫星同足迹大气窗口晴空/全云端元拟合"
            ),
            "simulated_spectrum": simulated_spectrum,
        }

    def compare(
        self,
        simulated: dict[str, Any],
        observed: dict[str, Any],
        scaling: str = "absolute",
    ) -> dict[str, Any]:
        sim_x = np.asarray(simulated["wavelength_um"], dtype=float)
        sim_y = np.asarray(simulated["values"], dtype=float)
        obs_x = np.asarray(observed["wavelength_um"], dtype=float)
        obs_y = np.asarray(observed["values"], dtype=float)
        lower = max(float(sim_x.min()), float(obs_x.min()))
        upper = min(float(sim_x.max()), float(obs_x.max()))
        mask = (obs_x >= lower) & (obs_x <= upper)
        common_x = obs_x[mask]
        observed_y = obs_y[mask]
        if common_x.size < 2:
            raise ValueError("仿真光谱与卫星产品没有足够的重叠波段。")
        instrument = str(observed.get("instrument", "")).strip().lower()
        common_wavenumber = np.divide(
            1.0e4,
            common_x,
            out=np.full(common_x.shape, np.nan, dtype=float),
            where=common_x > 0.0,
        )
        spectral_processing = "linear_interpolation + three_point_hamming"
        source_simulated_y = sim_y
        simulated_y = np.interp(common_x, sim_x, sim_y)
        if "cris" in instrument and sim_x.size >= 3 and common_x.size >= 3:
            sim_wavenumber = 1.0e4 / sim_x
            sim_spacing = np.abs(np.diff(np.sort(sim_wavenumber)))
            obs_spacing = np.abs(np.diff(np.sort(common_wavenumber)))
            sim_spacing = sim_spacing[np.isfinite(sim_spacing) & (sim_spacing > 0.0)]
            obs_spacing = obs_spacing[np.isfinite(obs_spacing) & (obs_spacing > 0.0)]
            if sim_spacing.size and obs_spacing.size:
                obs_nominal = float(np.percentile(obs_spacing, 25.0))
                sim_nominal = float(np.percentile(sim_spacing, 50.0))
                # 只有真正过采样的仿真才执行sinc积分；与CrIS同采样率的
                # 数据无法恢复未保存的高分辨率信息，避免伪卷积。
                if sim_nominal <= 0.5 * obs_nominal:
                    convolved = self.cris_sinc_resample(
                        sim_wavenumber,
                        source_simulated_y,
                        common_wavenumber,
                        channel_spacing_cm=obs_nominal,
                    )
                    finite_convolution = np.isfinite(convolved)
                    simulated_y[finite_convolution] = convolved[finite_convolution]
                    spectral_processing = (
                        f"CrIS sinc ILS ({obs_nominal:.6g} cm^-1) + "
                        "[0.23,0.54,0.23] Hamming"
                    )
                else:
                    spectral_processing = (
                        "CrIS同通道采样（未做sinc：仿真网格未过采样） + "
                        "[0.23,0.54,0.23] Hamming"
                    )
        # 两侧采用完全相同的官方三点切趾，且不跨CrIS谱段间隙卷积。
        hamming_coordinate = common_wavenumber if "cris" in instrument else None
        simulated_y = self.three_point_hamming(simulated_y, hamming_coordinate)
        observed_y = self.three_point_hamming(observed_y, hamming_coordinate)
        scale_factor = self._scale_factor(simulated_y, observed_y, scaling, common_x)
        simulated_scaled = simulated_y * scale_factor
        residual = simulated_scaled - observed_y
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        bias = float(np.mean(residual))
        observed_range = float(np.ptp(observed_y))
        nrmse = rmse / observed_range if observed_range > 0.0 else float("nan")
        if np.std(simulated_scaled) > 0.0 and np.std(observed_y) > 0.0:
            correlation = float(np.corrcoef(simulated_scaled, observed_y)[0, 1])
        else:
            correlation = float("nan")
        denominator = float(np.linalg.norm(simulated_scaled) * np.linalg.norm(observed_y))
        cosine = float(np.dot(simulated_scaled, observed_y) / denominator) if denominator > 0.0 else float("nan")
        spectral_angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))) if np.isfinite(cosine) else float("nan")
        return {
            "wavelength_um": common_x,
            "simulated": simulated_scaled,
            "observed": observed_y,
            "residual": residual,
            "metrics": {
                "sample_count": int(common_x.size),
                "overlap_min_um": lower,
                "overlap_max_um": upper,
                "scale_factor": float(scale_factor),
                "rmse": rmse,
                "nrmse": nrmse,
                "mae": mae,
                "bias": bias,
                "correlation": correlation,
                "spectral_angle_deg": spectral_angle_deg,
            },
            "scaling": scaling,
            "spectral_processing": spectral_processing,
        }

    def export_csv(self, result: dict[str, Any], file_path: str | Path) -> Path:
        path = Path(file_path)
        dataframe = pd.DataFrame({
            "wavelength_um": result["wavelength_um"],
            "simulated_atmospheric_detector": result["simulated"],
            "satellite_observed": result["observed"],
            "residual_simulated_minus_observed": result["residual"],
        })
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _read_table(self, path: Path) -> dict[str, np.ndarray]:
        try:
            dataframe = pd.read_csv(path, sep=None, engine="python", comment="#")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"无法读取光谱表格：{exc}") from exc
        if dataframe.columns.size and all(self._is_number(column) for column in dataframe.columns):
            dataframe = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
        numeric: dict[str, np.ndarray] = {}
        for column in dataframe.columns:
            values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
            if np.count_nonzero(np.isfinite(values)) >= 2:
                numeric[str(column)] = values
        if len(numeric) < 2:
            # 无表头的双列产品。
            dataframe = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
            for index in dataframe.columns:
                values = pd.to_numeric(dataframe[index], errors="coerce").to_numpy(dtype=float)
                if np.count_nonzero(np.isfinite(values)) >= 2:
                    numeric[f"column_{int(index) + 1}"] = values
        if len(numeric) < 2:
            raise ValueError("表格中未找到至少两个一维数值序列。")
        return numeric

    def _is_number(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _read_npz(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            series = {
                key: np.asarray(archive[key], dtype=float).reshape(-1)
                for key in archive.files
                if np.asarray(archive[key]).ndim == 1 and np.asarray(archive[key]).size >= 2
            }
        if len(series) < 2:
            raise ValueError("NPZ 中未找到至少两个一维数值数组。")
        return series

    def _read_netcdf(self, path: Path) -> dict[str, np.ndarray]:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("读取 NetCDF 卫星产品需要安装 netCDF4。") from exc
        series: dict[str, np.ndarray] = {}
        with Dataset(path, "r") as dataset:
            self._collect_netcdf_group(dataset, series)
        if len(series) < 2:
            raise ValueError("NetCDF 中未找到至少两个一维数值变量。")
        return series

    def _collect_netcdf_group(self, group: Any, target: dict[str, np.ndarray], prefix: str = "") -> None:
        for name, variable in group.variables.items():
            if getattr(variable, "ndim", 0) == 1 and int(variable.size) >= 2:
                try:
                    target[f"{prefix}{name}"] = np.asarray(variable[:], dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    pass
        for name, child in group.groups.items():
            self._collect_netcdf_group(child, target, f"{prefix}{name}/")

    def _read_hdf5(self, path: Path) -> dict[str, np.ndarray]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("读取 HDF5 卫星产品需要安装 h5py。") from exc
        series: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            def collect(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset) and item.ndim == 1 and item.size >= 2:
                    try:
                        series[name] = np.asarray(item[...], dtype=float).reshape(-1)
                    except (TypeError, ValueError):
                        pass
            handle.visititems(collect)
        if len(series) < 2:
            raise ValueError("HDF5 中未找到至少两个一维数值数据集。")
        return series

    def _resolve_axis_kind(self, name: str, requested: str) -> str:
        if requested != "auto":
            return requested
        normalized = name.lower().replace(" ", "_")
        if "wavenumber" in normalized or normalized in {"wn", "frequency"} or "cm-1" in normalized:
            return "wavenumber_cm"
        if "nm" in normalized:
            return "wavelength_nm"
        return "wavelength_um"

    def _scale_factor(self, simulated: np.ndarray, observed: np.ndarray, scaling: str, axis: np.ndarray) -> float:
        if scaling == "least_squares":
            denominator = float(np.dot(simulated, simulated))
            return float(np.dot(simulated, observed) / denominator) if denominator > 0.0 else 1.0
        if scaling == "peak":
            peak = float(np.max(np.abs(simulated)))
            return float(np.max(np.abs(observed)) / peak) if peak > 0.0 else 1.0
        if scaling == "area":
            integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            sim_area = float(integrate(simulated, axis))
            obs_area = float(integrate(observed, axis))
            return obs_area / sim_area if abs(sim_area) > np.finfo(float).eps else 1.0
        return 1.0

    def _match_alias(self, names: list[str], aliases: tuple[str, ...]) -> str | None:
        normalized = {name.lower().replace(" ", "_"): name for name in names}
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        for alias in aliases:
            for key, original in normalized.items():
                if alias in key:
                    return original
        return None
