from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_SPECTRUM_EXTENSIONS = {".csv", ".txt", ".dat", ".npz", ".nc", ".nc4", ".h5", ".hdf5"}

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
    def three_point_hamming(values: np.ndarray) -> np.ndarray:
        """对一维光谱执行归一化三点Hamming切趾。"""
        spectrum = np.asarray(values, dtype=float).reshape(-1)
        if spectrum.size < 3:
            return spectrum.copy()
        weights = np.hamming(3)
        weights /= np.sum(weights)
        padded = np.pad(spectrum, (1, 1), mode="edge")
        return np.convolve(padded, weights, mode="valid")

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
            "values": values,
            "axis_name": axis_name,
            "value_name": value_name,
            "axis_kind": resolved_kind,
        }

    def from_atmospheric_detector_result(self, spectrum: dict[str, Any]) -> dict[str, Any]:
        if not spectrum:
            raise ValueError("当前没有探测器接收的大气辐射传输光谱结果。")
        try:
            wavelength = np.asarray(spectrum["wavelength_um"], dtype=float)
            values = np.asarray(spectrum["earth_total_spectral_irradiance"], dtype=float)
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
        return result

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
        simulated_y = np.interp(common_x, sim_x, sim_y)
        # 两侧采用完全相同的三点切趾，避免把处理链差异误认为光学厚度误差。
        simulated_y = self.three_point_hamming(simulated_y)
        observed_y = self.three_point_hamming(observed_y)
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
