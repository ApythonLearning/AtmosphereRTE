from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable
import uuid

import numpy as np

from core.atmospheric_radiation_manager import LayeredAtmosphereSolver


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def _cancel_file_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def _write_progress_file(path: str | Path, completed: int) -> None:
    progress_path = Path(path)
    # Windows可能在父进程读取的极短时间内拒绝覆盖，短暂重试即可；父进程容忍旧值。
    for attempt in range(5):
        try:
            progress_path.write_text(str(int(completed)), encoding="ascii")
            return
        except PermissionError:
            if attempt >= 4:
                raise
            time.sleep(0.01)


class HapiCalculationCancelled(RuntimeError):
    """用户取消HAPI分层光学厚度计算。"""


@dataclass(frozen=True)
class GasDefinition:
    name: str
    molecule_id: int
    column_variable: str | None
    mixing_ratio_variable: str
    mixing_ratio_unit: str
    hitran_wavenumber_min_cm: float
    hitran_wavenumber_max_cm: float


ABSORBING_GASES: tuple[GasDefinition, ...] = (
    # 主同位素谱线范围来自HITRANonline；计算网格可以宽于单个气体的线表范围。
    GasDefinition("H2O", 1, "H2O", "H2O_MR", "kg/kg", 0.072, 41_999.696),
    GasDefinition("CO2", 2, None, "CO2", "ppm", 158.302, 17_696.930),
    GasDefinition("O3", 3, "O3", "O3_MR", "ppbv", 0.026, 6_996.681),
    GasDefinition("N2O", 4, "N2O", "N2O_MR", "ppbv", 0.835, 14_017.065),
    GasDefinition("CO", 5, "CO", "CO_MR", "ppbv", 3.705, 14_477.377),
    GasDefinition("CH4", 6, "CH4", "CH4_MR", "ppbv", 0.001, 13_922.613),
    GasDefinition("SO2", 9, "SO2", "SO2_MR", "ppbv", 0.017, 4_159.945),
    GasDefinition("HNO3", 12, "HNO3", "HNO3_MR", "ppbv", 0.007, 4_167.053),
)


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _gas_coefficient_cache_key(task: dict[str, Any]) -> str:
    """生成不含柱浓度的精确吸收截面缓存键。"""
    hapi_path = Path(task["hapi_path"])
    database_dir = Path(task["database_dir"])
    gas_name = str(task["gas_name"])
    table_name = str(task.get("table_name", gas_name))
    metadata = {
        "schema": 1,
        "gas_name": gas_name,
        "table_name": table_name,
        "molecule_id": int(task["molecule_id"]),
        "hapi": _file_signature(hapi_path),
        "line_data": _file_signature(database_dir / f"{table_name}.data"),
        "line_header": _file_signature(database_dir / f"{table_name}.header"),
        "prefilter_lines": bool(task["prefilter_lines"]),
        "prefilter_margin_cm": float(task["prefilter_margin_cm"]),
        "intensity_threshold": float(task["intensity_threshold"]),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in ("wavenumber", "temperature_k", "pressure_hpa"):
        array = np.ascontiguousarray(task[name], dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_single_hapi_table(
    hapi: Any,
    database_dir: Path,
    table_name: str,
    lower: float,
    upper: float,
    prefilter_lines: bool,
    prefilter_margin_cm: float,
) -> tuple[int, int]:
    """只加载当前气体，并在内存中裁掉本次网格之外的谱线。"""
    hapi.VARIABLES["BACKEND_DATABASE_NAME"] = str(database_dir)
    hapi.LOCAL_TABLE_CACHE.clear()
    with redirect_stdout(io.StringIO()):
        hapi.storage2cache(table_name)
    table = hapi.LOCAL_TABLE_CACHE[table_name]
    data = table["data"]
    line_centers = np.asarray(data["nu"], dtype=float)
    original_count = int(line_centers.size)
    if not prefilter_lines or original_count == 0:
        return original_count, original_count

    margin = max(float(prefilter_margin_cm), 0.0)
    selected = np.flatnonzero(
        (line_centers >= float(lower) - margin)
        & (line_centers <= float(upper) + margin)
    )
    if selected.size == original_count:
        return original_count, original_count

    filtered_data: dict[str, Any] = {}
    for name, values in data.items():
        try:
            value_count = len(values)
        except TypeError:
            filtered_data[name] = values
            continue
        if value_count != original_count:
            filtered_data[name] = values
        elif isinstance(values, np.ndarray):
            filtered_data[name] = values[selected]
        else:
            filtered_data[name] = [values[int(index)] for index in selected]
    table["data"] = filtered_data
    table["header"] = dict(table["header"])
    table["header"]["number_of_rows"] = int(selected.size)
    return original_count, int(selected.size)


def _gas_absorption_worker(task: dict[str, Any]) -> dict[str, Any]:
    """子进程入口：计算一种气体的全部大气层并写入临时数组。"""
    started = time.perf_counter()
    gas_name = str(task["gas_name"])
    table_name = str(task.get("table_name", gas_name))
    cancel_path = task.get("cancel_path")
    progress_path = Path(task["progress_path"])
    output_path = Path(task["output_path"])
    wavenumber = np.asarray(task["wavenumber"], dtype=float)
    temperature = np.asarray(task["temperature_k"], dtype=float)
    pressure = np.asarray(task["pressure_hpa"], dtype=float)
    columns = np.asarray(task["columns_molec_cm2"], dtype=float)
    layer_count = int(temperature.size)
    coefficient_layers: np.ndarray | None = None
    cache_hit = False
    original_line_count = 0
    filtered_line_count = 0

    cache_path: Path | None = None
    if bool(task["use_cache"]):
        cache_root = Path(task["cache_dir"])
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / gas_name / f"{_gas_coefficient_cache_key(task)}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    candidate = np.asarray(cached["coefficient_layers"], dtype=np.float64)
                    original_line_count = int(cached["original_line_count"])
                    filtered_line_count = int(cached["filtered_line_count"])
                if candidate.shape == (layer_count, wavenumber.size):
                    coefficient_layers = candidate
                    cache_hit = True
            except (OSError, ValueError, KeyError):
                coefficient_layers = None

    if coefficient_layers is None:
        manager = HapiOpticalDepthManager(
            hapi_path=task["hapi_path"], database_dir=task["database_dir"]
        )
        hapi = manager._load_hapi()
        original_line_count, filtered_line_count = _load_single_hapi_table(
            hapi,
            Path(task["database_dir"]),
            table_name,
            float(wavenumber[0]),
            float(wavenumber[-1]),
            bool(task["prefilter_lines"]),
            float(task["prefilter_margin_cm"]),
        )
        coefficient_layers = np.zeros((layer_count, wavenumber.size), dtype=np.float64)
        for layer_index in range(layer_count):
            if _cancel_file_exists(cancel_path):
                raise HapiCalculationCancelled("HAPI分层光学厚度计算已取消。")
            if columns[layer_index] > 0.0 and np.isfinite(columns[layer_index]):
                with redirect_stdout(io.StringIO()):
                    nu_out, coefficient = hapi.absorptionCoefficient_Voigt(
                        Components=[(int(task["molecule_id"]), 1)],
                        SourceTables=[table_name],
                        Environment={
                            "T": float(temperature[layer_index]),
                            "p": float(pressure[layer_index]) / 1013.25,
                        },
                        WavenumberGrid=wavenumber,
                        HITRAN_units=True,
                        Diluent={"air": 1.0},
                        IntensityThreshold=float(task["intensity_threshold"]),
                    )
                coefficient = np.asarray(coefficient, dtype=float)
                nu_out = np.asarray(nu_out, dtype=float)
                if coefficient.shape != wavenumber.shape or not np.allclose(
                    nu_out,
                    wavenumber,
                    rtol=0.0,
                    atol=max(float(task["wavenumber_step_cm"]) * 1.0e-6, 1.0e-10),
                ):
                    coefficient = np.interp(
                        wavenumber, nu_out, coefficient, left=0.0, right=0.0
                    )
                coefficient_layers[layer_index] = np.maximum(
                    np.nan_to_num(coefficient, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                )
            _write_progress_file(progress_path, layer_index + 1)

        if _cancel_file_exists(cancel_path):
            raise HapiCalculationCancelled("HAPI分层光学厚度计算已取消。")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache = cache_path.with_name(
                f".{cache_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary_cache.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        coefficient_layers=coefficient_layers,
                        original_line_count=np.int64(original_line_count),
                        filtered_line_count=np.int64(filtered_line_count),
                    )
                os.replace(temporary_cache, cache_path)
            finally:
                temporary_cache.unlink(missing_ok=True)
    else:
        _write_progress_file(progress_path, layer_count)

    if _cancel_file_exists(cancel_path):
        raise HapiCalculationCancelled("HAPI分层光学厚度计算已取消。")
    gas_tau = np.maximum(
        np.nan_to_num(
            coefficient_layers * columns[:, None], nan=0.0, posinf=0.0, neginf=0.0
        ),
        0.0,
    )
    np.save(output_path, gas_tau, allow_pickle=False)
    return {
        "gas_name": gas_name,
        "output_path": str(output_path),
        "cache_hit": cache_hit,
        "original_line_count": original_line_count,
        "filtered_line_count": filtered_line_count,
        "elapsed_seconds": time.perf_counter() - started,
    }


@dataclass
class LayeredAtmosphericProfile:
    source_path: Path
    for_index: int
    observation_time_utc: str
    latitude_deg: float
    longitude_deg: float
    quality_flag: int
    altitude_boundaries_km: np.ndarray
    altitude_mid_km: np.ndarray
    pressure_hpa: np.ndarray
    temperature_k: np.ndarray
    gas_columns_molec_cm2: dict[str, np.ndarray]
    gas_sources: dict[str, str]
    source_level_count: int
    source_top_altitude_km: float

    @property
    def gas_names(self) -> list[str]:
        return list(self.gas_columns_molec_cm2)


class NucapsAtmosphericProfileReader:
    """读取NUCAPS廓线并守恒重映射到ARTE Atmosphere的35层网格。"""

    RD_AIR = 287.05
    GRAVITY = 9.80665
    MOLAR_MASS_DRY_AIR = 0.0289647
    MOLAR_MASS_WATER = 0.01801528
    AVOGADRO = 6.02214076e23

    @classmethod
    def inspect(cls, source_path: str | Path) -> dict[str, Any]:
        Dataset = cls._dataset_class()
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"大气廓线文件不存在：{source}")
        with Dataset(source) as dataset:
            cls._validate_dataset(dataset)
            latitude = cls._read_all(dataset, "Latitude")
            longitude = cls._read_all(dataset, "Longitude")
            quality = cls._read_all(dataset, "Quality_Flag")
            time_ms = cls._read_all(dataset, "Time")
            gases = cls._available_gases(dataset)
            valid_counts = np.zeros(latitude.shape, dtype=int)
            pressure = np.ma.asarray(dataset.variables["Pressure"][:], dtype=float).filled(np.nan)
            temperature = np.ma.asarray(dataset.variables["Temperature"][:], dtype=float).filled(np.nan)
            for index in range(latitude.size):
                valid_counts[index] = int(np.count_nonzero(
                    np.isfinite(pressure[index])
                    & (pressure[index] > 0.0)
                    & np.isfinite(temperature[index])
                    & (temperature[index] > 0.0)
                ))
        return {
            "source_path": str(source),
            "profile_count": int(latitude.size),
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "quality_flag": quality,
            "time_ms": time_ms,
            "valid_level_count": valid_counts,
            "gas_names": [gas.name for gas in gases],
        }

    @classmethod
    def nearest_valid_for(
        cls,
        inspection: dict[str, Any],
        latitude_deg: float,
        longitude_deg: float,
    ) -> int:
        latitude = np.asarray(inspection["latitude_deg"], dtype=float)
        longitude = np.asarray(inspection["longitude_deg"], dtype=float)
        quality = np.asarray(inspection["quality_flag"], dtype=float)
        level_count = np.asarray(inspection["valid_level_count"], dtype=int)
        valid = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(quality)
            & (quality == 0)
            & (level_count >= 2)
        )
        if not np.any(valid):
            valid = np.isfinite(latitude) & np.isfinite(longitude) & (level_count >= 2)
        if not np.any(valid):
            raise ValueError("NUCAPS文件中没有可用的大气廓线。")
        distance = cls._great_circle_distance_km(
            float(latitude_deg), float(longitude_deg), latitude, longitude
        )
        distance[~valid] = np.inf
        return int(np.argmin(distance))

    @classmethod
    def profile_summary(cls, inspection: dict[str, Any], for_index: int) -> dict[str, Any]:
        count = int(inspection["profile_count"])
        if for_index < 0 or for_index >= count:
            raise IndexError(f"FOR索引超出范围：{for_index}")
        time_ms = float(np.asarray(inspection["time_ms"], dtype=float)[for_index])
        return {
            "for_index": int(for_index),
            "observation_time_utc": cls._format_time(time_ms),
            "latitude_deg": float(np.asarray(inspection["latitude_deg"])[for_index]),
            "longitude_deg": float(np.asarray(inspection["longitude_deg"])[for_index]),
            "quality_flag": int(np.asarray(inspection["quality_flag"])[for_index]),
            "valid_level_count": int(
                np.asarray(inspection["valid_level_count"])[for_index]
            ),
        }

    @classmethod
    def read(cls, source_path: str | Path, for_index: int) -> LayeredAtmosphericProfile:
        Dataset = cls._dataset_class()
        source = Path(source_path).expanduser().resolve()
        with Dataset(source) as dataset:
            cls._validate_dataset(dataset)
            profile_count = len(dataset.dimensions["Number_of_CrIS_FORs"])
            if for_index < 0 or for_index >= profile_count:
                raise IndexError(f"FOR索引超出范围：{for_index}")

            pressure_all = cls._read_profile_vector(dataset, "Pressure", for_index)
            temperature_all = cls._read_profile_vector(dataset, "Temperature", for_index)
            valid = (
                np.isfinite(pressure_all)
                & (pressure_all > 0.0)
                & np.isfinite(temperature_all)
                & (temperature_all > 0.0)
            )
            if np.count_nonzero(valid) < 2:
                raise ValueError(f"FOR {for_index} 的有效温度—压力层不足。")
            original_indices = np.flatnonzero(valid)
            pressure = pressure_all[valid]
            temperature = temperature_all[valid]
            order = np.argsort(pressure)[::-1]
            original_indices = original_indices[order]
            pressure = pressure[order]
            temperature = temperature[order]

            source_altitude_mid = cls._pressure_level_altitudes(pressure, temperature)
            source_altitude_bounds = cls._cell_boundaries(source_altitude_mid)
            source_pressure_bounds = cls._pressure_boundaries(pressure)
            source_air_column = cls._air_column_from_pressure(source_pressure_bounds)

            target_height = np.asarray(LayeredAtmosphereSolver.LAYER_HEIGHT_KM, dtype=float)
            target_bounds = np.r_[0.0, np.cumsum(target_height)]
            target_mid = 0.5 * (target_bounds[:-1] + target_bounds[1:])
            target_pressure_bounds = cls._target_pressure_boundaries(
                target_bounds,
                source_altitude_mid,
                pressure,
                temperature,
            )
            target_pressure = np.sqrt(
                np.maximum(target_pressure_bounds[:-1], 1.0e-12)
                * np.maximum(target_pressure_bounds[1:], 1.0e-12)
            )
            target_temperature = np.interp(
                target_mid,
                source_altitude_mid,
                temperature,
                left=float(temperature[0]),
                right=np.nan,
            )
            above = ~np.isfinite(target_temperature)
            if np.any(above):
                target_temperature[above] = cls._standard_temperature(target_mid[above])
            target_air_column = cls._air_column_from_pressure(target_pressure_bounds)

            gas_columns: dict[str, np.ndarray] = {}
            gas_sources: dict[str, str] = {}
            for gas in cls._available_gases(dataset):
                mixing_source = cls._mixing_ratio(
                    cls._read_profile_vector(dataset, gas.mixing_ratio_variable, for_index)[
                        original_indices
                    ],
                    gas.mixing_ratio_unit,
                )
                column_source: np.ndarray | None = None
                if gas.column_variable is not None and gas.column_variable in dataset.variables:
                    column_source = cls._read_profile_vector(
                        dataset, gas.column_variable, for_index
                    )[original_indices]
                    usable = np.isfinite(column_source) & (column_source >= 0.0)
                    fallback = mixing_source * source_air_column
                    column_source = np.where(usable, column_source, fallback)
                    source_label = f"NUCAPS {gas.column_variable}逐层分子柱密度"
                    if not np.all(usable):
                        source_label += f"（{np.count_nonzero(~usable)}层由混合比回退）"
                else:
                    column_source = mixing_source * source_air_column
                    source_label = (
                        f"NUCAPS {gas.mixing_ratio_variable}混合比与压力层空气柱密度"
                    )
                finite_column = np.isfinite(column_source) & (column_source >= 0.0)
                if not np.any(finite_column):
                    continue
                column_source = cls._fill_nonnegative(column_source)
                remapped = cls._conservative_remap(
                    source_altitude_bounds,
                    column_source,
                    target_bounds,
                )
                source_top = float(source_altitude_bounds[-1])
                missing_top = target_mid > source_top
                if np.any(missing_top):
                    top_vmr = cls._last_finite_positive(mixing_source)
                    if not np.isfinite(top_vmr):
                        top_vmr = cls._last_finite_positive(
                            column_source / np.maximum(source_air_column, 1.0)
                        )
                    if np.isfinite(top_vmr):
                        remapped[missing_top] += top_vmr * target_air_column[missing_top]
                        source_label += "；廓线顶以上按顶层混合比延拓"
                gas_columns[gas.name] = np.maximum(remapped, 0.0)
                gas_sources[gas.name] = source_label

            if not gas_columns:
                raise ValueError("NUCAPS文件中没有识别到可用于吸收计算的气体廓线。")

            latitude = float(cls._read_all(dataset, "Latitude")[for_index])
            longitude = float(cls._read_all(dataset, "Longitude")[for_index])
            quality = int(cls._read_all(dataset, "Quality_Flag")[for_index])
            time_ms = float(cls._read_all(dataset, "Time")[for_index])

        return LayeredAtmosphericProfile(
            source_path=source,
            for_index=int(for_index),
            observation_time_utc=cls._format_time(time_ms),
            latitude_deg=latitude,
            longitude_deg=longitude,
            quality_flag=quality,
            altitude_boundaries_km=target_bounds,
            altitude_mid_km=target_mid,
            pressure_hpa=target_pressure,
            temperature_k=target_temperature,
            gas_columns_molec_cm2=gas_columns,
            gas_sources=gas_sources,
            source_level_count=int(pressure.size),
            source_top_altitude_km=float(source_altitude_bounds[-1]),
        )

    @staticmethod
    def _dataset_class():
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("读取NUCAPS廓线需要安装netCDF4。") from exc
        return Dataset

    @staticmethod
    def _validate_dataset(dataset: Any) -> None:
        required = {"Pressure", "Temperature", "Latitude", "Longitude", "Quality_Flag", "Time"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"NUCAPS文件缺少必要变量：{', '.join(missing)}")

    @staticmethod
    def _read_all(dataset: Any, name: str) -> np.ndarray:
        return np.ma.asarray(dataset.variables[name][:], dtype=float).filled(np.nan)

    @staticmethod
    def _read_profile_vector(dataset: Any, name: str, for_index: int) -> np.ndarray:
        if name not in dataset.variables:
            return np.full(len(dataset.dimensions["Number_of_P_Levels"]), np.nan)
        return np.ma.asarray(dataset.variables[name][for_index], dtype=float).filled(np.nan)

    @staticmethod
    def _available_gases(dataset: Any) -> list[GasDefinition]:
        available: list[GasDefinition] = []
        for gas in ABSORBING_GASES:
            if gas.mixing_ratio_variable in dataset.variables or (
                gas.column_variable is not None and gas.column_variable in dataset.variables
            ):
                available.append(gas)
        return available

    @classmethod
    def _pressure_level_altitudes(
        cls, pressure_hpa: np.ndarray, temperature_k: np.ndarray
    ) -> np.ndarray:
        altitude = np.zeros(pressure_hpa.size, dtype=float)
        for index in range(1, pressure_hpa.size):
            mean_temperature = 0.5 * (temperature_k[index - 1] + temperature_k[index])
            ratio = max(pressure_hpa[index - 1] / pressure_hpa[index], 1.0)
            altitude[index] = altitude[index - 1] + (
                cls.RD_AIR * mean_temperature / cls.GRAVITY * math.log(ratio) / 1000.0
            )
        return altitude

    @staticmethod
    def _cell_boundaries(midpoints: np.ndarray) -> np.ndarray:
        boundaries = np.empty(midpoints.size + 1, dtype=float)
        boundaries[0] = 0.0
        boundaries[1:-1] = 0.5 * (midpoints[:-1] + midpoints[1:])
        boundaries[-1] = midpoints[-1] + 0.5 * (midpoints[-1] - midpoints[-2])
        return np.maximum.accumulate(boundaries)

    @staticmethod
    def _pressure_boundaries(pressure_hpa: np.ndarray) -> np.ndarray:
        boundaries = np.empty(pressure_hpa.size + 1, dtype=float)
        boundaries[0] = pressure_hpa[0]
        boundaries[1:-1] = np.sqrt(pressure_hpa[:-1] * pressure_hpa[1:])
        log_top = 1.5 * math.log(pressure_hpa[-1]) - 0.5 * math.log(pressure_hpa[-2])
        boundaries[-1] = math.exp(log_top)
        return boundaries

    @classmethod
    def _target_pressure_boundaries(
        cls,
        target_altitude_km: np.ndarray,
        source_altitude_km: np.ndarray,
        pressure_hpa: np.ndarray,
        temperature_k: np.ndarray,
    ) -> np.ndarray:
        log_pressure = np.interp(
            target_altitude_km,
            source_altitude_km,
            np.log(pressure_hpa),
            left=math.log(pressure_hpa[0]),
            right=np.nan,
        )
        above = ~np.isfinite(log_pressure)
        if np.any(above):
            scale_height_km = max(
                cls.RD_AIR * float(temperature_k[-1]) / cls.GRAVITY / 1000.0,
                1.0,
            )
            log_pressure[above] = math.log(pressure_hpa[-1]) - (
                target_altitude_km[above] - source_altitude_km[-1]
            ) / scale_height_km
        return np.exp(log_pressure)

    @classmethod
    def _air_column_from_pressure(cls, pressure_boundaries_hpa: np.ndarray) -> np.ndarray:
        pressure_difference_pa = np.maximum(
            (pressure_boundaries_hpa[:-1] - pressure_boundaries_hpa[1:]) * 100.0,
            0.0,
        )
        return (
            pressure_difference_pa
            * cls.AVOGADRO
            / (cls.GRAVITY * cls.MOLAR_MASS_DRY_AIR)
            / 1.0e4
        )

    @classmethod
    def _mixing_ratio(cls, values: np.ndarray, unit: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if unit == "ppm":
            return array * 1.0e-6
        if unit == "ppbv":
            return array * 1.0e-9
        if unit == "kg/kg":
            denominator = np.maximum(1.0 - array, 1.0e-12)
            return array * cls.MOLAR_MASS_DRY_AIR / (cls.MOLAR_MASS_WATER * denominator)
        return array

    @staticmethod
    def _conservative_remap(
        source_boundaries: np.ndarray,
        source_columns: np.ndarray,
        target_boundaries: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros(target_boundaries.size - 1, dtype=float)
        for source_index, amount in enumerate(source_columns):
            lower = float(source_boundaries[source_index])
            upper = float(source_boundaries[source_index + 1])
            width = upper - lower
            if width <= 0.0 or not np.isfinite(amount) or amount < 0.0:
                continue
            first = max(int(np.searchsorted(target_boundaries, lower, side="right") - 1), 0)
            last = min(int(np.searchsorted(target_boundaries, upper, side="left")), result.size - 1)
            for target_index in range(first, last + 1):
                overlap = max(
                    0.0,
                    min(upper, float(target_boundaries[target_index + 1]))
                    - max(lower, float(target_boundaries[target_index])),
                )
                if overlap > 0.0:
                    result[target_index] += amount * overlap / width
        return result

    @staticmethod
    def _fill_nonnegative(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float).copy()
        valid = np.isfinite(array) & (array >= 0.0)
        if not np.any(valid):
            return np.zeros(array.shape, dtype=float)
        indices = np.arange(array.size)
        array[~valid] = np.interp(indices[~valid], indices[valid], array[valid])
        return np.maximum(array, 0.0)

    @staticmethod
    def _last_finite_positive(values: np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        valid = np.flatnonzero(np.isfinite(array) & (array >= 0.0))
        return float(array[valid[-1]]) if valid.size else float("nan")

    @staticmethod
    def _standard_temperature(altitude_km: np.ndarray) -> np.ndarray:
        return np.interp(
            altitude_km,
            [0, 11, 20, 32, 47, 51, 71, 86, 100],
            [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87, 195.08],
        )

    @staticmethod
    def _great_circle_distance_km(
        target_latitude: float,
        target_longitude: float,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:
        lat1 = np.deg2rad(float(target_latitude))
        lon1 = np.deg2rad(float(target_longitude))
        lat2 = np.deg2rad(latitude)
        lon2 = np.deg2rad(longitude)
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        value = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))

    @staticmethod
    def _format_time(time_ms: float) -> str:
        if not np.isfinite(time_ms):
            return ""
        return datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )


class LayeredAtmosphericProfileCsvReader:
    """Read an already normalized ARTE 35-layer representative profile."""

    @classmethod
    def inspect(cls, source_path: str | Path) -> dict[str, Any]:
        profile = cls.read(source_path, 0)
        return {
            "source_path": str(profile.source_path),
            "source_type": "ARTE_35_LAYER_CSV",
            "profile_count": 1,
            "latitude_deg": np.asarray([profile.latitude_deg]),
            "longitude_deg": np.asarray([profile.longitude_deg]),
            "quality_flag": np.asarray([profile.quality_flag]),
            "time_ms": np.asarray([np.nan]),
            "observation_time_utc": [profile.observation_time_utc],
            "valid_level_count": np.asarray([profile.source_level_count]),
            "gas_names": profile.gas_names,
        }

    @classmethod
    def read(cls, source_path: str | Path, for_index: int = 0) -> LayeredAtmosphericProfile:
        source = Path(source_path).expanduser().resolve()
        if int(for_index) != 0:
            raise IndexError("35层CSV廓线只有索引0。")
        if not source.is_file():
            raise FileNotFoundError(f"35层大气廓线不存在：{source}")
        with source.open("r", encoding="utf-8-sig") as stream:
            header = stream.readline().strip()
        columns = tuple(item.strip() for item in header.split(","))
        values = np.loadtxt(source, delimiter=",", skiprows=1, ndmin=2)
        expected_layers = len(LayeredAtmosphereSolver.LAYER_HEIGHT_KM)
        if values.shape != (expected_layers, len(columns)):
            raise ValueError(
                f"HAPI代表廓线必须是{expected_layers}层，且数据列须与表头一致。"
            )
        indices = {name: index for index, name in enumerate(columns)}
        required = {"pressure(hPa)", "temperature(K)"}
        if not required.issubset(indices):
            raise ValueError("35层CSV缺少 pressure(hPa) 或 temperature(K)。")
        pressure = np.asarray(values[:, indices["pressure(hPa)"]], dtype=float)
        temperature = np.asarray(values[:, indices["temperature(K)"]], dtype=float)
        if (
            not np.isfinite(pressure).all()
            or not np.isfinite(temperature).all()
            or np.any(pressure <= 0.0)
            or np.any(temperature <= 0.0)
        ):
            raise ValueError("35层CSV包含无效温度或气压。")
        heights = np.asarray(LayeredAtmosphereSolver.LAYER_HEIGHT_KM, dtype=float)
        boundaries = np.r_[0.0, np.cumsum(heights)]
        default_mid = 0.5 * (boundaries[:-1] + boundaries[1:])
        altitude = (
            np.asarray(values[:, indices["altitude_mid(km)"]], dtype=float)
            if "altitude_mid(km)" in indices
            else default_mid
        )
        gas_columns: dict[str, np.ndarray] = {}
        for name, index in indices.items():
            if not name.startswith("column_") or "(" not in name:
                continue
            gas_name = name[len("column_") : name.index("(")].strip().upper()
            if gas_name not in {gas.name for gas in ABSORBING_GASES}:
                continue
            column = np.asarray(values[:, index], dtype=float)
            if np.isfinite(column).all() and np.all(column >= 0.0):
                gas_columns[gas_name] = column
        if not gas_columns:
            raise ValueError("35层CSV没有可用于HAPI的气体分子柱密度列。")

        metadata: dict[str, Any] = {}
        sidecar = source.with_suffix(".json")
        if sidecar.is_file():
            try:
                metadata = dict(json.loads(sidecar.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                metadata = {}
        return LayeredAtmosphericProfile(
            source_path=source,
            for_index=0,
            observation_time_utc=str(metadata.get("observation_time_utc", "")),
            latitude_deg=float(metadata.get("latitude_deg", np.nan)),
            longitude_deg=float(metadata.get("longitude_deg", np.nan)),
            quality_flag=int(metadata.get("quality_flag", 0)),
            altitude_boundaries_km=boundaries,
            altitude_mid_km=altitude,
            pressure_hpa=pressure,
            temperature_k=temperature,
            gas_columns_molec_cm2=gas_columns,
            gas_sources={
                name: "大气廓线模式学习导出的35层分子柱密度"
                for name in gas_columns
            },
            source_level_count=expected_layers,
            source_top_altitude_km=float(boundaries[-1]),
        )

    @staticmethod
    def nearest_valid_for(
        inspection: dict[str, Any], latitude_deg: float, longitude_deg: float
    ) -> int:
        del inspection, latitude_deg, longitude_deg
        return 0

    @staticmethod
    def profile_summary(inspection: dict[str, Any], for_index: int) -> dict[str, Any]:
        if int(for_index) != 0:
            raise IndexError("35层CSV廓线只有索引0。")
        return {
            "for_index": 0,
            "observation_time_utc": str(
                list(inspection.get("observation_time_utc", [""]))[0]
            ),
            "latitude_deg": float(np.asarray(inspection["latitude_deg"])[0]),
            "longitude_deg": float(np.asarray(inspection["longitude_deg"])[0]),
            "quality_flag": int(np.asarray(inspection["quality_flag"])[0]),
            "valid_level_count": int(np.asarray(inspection["valid_level_count"])[0]),
        }


class AtmosphericProfileReader:
    """Dispatch NUCAPS products and learned 35-layer CSV modes."""

    @staticmethod
    def reader_for(source_path: str | Path) -> Any:
        source = Path(source_path)
        if source.suffix.lower() == ".csv":
            return LayeredAtmosphericProfileCsvReader
        return NucapsAtmosphericProfileReader

    @classmethod
    def inspect(cls, source_path: str | Path) -> dict[str, Any]:
        return cls.reader_for(source_path).inspect(source_path)

    @classmethod
    def read(cls, source_path: str | Path, for_index: int) -> LayeredAtmosphericProfile:
        return cls.reader_for(source_path).read(source_path, for_index)

    @classmethod
    def nearest_valid_for(
        cls,
        inspection: dict[str, Any],
        latitude_deg: float,
        longitude_deg: float,
    ) -> int:
        source = str(inspection.get("source_path", ""))
        return cls.reader_for(source).nearest_valid_for(
            inspection, latitude_deg, longitude_deg
        )

    @classmethod
    def profile_summary(
        cls, inspection: dict[str, Any], for_index: int
    ) -> dict[str, Any]:
        source = str(inspection.get("source_path", ""))
        return cls.reader_for(source).profile_summary(inspection, for_index)


class HapiOpticalDepthManager:
    """调用项目自带HAPI，生成求解器可直接使用的35层气体吸收光学厚度。"""

    DEFAULT_WAVENUMBER_MIN_CM = 500.0
    DEFAULT_WAVENUMBER_MAX_CM = 33_300.0
    DEFAULT_WAVENUMBER_STEP_CM = 0.5
    DOWNLOAD_CHUNK_WIDTH_CM = 5_000.0
    DOWNLOAD_MIN_CHUNK_WIDTH_CM = 250.0
    DOWNLOAD_MAX_SPLIT_DEPTH = 6
    DOWNLOAD_MAX_ATTEMPTS = 4
    DOWNLOAD_SPLIT_AFTER_ATTEMPTS = 2
    DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 1.0
    # 单张谱线表可达数百MB；默认两个并发兼顾速度与内存，用户可手动提高到4。
    DEFAULT_MAX_WORKERS = min(2, max((os.cpu_count() or 2) - 1, 1))
    DEFAULT_PREFILTER_MARGIN_CM = 100.0

    def __init__(
        self,
        hapi_path: str | Path | None = None,
        database_dir: str | Path | None = None,
        table_sources: dict[str, str | Path] | None = None,
    ) -> None:
        application_dir = Path(__file__).resolve().parents[1]
        self.hapi_path = Path(hapi_path or application_dir / "thirdParty" / "hapi.py").resolve()
        self.database_dir = Path(
            database_dir or application_dir / "resources" / "data" / "gas_absorption"
        ).resolve()
        self.profile_reader = AtmosphericProfileReader()
        self._hapi: Any | None = None
        self._table_sources: dict[str, Path] = {}
        self.set_table_sources(table_sources or {})

    def set_table_sources(self, sources: dict[str, str | Path]) -> None:
        """Restore external HAPI table pairs without copying their contents."""
        known_gases = {gas.name.upper(): gas.name for gas in ABSORBING_GASES}
        restored: dict[str, Path] = {}
        for raw_name, raw_path in dict(sources).items():
            gas_name = known_gases.get(str(raw_name).upper())
            if gas_name is None or not str(raw_path).strip():
                continue
            data_path = Path(raw_path).expanduser().resolve()
            if data_path.suffix.lower() == ".header":
                data_path = data_path.with_suffix(".data")
            elif data_path.suffix.lower() != ".data":
                data_path = Path(f"{data_path}.data")
            if self._valid_table_files(data_path, data_path.with_suffix(".header")):
                restored[gas_name] = data_path
        self._table_sources = restored

    def export_table_sources(self) -> dict[str, str]:
        """Return serializable external table references for project persistence."""
        return {
            name: str(path)
            for name, path in sorted(self._table_sources.items())
            if self._valid_table_files(path, path.with_suffix(".header"))
        }

    def line_database_source(self, name: str) -> tuple[Path, str]:
        """Return the directory and HAPI table name used for one gas."""
        external_data = self._table_sources.get(name)
        if external_data is not None and self._valid_table_files(
            external_data, external_data.with_suffix(".header")
        ):
            return external_data.parent, external_data.stem
        return self.database_dir, name

    def line_database_status(self, gas_names: list[str]) -> dict[str, bool]:
        return {
            name: self._valid_local_table(name)
            for name in gas_names
        }

    def import_line_tables(
        self,
        sources: list[str | Path] | tuple[str | Path, ...],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Import local HAPI ``.data/.header`` pairs into the line database.

        A directory is expanded to all header files directly inside it. File
        inputs may point to either member of a pair; the sibling is discovered
        automatically. All pairs are validated before any destination is
        changed.
        """
        source_paths: list[Path] = []
        for value in sources:
            path = Path(value).expanduser().resolve()
            if path.is_dir():
                headers = sorted(path.glob("*.header"))
                header_stems = {item.stem.lower() for item in headers}
                source_paths.extend(headers)
                source_paths.extend(
                    item
                    for item in sorted(path.glob("*.par"))
                    if item.stem.lower() not in header_stems
                )
            elif path.is_file():
                source_paths.append(path)
            else:
                raise FileNotFoundError(f"本地HITRAN数据不存在：{path}")
        if not source_paths:
            raise ValueError("未找到可导入的HITRAN .data/.header文件对。")

        known_gases = {gas.name.upper(): gas.name for gas in ABSORBING_GASES}
        validated: dict[str, tuple[Path, Path, dict[str, Any], bool]] = {}
        visited_stems: set[tuple[Path, str]] = set()
        for source in source_paths:
            if source.suffix.lower() not in {".data", ".header", ".par"}:
                raise ValueError(f"不支持的HITRAN文件类型：{source.name}")
            stem_key = (source.parent, source.stem.lower())
            if stem_key in visited_stems:
                continue
            visited_stems.add(stem_key)
            raw_par = source.suffix.lower() == ".par"
            data_path = source if raw_par else source.with_suffix(".data")
            header_path = source.with_suffix(".header")
            if not data_path.is_file() or (not raw_par and not header_path.is_file()):
                raise ValueError(
                    f"HAPI文件必须成对存在：{data_path.name} 与 {header_path.name}；"
                    "或直接选择原始HITRAN .par文件。"
                )
            if data_path.stat().st_size <= 0:
                raise ValueError(f"HITRAN谱线数据为空：{data_path}")
            if header_path.is_file():
                try:
                    header = json.loads(header_path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError, TypeError) as exc:
                    raise ValueError(f"HITRAN头文件无效：{header_path}") from exc
            else:
                hapi = self._load_hapi()
                header = json.loads(json.dumps(hapi.HITRAN_DEFAULT_HEADER))
            first_line = ""
            line_count = 0
            try:
                reported_count = int(header.get("number_of_rows", 0))
            except (TypeError, ValueError):
                reported_count = 0
            with data_path.open("r", encoding="ascii", errors="ignore") as stream:
                for line in stream:
                    if line.strip():
                        line_count += 1
                        if not first_line:
                            first_line = line
                        if reported_count > 0:
                            line_count = reported_count
                            break
            if line_count <= 0:
                raise ValueError(f"HITRAN谱线数据没有有效记录：{data_path}")
            table_name = str(header.get("table_name", source.stem)).strip().upper()
            gas_name = known_gases.get(table_name)
            if gas_name is None:
                gas_name = known_gases.get(source.stem.upper())
            if gas_name is None and len(first_line) >= 2:
                try:
                    molecule_id = int(first_line[:2])
                except ValueError:
                    molecule_id = -1
                gas_name = next(
                    (
                        gas.name
                        for gas in ABSORBING_GASES
                        if gas.molecule_id == molecule_id
                    ),
                    None,
                )
            if gas_name is None:
                raise ValueError(
                    f"无法识别气体“{header.get('table_name', source.stem)}”；"
                    f"支持：{', '.join(gas.name for gas in ABSORBING_GASES)}。"
                )
            if gas_name in validated:
                raise ValueError(f"一次导入中包含多组{gas_name}谱线文件。")
            normalized_header = dict(header)
            normalized_header["table_name"] = gas_name
            normalized_header["number_of_rows"] = line_count
            normalized_header["size_in_bytes"] = int(data_path.stat().st_size)
            validated[gas_name] = (data_path, header_path, normalized_header, raw_par)

        existing = [
            name
            for name, (data_path, _header_path, _header, raw_par) in validated.items()
            if self._valid_local_table(name)
            and (raw_par or self._table_sources.get(name) != data_path)
        ]
        if existing and not overwrite:
            raise FileExistsError(
                "本地数据库已存在以下气体：" + "、".join(existing)
            )

        raw_tables = {
            gas_name: values
            for gas_name, values in validated.items()
            if values[3]
        }
        if raw_tables:
            self.database_dir.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix="hapi_import_", dir=self.database_dir)
            )
            try:
                for gas_name, (data_path, _header_path, header, _raw_par) in raw_tables.items():
                    shutil.copy2(data_path, staging / f"{gas_name}.data")
                    (staging / f"{gas_name}.header").write_text(
                        json.dumps(header, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                for gas_name in raw_tables:
                    (staging / f"{gas_name}.data").replace(
                        self.database_dir / f"{gas_name}.data"
                    )
                    (staging / f"{gas_name}.header").replace(
                        self.database_dir / f"{gas_name}.header"
                    )
                    self._table_sources.pop(gas_name, None)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        referenced = sorted(set(validated) - set(raw_tables))
        for gas_name in referenced:
            self._table_sources[gas_name] = validated[gas_name][0]

        imported = sorted(validated)
        invalid = [name for name in imported if not self._valid_local_table(name)]
        if invalid:
            raise RuntimeError("导入后校验失败：" + "、".join(invalid))
        return {
            "imported_gases": imported,
            "database_dir": str(self.database_dir),
            "overwritten_gases": sorted(existing),
            "referenced_gases": referenced,
            "copied_gases": sorted(raw_tables),
            "source_directories": sorted(
                {str(self._table_sources[name].parent) for name in referenced}
            ),
        }

    def calculate(
        self,
        profile_path: str | Path,
        for_index: int,
        wavenumber_min_cm: float,
        wavenumber_max_cm: float,
        wavenumber_step_cm: float,
        output_root: str | Path,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
        max_workers: int | None = None,
        use_cache: bool = True,
        prefilter_lines: bool = True,
        save_components: bool = True,
        calculation_mode: str = "standard",
        intensity_threshold: float = 0.0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        lower, upper, step = self._validate_grid(
            wavenumber_min_cm, wavenumber_max_cm, wavenumber_step_cm
        )
        worker_count = max(1, min(int(max_workers or self.DEFAULT_MAX_WORKERS), 4))
        threshold = float(intensity_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("谱线强度阈值必须是非负有限数值。")
        progress = progress or (lambda _value, _maximum, _message: None)
        cancelled = cancelled or (lambda: False)
        self._check_cancelled(cancelled)
        progress(0, 1, "正在读取35层大气廓线…")
        profile_started = time.perf_counter()
        profile = self.profile_reader.read(profile_path, int(for_index))
        profile_seconds = time.perf_counter() - profile_started
        definitions = [gas for gas in ABSORBING_GASES if gas.name in profile.gas_names]
        if not definitions:
            raise ValueError("所选廓线中没有可计算的气相吸收组分。")
        active_definitions = [
            gas for gas in definitions if self._hitran_query_range(gas, lower, upper) is not None
        ]
        active_gas_names = {gas.name for gas in active_definitions}

        self._check_cancelled(cancelled)
        progress(0, 1, "正在检查HITRAN谱线数据库…")
        download_started = time.perf_counter()
        downloaded = self._ensure_line_tables(
            active_definitions, lower, upper, progress, cancelled
        )
        database_seconds = time.perf_counter() - download_started

        wavenumber = np.arange(lower, upper, step, dtype=float)
        if wavenumber.size == 0 or wavenumber[-1] < upper - max(step * 1.0e-10, 1.0e-12):
            wavenumber = np.r_[wavenumber, upper]
        else:
            wavenumber[-1] = upper
        layer_count = profile.altitude_mid_km.size
        total_operations = len(definitions) * layer_count
        total_tau = np.zeros((layer_count, wavenumber.size), dtype=np.float64)
        components: dict[str, np.ndarray] = {}
        output_root_path = Path(output_root).expanduser().resolve()
        output_root_path.mkdir(parents=True, exist_ok=True)
        cache_dir = output_root_path / ".coefficient_cache"
        compute_dir = Path(
            tempfile.mkdtemp(prefix=".compute_", dir=output_root_path)
        )
        internal_cancel_path = compute_dir / "cancel.flag"
        progress_paths: dict[str, Path] = {}
        gas_diagnostics: list[dict[str, Any]] = []
        compute_started = time.perf_counter()
        executor: ProcessPoolExecutor | None = None
        try:
            active_worker_count = min(worker_count, max(len(active_definitions), 1))
            executor = ProcessPoolExecutor(
                max_workers=active_worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            )
            future_to_gas: dict[Any, GasDefinition] = {}
            for gas in definitions:
                if gas.name not in active_gas_names:
                    zero_tau = np.zeros(total_tau.shape, dtype=np.float32)
                    if save_components:
                        components[gas.name] = zero_tau
                    continue
                gas_progress_path = compute_dir / f"{gas.name}.progress"
                gas_output_path = compute_dir / f"{gas.name}.npy"
                progress_paths[gas.name] = gas_progress_path
                table_database_dir, table_name = self.line_database_source(gas.name)
                task = {
                    "gas_name": gas.name,
                    "table_name": table_name,
                    "molecule_id": gas.molecule_id,
                    "hapi_path": str(self.hapi_path),
                    "database_dir": str(table_database_dir),
                    "cache_dir": str(cache_dir),
                    "wavenumber": wavenumber,
                    "wavenumber_step_cm": step,
                    "temperature_k": profile.temperature_k,
                    "pressure_hpa": profile.pressure_hpa,
                    "columns_molec_cm2": profile.gas_columns_molec_cm2[gas.name],
                    "use_cache": bool(use_cache),
                    "prefilter_lines": bool(prefilter_lines),
                    "prefilter_margin_cm": self.DEFAULT_PREFILTER_MARGIN_CM,
                    "intensity_threshold": threshold,
                    "cancel_path": str(internal_cancel_path),
                    "progress_path": str(gas_progress_path),
                    "output_path": str(gas_output_path),
                }
                future_to_gas[executor.submit(_gas_absorption_worker, task)] = gas

            pending = set(future_to_gas)
            completed_layers: dict[str, int] = {
                gas.name: (0 if gas.name in active_gas_names else layer_count)
                for gas in definitions
            }
            while pending:
                if cancelled():
                    internal_cancel_path.touch(exist_ok=True)
                    raise HapiCalculationCancelled("HAPI分层光学厚度计算已取消。")
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for gas_name, gas_progress_path in progress_paths.items():
                    try:
                        completed_layers[gas_name] = min(
                            int(gas_progress_path.read_text(encoding="ascii")), layer_count
                        )
                    except (OSError, ValueError):
                        pass
                completed_count = sum(completed_layers.values())
                active_text = "、".join(
                    future_to_gas[future].name for future in pending
                )
                progress(
                    completed_count,
                    total_operations,
                    f"正在并行计算气体分层吸收：{active_text or '正在汇总结果'}",
                )
                for future in done:
                    diagnostic = dict(future.result())
                    gas_name = str(diagnostic["gas_name"])
                    gas_tau = np.load(str(diagnostic["output_path"]), allow_pickle=False)
                    total_tau += np.asarray(gas_tau, dtype=np.float64)
                    if save_components:
                        components[gas_name] = np.asarray(gas_tau, dtype=np.float32)
                    completed_layers[gas_name] = layer_count
                    gas_diagnostics.append(diagnostic)
        except BaseException:
            internal_cancel_path.touch(exist_ok=True)
            if executor is not None:
                for future in locals().get("future_to_gas", {}):
                    future.cancel()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            compute_seconds = time.perf_counter() - compute_started
            shutil.rmtree(compute_dir, ignore_errors=True)

        self._check_cancelled(cancelled)
        progress(total_operations, total_operations, "正在保存逐层光学厚度和计算记录…")
        save_started = time.perf_counter()
        try:
            performance = {
                "profile_read_seconds": profile_seconds,
                "line_database_seconds": database_seconds,
                "gas_computation_seconds": compute_seconds,
                "save_seconds": 0.0,
                "total_seconds": 0.0,
                "gas_tasks": sorted(gas_diagnostics, key=lambda item: item["gas_name"]),
            }
            optimization = {
                "calculation_mode": str(calculation_mode),
                "process_isolation": True,
                "max_workers": min(worker_count, max(len(active_definitions), 1)),
                "coefficient_cache_enabled": bool(use_cache),
                "coefficient_cache_hits": sum(
                    bool(item.get("cache_hit")) for item in gas_diagnostics
                ),
                "line_prefilter_enabled": bool(prefilter_lines),
                "line_prefilter_margin_cm-1": self.DEFAULT_PREFILTER_MARGIN_CM,
                "intensity_threshold": threshold,
                "component_output_enabled": bool(save_components),
            }
            result = self._save_result(
                output_root_path,
                profile,
                wavenumber,
                total_tau,
                components,
                lower,
                upper,
                step,
                downloaded,
                save_components=save_components,
                performance=performance,
                optimization=optimization,
            )
            performance["save_seconds"] = time.perf_counter() - save_started
            performance["total_seconds"] = time.perf_counter() - started
            manifest_path = Path(result["manifest_file"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["performance"] = performance
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["performance"] = performance
            result["optimization"] = optimization
        finally:
            shutil.rmtree(compute_dir, ignore_errors=True)
        progress(total_operations, total_operations, "分层吸收光学厚度计算完成。")
        return result

    def _ensure_line_tables(
        self,
        definitions: list[GasDefinition],
        lower: float,
        upper: float,
        progress: ProgressCallback,
        cancelled: CancelCallback,
    ) -> list[str]:
        self.database_dir.mkdir(parents=True, exist_ok=True)
        missing = [gas for gas in definitions if not self._valid_local_table(gas.name)]
        if not missing:
            return []
        if not self.hapi_path.exists():
            raise FileNotFoundError(f"找不到HAPI脚本：{self.hapi_path}")
        hapi = self._load_hapi()
        # 旧版HAPI默认使用http地址并依赖重定向。直接使用HTTPS可少一次连接跳转，
        # 在部分Windows/OpenSSL环境下能显著减少ASN1/NOT_ENOUGH_DATA中断。
        variables = getattr(hapi, "VARIABLES", None)
        if isinstance(variables, dict):
            variables["GLOBAL_HOST"] = "https://hitran.org"
        downloaded: list[str] = []
        temporary_dir = Path(tempfile.mkdtemp(prefix="hapi_download_", dir=self.database_dir))
        try:
            with redirect_stdout(io.StringIO()):
                hapi.db_begin(str(temporary_dir))
            for index, gas in enumerate(missing, start=1):
                self._check_cancelled(cancelled)
                query_range = self._hitran_query_range(gas, lower, upper)
                if query_range is None:
                    continue
                query_lower, query_upper = query_range
                data_path, header_path = self._download_line_table(
                    hapi,
                    temporary_dir,
                    gas,
                    query_lower,
                    query_upper,
                    index - 1,
                    len(missing),
                    progress,
                    cancelled,
                )
                data_path.replace(self.database_dir / data_path.name)
                header_path.replace(self.database_dir / header_path.name)
                downloaded.append(gas.name)
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        return downloaded

    @staticmethod
    def _hitran_query_range(
        gas: GasDefinition, requested_lower: float, requested_upper: float
    ) -> tuple[float, float] | None:
        """把公共计算范围裁剪到指定气体主同位素的实际线表范围。"""
        lower = max(float(requested_lower), gas.hitran_wavenumber_min_cm)
        upper = min(float(requested_upper), gas.hitran_wavenumber_max_cm)
        if upper <= lower:
            return None
        return lower, upper

    def _download_line_table(
        self,
        hapi: Any,
        temporary_dir: Path,
        gas: GasDefinition,
        lower: float,
        upper: float,
        gas_index: int,
        gas_count: int,
        progress: ProgressCallback,
        cancelled: CancelCallback,
    ) -> tuple[Path, Path]:
        """分段下载并合并谱线，避免大请求因TLS连接中断而整体失败。"""
        chunk_width = max(float(self.DOWNLOAD_CHUNK_WIDTH_CM), 1.0)
        chunk_bounds: list[tuple[float, float]] = []
        chunk_lower = lower
        while chunk_lower < upper:
            chunk_upper = min(chunk_lower + chunk_width, upper)
            chunk_bounds.append((chunk_lower, chunk_upper))
            chunk_lower = chunk_upper
        if not chunk_bounds:
            chunk_bounds.append((lower, upper))

        part_paths: list[tuple[Path, Path]] = []
        for part_index, (part_lower, part_upper) in enumerate(chunk_bounds, start=1):
            self._check_cancelled(cancelled)
            table_name = f"_stirs_{gas.name}_{part_index:03d}"
            downloaded_parts = self._download_line_table_part_adaptive(
                hapi,
                temporary_dir,
                table_name,
                gas,
                part_lower,
                part_upper,
                part_index,
                len(chunk_bounds),
                gas_index,
                gas_count,
                progress,
                cancelled,
            )
            part_paths.extend(downloaded_parts)

        data_path = temporary_dir / f"{gas.name}.data"
        header_path = temporary_dir / f"{gas.name}.header"
        line_count = self._merge_downloaded_parts(part_paths, data_path)
        if line_count <= 0:
            raise RuntimeError(
                f"{gas.name}在{lower:g}–{upper:g} cm⁻¹范围内未下载到任何谱线。"
            )
        try:
            header = json.loads(part_paths[0][1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{gas.name}谱线下载头文件无效：{exc}") from exc
        header["table_name"] = gas.name
        header["number_of_rows"] = line_count
        header["size_in_bytes"] = data_path.stat().st_size
        header["comment"] = (
            f"Contains lines for HITRAN molecule {gas.molecule_id}, isotopologue 1"
            f"\n in {lower:.3f}-{upper:.3f} wavenumber range"
        )
        header_path.write_text(json.dumps(header, indent=2), encoding="utf-8")
        return data_path, header_path

    def _download_line_table_part_adaptive(
        self,
        hapi: Any,
        temporary_dir: Path,
        table_name: str,
        gas: GasDefinition,
        lower: float,
        upper: float,
        part_index: int,
        part_count: int,
        gas_index: int,
        gas_count: int,
        progress: ProgressCallback,
        cancelled: CancelCallback,
        split_depth: int = 0,
    ) -> list[tuple[Path, Path]]:
        """下载一个片段；TLS截断时自动二分，直到响应足够小。"""
        can_split = (
            upper - lower > float(self.DOWNLOAD_MIN_CHUNK_WIDTH_CM)
            and split_depth < int(self.DOWNLOAD_MAX_SPLIT_DEPTH)
        )
        attempt_limit = (
            min(int(self.DOWNLOAD_SPLIT_AFTER_ATTEMPTS), int(self.DOWNLOAD_MAX_ATTEMPTS))
            if can_split
            else int(self.DOWNLOAD_MAX_ATTEMPTS)
        )
        try:
            self._download_line_table_part(
                hapi,
                temporary_dir,
                table_name,
                gas,
                lower,
                upper,
                part_index,
                part_count,
                gas_index,
                gas_count,
                progress,
                cancelled,
                attempt_limit=attempt_limit,
            )
            return [
                (
                    temporary_dir / f"{table_name}.data",
                    temporary_dir / f"{table_name}.header",
                )
            ]
        except RuntimeError as exc:
            if not can_split or not self._is_interrupted_transfer_error(exc):
                raise

        midpoint = lower + (upper - lower) / 2.0
        progress(
            gas_index,
            gas_count,
            f"{gas.name}谱线片段{part_index}/{part_count}传输不完整，"
            f"自动拆分为{lower:g}–{midpoint:g}和{midpoint:g}–{upper:g} cm⁻¹…",
        )
        result: list[tuple[Path, Path]] = []
        for suffix, child_lower, child_upper in (
            ("a", lower, midpoint),
            ("b", midpoint, upper),
        ):
            result.extend(
                self._download_line_table_part_adaptive(
                    hapi,
                    temporary_dir,
                    f"{table_name}_{suffix}",
                    gas,
                    child_lower,
                    child_upper,
                    part_index,
                    part_count,
                    gas_index,
                    gas_count,
                    progress,
                    cancelled,
                    split_depth + 1,
                )
            )
        return result

    def _download_line_table_part(
        self,
        hapi: Any,
        temporary_dir: Path,
        table_name: str,
        gas: GasDefinition,
        lower: float,
        upper: float,
        part_index: int,
        part_count: int,
        gas_index: int,
        gas_count: int,
        progress: ProgressCallback,
        cancelled: CancelCallback,
        attempt_limit: int | None = None,
    ) -> None:
        last_error: Exception | None = None
        maximum_attempts = max(
            1,
            int(self.DOWNLOAD_MAX_ATTEMPTS if attempt_limit is None else attempt_limit),
        )
        for attempt in range(1, maximum_attempts + 1):
            self._check_cancelled(cancelled)
            self._remove_temporary_hapi_table(hapi, temporary_dir, table_name)
            attempt_text = "" if attempt == 1 else f"，第{attempt}次尝试"
            progress(
                gas_index,
                gas_count,
                f"正在下载{gas.name}谱线：片段{part_index}/{part_count}"
                f"（{lower:g}–{upper:g} cm⁻¹{attempt_text}）…",
            )
            try:
                with redirect_stdout(io.StringIO()):
                    hapi.fetch(table_name, gas.molecule_id, 1, lower, upper)
                data_path = temporary_dir / f"{table_name}.data"
                header_path = temporary_dir / f"{table_name}.header"
                if not data_path.is_file() or not header_path.is_file():
                    raise RuntimeError("服务器返回的谱线文件不完整")
                # fetch()会把整张表载入HAPI缓存；合并只需磁盘文件，及时释放内存。
                getattr(hapi, "LOCAL_TABLE_CACHE", {}).pop(table_name, None)
                return
            except Exception as exc:
                last_error = exc
                self._remove_temporary_hapi_table(hapi, temporary_dir, table_name)
                if attempt >= maximum_attempts or not self._is_interrupted_transfer_error(exc):
                    break
                delay = float(self.DOWNLOAD_RETRY_BASE_DELAY_SECONDS) * (2 ** (attempt - 1))
                progress(
                    gas_index,
                    gas_count,
                    f"{gas.name}片段{part_index}/{part_count}下载中断，"
                    f"{delay:g}秒后自动重试…",
                )
                self._wait_for_retry(delay, cancelled)
        raise RuntimeError(
            f"{gas.name} HITRAN谱线片段{part_index}/{part_count}下载失败，"
            f"已自动尝试{maximum_attempts}次：{last_error}"
        ) from last_error

    @staticmethod
    def _is_interrupted_transfer_error(error: BaseException) -> bool:
        message = str(error).lower()
        markers = (
            "not_enough_data",
            "not enough data",
            "incompleteread",
            "incomplete read",
            "传输不完整",
            "文件不完整",
            "connection reset",
            "remote end closed",
            "unexpected eof",
            "eof occurred",
            "ssl",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _remove_temporary_hapi_table(hapi: Any, directory: Path, table_name: str) -> None:
        getattr(hapi, "LOCAL_TABLE_CACHE", {}).pop(table_name, None)
        for suffix in (".data", ".header"):
            path = directory / f"{table_name}{suffix}"
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _merge_downloaded_parts(
        part_paths: list[tuple[Path, Path]], destination: Path
    ) -> int:
        """流式合并相邻波数片段，并去除边界处可能返回的重复谱线。"""
        line_count = 0
        previous_line: str | None = None
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            for data_path, _header_path in part_paths:
                with data_path.open("r", encoding="utf-8", errors="strict") as source:
                    for line in source:
                        normalized = line.rstrip("\r\n")
                        if not normalized or normalized == previous_line:
                            continue
                        output.write(normalized + "\n")
                        previous_line = normalized
                        line_count += 1
        return line_count

    @classmethod
    def _wait_for_retry(cls, seconds: float, cancelled: CancelCallback) -> None:
        deadline = time.monotonic() + max(seconds, 0.0)
        while time.monotonic() < deadline:
            cls._check_cancelled(cancelled)
            time.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))

    def _load_hapi(self) -> Any:
        if self._hapi is not None:
            return self._hapi
        if not self.hapi_path.exists():
            raise FileNotFoundError(f"找不到HAPI脚本：{self.hapi_path}")
        module_name = "stirs_third_party_hapi"
        module = sys.modules.get(module_name)
        if module is None:
            specification = importlib.util.spec_from_file_location(module_name, self.hapi_path)
            if specification is None or specification.loader is None:
                raise RuntimeError(f"无法加载HAPI脚本：{self.hapi_path}")
            module = importlib.util.module_from_spec(specification)
            sys.modules[module_name] = module
            with redirect_stdout(io.StringIO()):
                specification.loader.exec_module(module)
        self._hapi = module
        return module

    def _valid_local_table(self, name: str) -> bool:
        database_dir, table_name = self.line_database_source(name)
        data_path = database_dir / f"{table_name}.data"
        header_path = database_dir / f"{table_name}.header"
        return self._valid_table_files(data_path, header_path)

    @staticmethod
    def _valid_table_files(data_path: Path, header_path: Path) -> bool:
        if not data_path.is_file() or not header_path.is_file() or data_path.stat().st_size <= 0:
            return False
        try:
            header = json.loads(header_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return False
        return isinstance(header, dict)

    @staticmethod
    def _validate_grid(lower: float, upper: float, step: float) -> tuple[float, float, float]:
        values = np.asarray([lower, upper, step], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("波数范围和间隔必须为有限数值。")
        lower_value, upper_value, step_value = map(float, values)
        if lower_value <= 0.0 or upper_value <= lower_value:
            raise ValueError("波数范围无效。")
        if step_value <= 0.0 or step_value > 1.0:
            raise ValueError("波数间隔必须大于0且不超过1 cm⁻¹。")
        point_count = int(math.ceil((upper_value - lower_value) / step_value)) + 1
        if point_count > 2_000_001:
            raise ValueError("波数采样点超过200万，请增大间隔或缩小范围。")
        return lower_value, upper_value, step_value

    @staticmethod
    def _check_cancelled(cancelled: CancelCallback) -> None:
        if cancelled():
            raise HapiCalculationCancelled("HAPI分层光学厚度计算已取消。")

    def _save_result(
        self,
        output_root: str | Path,
        profile: LayeredAtmosphericProfile,
        wavenumber: np.ndarray,
        total_tau: np.ndarray,
        components: dict[str, np.ndarray],
        lower: float,
        upper: float,
        step: float,
        downloaded: list[str],
        save_components: bool = True,
        performance: dict[str, Any] | None = None,
        optimization: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        final_dir = root / timestamp
        temporary_dir = root / f".{timestamp}.tmp"
        temporary_dir.mkdir(parents=False, exist_ok=False)
        try:
            boundaries = profile.altitude_boundaries_km
            layer_headers = [
                f"layer_{index + 1:02d}({boundaries[index]:g}-{boundaries[index + 1]:g}km)"
                for index in range(total_tau.shape[0])
            ]
            total_path = temporary_dir / "optical_depth_total.csv"
            np.savetxt(
                total_path,
                np.column_stack([wavenumber, total_tau.T]),
                delimiter=",",
                header="wavenumber(cm-1)," + ",".join(layer_headers),
                comments="",
                fmt="%.8e",
            )

            component_path: Path | None = None
            if save_components:
                component_path = temporary_dir / "optical_depth_components.npz"
                np.savez_compressed(
                    component_path,
                    wavenumber_cm=wavenumber,
                    total_tau_layers=total_tau.astype(np.float32),
                    **{f"tau_{name}": value for name, value in components.items()},
                )

            profile_columns = [
                profile.altitude_mid_km,
                profile.pressure_hpa,
                profile.temperature_k,
            ]
            profile_header = ["altitude_mid(km)", "pressure(hPa)", "temperature(K)"]
            for name in profile.gas_names:
                profile_columns.append(profile.gas_columns_molec_cm2[name])
                profile_header.append(f"column_{name}(molec_cm-2)")
            profile_path = temporary_dir / "atmospheric_profile_35_layers.csv"
            np.savetxt(
                profile_path,
                np.column_stack(profile_columns),
                delimiter=",",
                header=",".join(profile_header),
                comments="",
                fmt="%.8e",
            )

            database_status = self.line_database_status(profile.gas_names)
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "model": "HAPI Voigt line absorption",
                "hapi_path": str(self.hapi_path),
                "line_database_directory": str(self.database_dir),
                "line_database_sources": {
                    name: {
                        "directory": str(self.line_database_source(name)[0]),
                        "table_name": self.line_database_source(name)[1],
                    }
                    for name in profile.gas_names
                    if database_status.get(name)
                },
                "profile": {
                    "source_file": str(profile.source_path),
                    "for_index": profile.for_index,
                    "observation_time_utc": profile.observation_time_utc,
                    "latitude_deg": profile.latitude_deg,
                    "longitude_deg": profile.longitude_deg,
                    "quality_flag": profile.quality_flag,
                    "source_valid_level_count": profile.source_level_count,
                    "source_top_altitude_km": profile.source_top_altitude_km,
                    "target_layer_count": int(profile.altitude_mid_km.size),
                    "target_top_altitude_km": float(profile.altitude_boundaries_km[-1]),
                },
                "spectral_grid": {
                    "wavenumber_min_cm-1": lower,
                    "wavenumber_max_cm-1": upper,
                    "wavenumber_step_cm-1": step,
                    "point_count": int(wavenumber.size),
                },
                "gases": profile.gas_names,
                "gas_profile_sources": profile.gas_sources,
                "line_database_available": database_status,
                "downloaded_during_calculation": downloaded,
                "optimization": dict(optimization or {}),
                "performance": dict(performance or {}),
                "isotopologues": "HITRAN local isotopologue 1 for every gas",
                "limitations": [
                    "gas-phase Voigt line absorption only",
                    "no molecular continuum or collision-induced absorption",
                    "cloud, aerosol and Rayleigh extinction are handled by the ARTE Atmosphere solver",
                ],
                "files": {
                    "total_optical_depth": "optical_depth_total.csv",
                    "component_optical_depth": (
                        "optical_depth_components.npz" if component_path is not None else None
                    ),
                    "layered_profile": "atmospheric_profile_35_layers.csv",
                },
            }
            manifest_path = temporary_dir / "calculation_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_dir.rename(final_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        return {
            "output_directory": str(final_dir),
            "total_optical_depth_file": str(final_dir / total_path.name),
            "component_optical_depth_file": (
                str(final_dir / component_path.name) if component_path is not None else ""
            ),
            "profile_file": str(final_dir / profile_path.name),
            "manifest_file": str(final_dir / manifest_path.name),
            "gas_names": profile.gas_names,
            "wavenumber_min_cm": lower,
            "wavenumber_max_cm": upper,
            "wavenumber_step_cm": step,
            "spectral_point_count": int(wavenumber.size),
            "layer_count": int(total_tau.shape[0]),
            "downloaded_gases": downloaded,
            "profile_summary": {
                "for_index": profile.for_index,
                "observation_time_utc": profile.observation_time_utc,
                "latitude_deg": profile.latitude_deg,
                "longitude_deg": profile.longitude_deg,
                "quality_flag": profile.quality_flag,
            },
        }
