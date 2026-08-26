from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.atmospheric_radiation_manager import LayeredAtmosphereSolver
from core.hapi_optical_depth_manager import (
    LayeredAtmosphericProfile,
    NucapsAtmosphericProfileReader,
)


class GfsGlobalAtmosphericProfileReader:
    """Stream compact GFS pressure-level libraries into the ARTE 35-layer grid."""

    REQUIRED_VARIABLES = {
        "time",
        "level",
        "latitude",
        "longitude",
        "temperature_k",
        "specific_humidity_kg_kg",
        "ozone_mass_mixing_ratio_kg_kg",
        "surface_pressure_pa",
    }
    MOLAR_MASS_OZONE = 0.0479982
    # GFS文件没有这些长寿命气体。用统一干空气背景补齐，使最终代表模式可进行
    # 完整的主要气体HAPI计算；这些常量不会制造额外的空间模式。
    BACKGROUND_DRY_AIR_MOLE_FRACTIONS = {
        "CO2": 425.0e-6,
        "N2O": 338.0e-9,
        "CO": 100.0e-9,
        "CH4": 1.95e-6,
    }

    @staticmethod
    def _dataset_class():
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("读取全球GFS廓线库需要安装netCDF4。") from exc
        return Dataset

    @classmethod
    def is_supported(cls, source_path: str | Path) -> bool:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in {".nc", ".nc4"}:
            return False
        Dataset = cls._dataset_class()
        try:
            with Dataset(source) as dataset:
                return cls.REQUIRED_VARIABLES.issubset(dataset.variables)
        except (OSError, RuntimeError):
            return False

    @classmethod
    def read_sampled(
        cls,
        source_path: str | Path,
        maximum_samples: int,
        random_seed: int = 42,
    ) -> list[LayeredAtmosphericProfile]:
        """Read a deterministic, approximately area-balanced global sample.

        One time slice is loaded at a time. No converted profile or GRIB cache is
        written to disk.
        """

        Dataset = cls._dataset_class()
        source = Path(source_path).expanduser().resolve()
        maximum = max(2, int(maximum_samples))
        profiles: list[LayeredAtmosphericProfile] = []
        with Dataset(source) as dataset:
            missing = sorted(cls.REQUIRED_VARIABLES.difference(dataset.variables))
            if missing:
                raise ValueError(f"GFS全球廓线文件缺少变量：{', '.join(missing)}")
            levels = cls._filled(dataset.variables["level"][:])
            latitudes = cls._filled(dataset.variables["latitude"][:])
            longitudes = cls._filled(dataset.variables["longitude"][:])
            times = cls._filled(dataset.variables["time"][:])
            time_count = int(times.size)
            latitude_count = int(latitudes.size)
            longitude_count = int(longitudes.size)
            if time_count < 1 or latitude_count < 1 or longitude_count < 1:
                raise ValueError("GFS全球廓线文件没有可用的时空网格。")

            selected = cls._sample_grid_indices(
                time_count,
                latitudes,
                longitude_count,
                maximum,
                int(random_seed),
            )
            by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for time_index, latitude_index, longitude_index in selected:
                by_time[time_index].append((latitude_index, longitude_index))

            for time_index, locations in sorted(by_time.items()):
                # 每个三维变量约10.7 MiB（当前1°文件），处理完本月即释放。
                temperature = cls._filled(
                    dataset.variables["temperature_k"][time_index], np.float32
                )
                humidity = cls._filled(
                    dataset.variables["specific_humidity_kg_kg"][time_index],
                    np.float32,
                )
                ozone = cls._filled(
                    dataset.variables["ozone_mass_mixing_ratio_kg_kg"][time_index],
                    np.float32,
                )
                surface_pressure = cls._filled(
                    dataset.variables["surface_pressure_pa"][time_index], np.float32
                )
                terrain_variable = dataset.variables.get("surface_geopotential_height_m")
                terrain = (
                    cls._filled(terrain_variable[time_index], np.float32)
                    if terrain_variable is not None
                    else None
                )
                for latitude_index, longitude_index in locations:
                    try:
                        profile = cls._build_profile(
                            source,
                            time_index,
                            latitude_index,
                            longitude_index,
                            float(times[time_index]),
                            float(latitudes[latitude_index]),
                            float(longitudes[longitude_index]),
                            levels,
                            temperature[:, latitude_index, longitude_index],
                            humidity[:, latitude_index, longitude_index],
                            ozone[:, latitude_index, longitude_index],
                            float(surface_pressure[latitude_index, longitude_index]),
                            (
                                float(terrain[latitude_index, longitude_index])
                                if terrain is not None
                                else float("nan")
                            ),
                        )
                    except ValueError:
                        continue
                    profiles.append(profile)
        if len(profiles) < 2:
            raise ValueError("GFS全球廓线文件中不足2条有效温压廓线。")
        return profiles[:maximum]

    @staticmethod
    def _filled(values: Any, dtype: Any = float) -> np.ndarray:
        return np.ma.asarray(values, dtype=dtype).filled(np.nan)

    @staticmethod
    def _sample_grid_indices(
        time_count: int,
        latitudes: np.ndarray,
        longitude_count: int,
        maximum_samples: int,
        random_seed: int,
    ) -> list[tuple[int, int, int]]:
        spatial_count = int(latitudes.size) * int(longitude_count)
        total_count = time_count * spatial_count
        if maximum_samples >= total_count:
            return [
                (time_index, latitude_index, longitude_index)
                for time_index in range(time_count)
                for latitude_index in range(latitudes.size)
                for longitude_index in range(longitude_count)
            ]
        generator = np.random.default_rng(random_seed)
        latitude_weights = np.maximum(np.cos(np.deg2rad(latitudes)), 1.0e-6)
        spatial_weights = np.repeat(latitude_weights, longitude_count)
        spatial_weights /= np.sum(spatial_weights)
        base = maximum_samples // time_count
        remainder = maximum_samples % time_count
        result: list[tuple[int, int, int]] = []
        for time_index in range(time_count):
            count = min(base + (1 if time_index < remainder else 0), spatial_count)
            if count <= 0:
                continue
            flat_indices = generator.choice(
                spatial_count, size=count, replace=False, p=spatial_weights
            )
            result.extend(
                (
                    time_index,
                    int(flat_index) // longitude_count,
                    int(flat_index) % longitude_count,
                )
                for flat_index in flat_indices
            )
        return result

    @classmethod
    def _build_profile(
        cls,
        source: Path,
        time_index: int,
        latitude_index: int,
        longitude_index: int,
        timestamp_seconds: float,
        latitude: float,
        longitude: float,
        pressure_levels_hpa: np.ndarray,
        temperature_levels_k: np.ndarray,
        humidity_levels: np.ndarray,
        ozone_levels: np.ndarray,
        surface_pressure_pa: float,
        terrain_height_m: float,
    ) -> LayeredAtmosphericProfile:
        surface_hpa = float(surface_pressure_pa) / 100.0
        valid = (
            np.isfinite(pressure_levels_hpa)
            & (pressure_levels_hpa > 0.0)
            & (pressure_levels_hpa <= surface_hpa * 1.001)
            & np.isfinite(temperature_levels_k)
            & (temperature_levels_k > 100.0)
        )
        if np.count_nonzero(valid) < 2 or not np.isfinite(surface_hpa):
            raise ValueError("GFS网格点的有效温压层不足。")
        pressure = np.asarray(pressure_levels_hpa[valid], dtype=float)
        temperature = np.asarray(temperature_levels_k[valid], dtype=float)
        humidity = np.asarray(humidity_levels[valid], dtype=float)
        ozone = np.asarray(ozone_levels[valid], dtype=float)
        order = np.argsort(pressure)[::-1]
        pressure = pressure[order]
        temperature = temperature[order]
        humidity = humidity[order]
        ozone = ozone[order]
        if surface_hpa > pressure[0] * 1.001:
            pressure = np.r_[surface_hpa, pressure]
            temperature = np.r_[temperature[0], temperature]
            humidity = np.r_[humidity[0], humidity]
            ozone = np.r_[ozone[0], ozone]

        reader = NucapsAtmosphericProfileReader
        temperature = reader._fill_nonnegative(temperature)
        humidity = np.clip(reader._fill_nonnegative(humidity), 0.0, 0.5)
        ozone = np.maximum(reader._fill_nonnegative(ozone), 0.0)
        source_mid = reader._pressure_level_altitudes(pressure, temperature)
        source_bounds = reader._cell_boundaries(source_mid)
        source_pressure_bounds = reader._pressure_boundaries(pressure)
        source_air_column = reader._air_column_from_pressure(source_pressure_bounds)

        target_heights = np.asarray(LayeredAtmosphereSolver.LAYER_HEIGHT_KM, dtype=float)
        target_bounds = np.r_[0.0, np.cumsum(target_heights)]
        target_mid = 0.5 * (target_bounds[:-1] + target_bounds[1:])
        target_pressure_bounds = reader._target_pressure_boundaries(
            target_bounds, source_mid, pressure, temperature
        )
        target_pressure = np.sqrt(
            np.maximum(target_pressure_bounds[:-1], 1.0e-12)
            * np.maximum(target_pressure_bounds[1:], 1.0e-12)
        )
        target_temperature = np.interp(
            target_mid,
            source_mid,
            temperature,
            left=float(temperature[0]),
            right=np.nan,
        )
        above = ~np.isfinite(target_temperature)
        target_temperature[above] = reader._standard_temperature(target_mid[above])
        target_air_column = reader._air_column_from_pressure(target_pressure_bounds)

        water_molar_ratio = reader._mixing_ratio(humidity, "kg/kg")
        ozone_molar_ratio = (
            ozone * reader.MOLAR_MASS_DRY_AIR / cls.MOLAR_MASS_OZONE
        )
        gas_columns: dict[str, np.ndarray] = {}
        gas_sources: dict[str, str] = {}
        for name, molar_ratio, label in (
            ("H2O", water_molar_ratio, "GFS specific_humidity_kg_kg"),
            ("O3", ozone_molar_ratio, "GFS ozone_mass_mixing_ratio_kg_kg"),
        ):
            source_columns = np.maximum(molar_ratio * source_air_column, 0.0)
            remapped = reader._conservative_remap(
                source_bounds, source_columns, target_bounds
            )
            missing_top = target_mid > float(source_bounds[-1])
            if np.any(missing_top):
                top_ratio = reader._last_finite_positive(molar_ratio)
                if np.isfinite(top_ratio):
                    remapped[missing_top] += top_ratio * target_air_column[missing_top]
            gas_columns[name] = np.maximum(remapped, 0.0)
            gas_sources[name] = f"{label}，按压力空气柱守恒映射到35层"
        for name, mole_fraction in cls.BACKGROUND_DRY_AIR_MOLE_FRACTIONS.items():
            gas_columns[name] = target_air_column * float(mole_fraction)
            gas_sources[name] = (
                f"GFS未提供{name}；采用固定干空气背景摩尔分数"
                f" {float(mole_fraction):.6g}"
            )

        normalized_longitude = ((float(longitude) + 180.0) % 360.0) - 180.0
        time_text = (
            datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        profile_index = (
            int(time_index) * 1_000_000
            + int(latitude_index) * 1_000
            + int(longitude_index)
        )
        profile = LayeredAtmosphericProfile(
            source_path=source,
            for_index=profile_index,
            observation_time_utc=time_text,
            latitude_deg=float(latitude),
            longitude_deg=normalized_longitude,
            quality_flag=0,
            altitude_boundaries_km=target_bounds,
            altitude_mid_km=target_mid,
            pressure_hpa=target_pressure,
            temperature_k=target_temperature,
            gas_columns_molec_cm2=gas_columns,
            gas_sources=gas_sources,
            source_level_count=int(pressure.size),
            source_top_altitude_km=float(source_bounds[-1]),
        )
        # Extra grid metadata is attached dynamically so the common profile object
        # stays compatible with the existing HAPI manager.
        profile.gfs_time_index = int(time_index)  # type: ignore[attr-defined]
        profile.gfs_latitude_index = int(latitude_index)  # type: ignore[attr-defined]
        profile.gfs_longitude_index = int(longitude_index)  # type: ignore[attr-defined]
        profile.terrain_height_m = float(terrain_height_m)  # type: ignore[attr-defined]
        return profile
