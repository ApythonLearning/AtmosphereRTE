from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0
SIGMA = 5.670374419e-8
SOLAR_CONSTANT = 1361.0
SPECTRAL_WAVENUMBER_MIN_CM = 500.0
SPECTRAL_WAVENUMBER_MAX_CM = 33_300.0
SOLAR_REFERENCE_SOURCE = "TSIS-1 HSRS v2 (2022-11-30, 1 AU)"
SOLAR_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "data"
    / "tsis1_hsrs_v2_20221130.nc"
)


def build_specified_location_sample(parameters: dict[str, Any]) -> dict[str, float]:
    """由界面参数构造不依赖航迹的指定位置大气辐射求解样本。"""
    values = {
        "time": float(parameters.get("specified_time_s", 0.0)),
        "lat": float(parameters.get("specified_latitude_deg", 0.0)),
        "lon": float(parameters.get("specified_longitude_deg", 0.0)),
        "alt": float(parameters.get("specified_altitude", 700.0)),
        "right_ascension": float(parameters.get("specified_solar_right_ascension_deg", 0.0)),
        "declination": float(parameters.get("specified_solar_declination_deg", 0.0)),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("指定位置、时刻和太阳方向参数必须是有限数值。")
    if not -90.0 <= values["lat"] <= 90.0:
        raise ValueError("指定位置纬度必须在 -90～90° 范围内。")
    if not -180.0 <= values["lon"] <= 180.0:
        raise ValueError("指定位置经度必须在 -180～180° 范围内。")
    if values["alt"] <= 0.0:
        raise ValueError("指定位置观测高度必须大于 0。")
    if not -360.0 <= values["right_ascension"] <= 360.0:
        raise ValueError("指定太阳赤经必须在 -360～360° 范围内。")
    if not -90.0 <= values["declination"] <= 90.0:
        raise ValueError("指定太阳赤纬必须在 -90～90° 范围内。")
    return {
        **values,
        "sun_x": 1.0,
        "sun_y": 0.0,
        "sun_z": 0.0,
        "earth_x": 0.0,
        "earth_y": 0.0,
        "earth_z": -1.0,
    }


@dataclass
class EarthEnvironmentGrid:
    latitude: np.ndarray
    longitude: np.ndarray
    surface_temperature_k: np.ndarray
    surface_type: np.ndarray
    surface_albedo: np.ndarray
    surface_emissivity: np.ndarray
    cloud_fraction: np.ndarray
    cloud_top_temperature_k: np.ndarray
    cloud_top_height_m: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]
    cloud_effective_radius_um: np.ndarray | None = None
    cloud_liquid_water_path_g_m2: np.ndarray | None = None

    def validate(self) -> None:
        shape = self.surface_temperature_k.shape
        if len(shape) != 2 or any(np.asarray(value).shape != shape for value in (
            self.surface_type, self.surface_albedo, self.surface_emissivity,
            self.cloud_fraction, self.cloud_top_temperature_k,
            self.cloud_top_height_m, self.valid_mask,
        )):
            raise ValueError("地球环境场数组形状不一致。")
        if self.latitude.shape != shape or self.longitude.shape != shape:
            raise ValueError("经纬度网格与环境场形状不一致。")
        for value in (self.cloud_effective_radius_um, self.cloud_liquid_water_path_g_m2):
            if value is not None and np.asarray(value).shape != shape:
                raise ValueError("云微物理参数与环境场形状不一致。")
        if not np.isfinite(self.surface_temperature_k[self.valid_mask]).all():
            raise ValueError("地表温度包含无效值。")


@dataclass
class Merra2AerosolField:
    """MERRA-2 M2T1NXAER 的逐小时二维气溶胶场。"""

    latitude: np.ndarray
    longitude: np.ndarray
    time_seconds: np.ndarray
    aerosol_optical_depth_550: np.ndarray
    angstrom_exponent: np.ndarray | None
    time_labels: tuple[str, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        shape = self.aerosol_optical_depth_550.shape
        expected = (self.time_seconds.size, self.latitude.size, self.longitude.size)
        if shape != expected:
            raise ValueError(f"MERRA-2 AOD数组形状{shape}与时间/经纬度坐标{expected}不一致。")
        if self.angstrom_exponent is not None and self.angstrom_exponent.shape != shape:
            raise ValueError("MERRA-2 Ångström指数与AOD数组形状不一致。")
        if not np.isfinite(self.aerosol_optical_depth_550).any():
            raise ValueError("MERRA-2产品中没有有效的550 nm气溶胶光学厚度。")


class Merra2AerosolManager:
    """读取 MERRA-2 M2T1NXAER，并提供时空匹配的 AOD550。"""

    AOD_NAMES = ("TOTEXTTAU", "AODANA", "AOD550", "AOD_550")
    ANGSTROM_NAMES = ("TOTANGSTR", "ANGSTROM", "ANGSTROM_EXPONENT")
    LATITUDE_NAMES = ("lat", "latitude")
    LONGITUDE_NAMES = ("lon", "longitude")
    TIME_NAMES = ("time",)

    def __init__(self) -> None:
        self._field: Merra2AerosolField | None = None
        self._source_path = ""

    def get_field(self) -> Merra2AerosolField | None:
        return self._field

    def clear(self) -> None:
        self._field = None
        self._source_path = ""

    def load_product(self, path: str | Path) -> Merra2AerosolField:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"MERRA-2气溶胶产品不存在：{source}")
        try:
            from netCDF4 import Dataset, num2date  # type: ignore
        except ImportError as exc:
            raise RuntimeError("读取MERRA-2 NetCDF产品需要安装netCDF4。") from exc

        try:
            with Dataset(source) as dataset:
                variables = {name.lower(): name for name in dataset.variables}
                aod_name = self._find_name(variables, self.AOD_NAMES, "550 nm总消光AOD")
                lat_name = self._find_name(variables, self.LATITUDE_NAMES, "纬度")
                lon_name = self._find_name(variables, self.LONGITUDE_NAMES, "经度")
                time_name = self._find_name(variables, self.TIME_NAMES, "时间")
                angstrom_name = self._find_name(variables, self.ANGSTROM_NAMES, "", required=False)

                latitude = self._numeric_values(dataset.variables[lat_name]).reshape(-1)
                longitude = self._numeric_values(dataset.variables[lon_name]).reshape(-1)
                time_variable = dataset.variables[time_name]
                raw_time = self._numeric_values(time_variable).reshape(-1)
                if raw_time.size == 0 or latitude.size < 2 or longitude.size < 2:
                    raise ValueError("MERRA-2产品的时间或经纬度坐标数量不足。")
                time_seconds = self._relative_time_seconds(raw_time, str(getattr(time_variable, "units", "")))
                time_labels: tuple[str, ...]
                try:
                    dates = num2date(
                        raw_time,
                        units=str(time_variable.units),
                        calendar=str(getattr(time_variable, "calendar", "standard")),
                        only_use_cftime_datetimes=False,
                    )
                    time_labels = tuple(
                        value.strftime("%Y-%m-%d %H:%M:%S") for value in np.atleast_1d(dates)
                    )
                except Exception:  # noqa: BLE001 - 非标准时间单位仍可按相对时间使用
                    time_labels = tuple(f"+{value / 3600.0:.3f} h" for value in time_seconds)

                aod = self._ordered_field(
                    dataset.variables[aod_name], time_name, lat_name, lon_name
                )
                angstrom = (
                    self._ordered_field(dataset.variables[angstrom_name], time_name, lat_name, lon_name)
                    if angstrom_name else None
                )
                aod = np.where(np.isfinite(aod) & (aod >= 0.0) & (aod <= 10.0), aod, np.nan)
                if angstrom is not None:
                    angstrom = np.where(
                        np.isfinite(angstrom) & (angstrom >= -1.0) & (angstrom <= 5.0),
                        angstrom,
                        np.nan,
                    )

                time_order = np.argsort(time_seconds)
                lat_order = np.argsort(latitude)
                canonical_lon = (longitude + 180.0) % 360.0 - 180.0
                lon_order = np.argsort(canonical_lon)
                time_seconds = time_seconds[time_order]
                time_labels = tuple(time_labels[index] for index in time_order)
                latitude = latitude[lat_order]
                longitude = canonical_lon[lon_order]
                if np.unique(latitude).size != latitude.size or np.unique(longitude).size != longitude.size:
                    raise ValueError("MERRA-2产品包含重复的经纬度坐标。")
                aod = aod[time_order, :, :][:, lat_order, :][:, :, lon_order]
                if angstrom is not None:
                    angstrom = angstrom[time_order, :, :][:, lat_order, :][:, :, lon_order]

                field = Merra2AerosolField(
                    latitude=latitude,
                    longitude=longitude,
                    time_seconds=time_seconds,
                    aerosol_optical_depth_550=aod,
                    angstrom_exponent=angstrom,
                    time_labels=time_labels,
                    metadata={
                        "product": "MERRA-2 M2T1NXAER",
                        "source_path": str(source),
                        "aod_variable": aod_name,
                        "angstrom_variable": angstrom_name or "",
                        "loaded_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
        except OSError as exc:
            raise RuntimeError(f"无法打开MERRA-2 NetCDF产品：{source}") from exc
        field.validate()
        self._field = field
        self._source_path = str(source)
        return field

    def ensure_loaded(self, path: str | Path) -> Merra2AerosolField:
        source = str(Path(path).expanduser().resolve())
        if self._field is None or self._source_path != source:
            return self.load_product(source)
        return self._field

    def resample_for_grid(
        self,
        elapsed_seconds: float,
        target_latitude: np.ndarray,
        target_longitude: np.ndarray,
        subpoint_latitude_deg: float,
        subpoint_longitude_deg: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._field is None:
            raise RuntimeError("尚未读取MERRA-2气溶胶产品。")
        field = self._field
        time_index = int(np.argmin(np.abs(field.time_seconds - float(elapsed_seconds))))
        aod_source = field.aerosol_optical_depth_550[time_index]
        aod_grid = self._interpolate_regular_grid(
            field.latitude, field.longitude, aod_source, target_latitude, target_longitude
        )
        subpoint_aod = float(self._interpolate_regular_grid(
            field.latitude,
            field.longitude,
            aod_source,
            np.asarray([[subpoint_latitude_deg]], dtype=float),
            np.asarray([[subpoint_longitude_deg]], dtype=float),
        )[0, 0])
        angstrom_value = np.nan
        if field.angstrom_exponent is not None:
            angstrom_value = float(self._interpolate_regular_grid(
                field.latitude,
                field.longitude,
                field.angstrom_exponent[time_index],
                np.asarray([[subpoint_latitude_deg]], dtype=float),
                np.asarray([[subpoint_longitude_deg]], dtype=float),
            )[0, 0])
        nearest_lat = int(np.argmin(np.abs(field.latitude - float(subpoint_latitude_deg))))
        longitude_delta = np.abs(
            (field.longitude - float(subpoint_longitude_deg) + 180.0) % 360.0 - 180.0
        )
        nearest_lon = int(np.argmin(longitude_delta))
        info = {
            "aerosol_optical_depth_source": "MERRA-2 M2T1NXAER TOTEXTTAU",
            "aerosol_optical_depth_550": subpoint_aod,
            "aerosol_angstrom_exponent": angstrom_value,
            "merra2_grid_latitude_deg": float(field.latitude[nearest_lat]),
            "merra2_grid_longitude_deg": float(field.longitude[nearest_lon]),
            "merra2_time_index": time_index,
            "merra2_time": field.time_labels[time_index],
            "merra2_elapsed_seconds": float(field.time_seconds[time_index]),
        }
        return aod_grid, info

    @staticmethod
    def _find_name(
        variables: dict[str, str], names: tuple[str, ...], label: str, required: bool = True
    ) -> str:
        for name in names:
            if name.lower() in variables:
                return variables[name.lower()]
        if required:
            raise KeyError(f"MERRA-2产品中未找到{label}变量：{', '.join(names)}")
        return ""

    @staticmethod
    def _numeric_values(variable: Any) -> np.ndarray:
        # netCDF4's automatic mask/scale path tries to cast ``valid_range``
        # to the storage dtype.  Some official MERRA-2 products store that
        # attribute with a different numeric dtype, which produces warnings
        # and may leave invalid packed values unmasked.  Read the packed data
        # first and apply the relevant CF attributes explicitly instead.
        mask_enabled = bool(getattr(variable, "mask", True))
        scale_enabled = bool(getattr(variable, "scale", True))
        variable.set_auto_maskandscale(False)
        try:
            packed = variable[:]
        finally:
            variable.set_auto_mask(mask_enabled)
            variable.set_auto_scale(scale_enabled)

        packed_array = np.asanyarray(packed)
        invalid = np.ma.getmaskarray(packed_array).copy()
        packed_values = np.asarray(np.ma.getdata(packed_array), dtype=float)
        invalid |= ~np.isfinite(packed_values)

        for attribute_name in ("_FillValue", "missing_value"):
            attribute = getattr(variable, attribute_name, None)
            if attribute is None:
                continue
            for marker in np.asarray(attribute).reshape(-1):
                try:
                    marker_value = float(marker)
                except (TypeError, ValueError):
                    continue
                if np.isnan(marker_value):
                    invalid |= np.isnan(packed_values)
                elif np.isfinite(marker_value):
                    invalid |= packed_values == marker_value

        valid_min = getattr(variable, "valid_min", None)
        valid_max = getattr(variable, "valid_max", None)
        valid_range = getattr(variable, "valid_range", None)
        if valid_range is not None:
            range_values = np.asarray(valid_range).reshape(-1)
            if range_values.size >= 2:
                valid_min, valid_max = range_values[:2]
        for bound, is_lower in ((valid_min, True), (valid_max, False)):
            if bound is None:
                continue
            try:
                bound_value = float(np.asarray(bound).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                continue
            if not np.isfinite(bound_value):
                continue
            invalid |= packed_values < bound_value if is_lower else packed_values > bound_value

        scale_factor = float(getattr(variable, "scale_factor", 1.0))
        add_offset = float(getattr(variable, "add_offset", 0.0))
        values = packed_values * scale_factor + add_offset
        values[invalid] = np.nan
        return values

    def _ordered_field(self, variable: Any, time_name: str, lat_name: str, lon_name: str) -> np.ndarray:
        values = self._numeric_values(variable)
        dimensions = list(variable.dimensions)
        required = [time_name, lat_name, lon_name]
        missing = [name for name in required if name not in dimensions]
        if missing:
            raise ValueError(f"变量{variable.name}缺少维度：{', '.join(missing)}")
        extra_axes = [index for index, name in enumerate(dimensions) if name not in required]
        for axis in reversed(extra_axes):
            if values.shape[axis] != 1:
                raise ValueError(f"变量{variable.name}含不支持的非单值维度{dimensions[axis]}。")
            values = np.take(values, 0, axis=axis)
            dimensions.pop(axis)
        order = [dimensions.index(name) for name in required]
        return np.transpose(values, order)

    @staticmethod
    def _relative_time_seconds(values: np.ndarray, units: str) -> np.ndarray:
        lower = units.lower()
        factor = 1.0
        if "day" in lower:
            factor = 86400.0
        elif "hour" in lower:
            factor = 3600.0
        elif "minute" in lower:
            factor = 60.0
        return (np.asarray(values, dtype=float) - float(values[0])) * factor

    @staticmethod
    def _interpolate_regular_grid(
        latitude: np.ndarray,
        longitude: np.ndarray,
        values: np.ndarray,
        target_latitude: np.ndarray,
        target_longitude: np.ndarray,
    ) -> np.ndarray:
        canonical_target_lon = (np.asarray(target_longitude, dtype=float) + 180.0) % 360.0 - 180.0
        longitude_extended = np.concatenate((longitude, [longitude[0] + 360.0]))
        values_extended = np.concatenate((values, values[:, :1]), axis=1)
        canonical_target_lon = np.where(
            canonical_target_lon < longitude_extended[0], canonical_target_lon + 360.0, canonical_target_lon
        )
        target_lat = np.clip(
            np.asarray(target_latitude, dtype=float).reshape(-1), latitude[0], latitude[-1]
        )
        target_lon = canonical_target_lon.reshape(-1)
        lat_upper = np.clip(np.searchsorted(latitude, target_lat, side="right"), 1, latitude.size - 1)
        lat_lower = lat_upper - 1
        lon_upper = np.clip(
            np.searchsorted(longitude_extended, target_lon, side="right"),
            1,
            longitude_extended.size - 1,
        )
        lon_lower = lon_upper - 1
        lat_weight = np.divide(
            target_lat - latitude[lat_lower],
            latitude[lat_upper] - latitude[lat_lower],
            out=np.zeros_like(target_lat),
            where=(latitude[lat_upper] - latitude[lat_lower]) != 0.0,
        )
        lon_weight = np.divide(
            target_lon - longitude_extended[lon_lower],
            longitude_extended[lon_upper] - longitude_extended[lon_lower],
            out=np.zeros_like(target_lon),
            where=(longitude_extended[lon_upper] - longitude_extended[lon_lower]) != 0.0,
        )
        corner_values = np.stack((
            values_extended[lat_lower, lon_lower],
            values_extended[lat_lower, lon_upper],
            values_extended[lat_upper, lon_lower],
            values_extended[lat_upper, lon_upper],
        ))
        corner_weights = np.stack((
            (1.0 - lat_weight) * (1.0 - lon_weight),
            (1.0 - lat_weight) * lon_weight,
            lat_weight * (1.0 - lon_weight),
            lat_weight * lon_weight,
        ))
        finite = np.isfinite(corner_values)
        weight_sum = np.sum(np.where(finite, corner_weights, 0.0), axis=0)
        result = np.divide(
            np.sum(np.where(finite, corner_values * corner_weights, 0.0), axis=0),
            weight_sum,
            out=np.full(target_lat.shape, np.nan),
            where=weight_sum > 1e-12,
        )
        missing = ~np.isfinite(result)
        if np.any(missing):
            nearest_lat = np.abs(latitude[:, None] - target_lat[missing]).argmin(axis=0)
            nearest_lon = np.abs(longitude_extended[:, None] - target_lon[missing]).argmin(axis=0)
            result[missing] = values_extended[nearest_lat, nearest_lon]
        return np.asarray(result, dtype=float).reshape(np.asarray(target_latitude).shape)


class ModisDataManager:
    """读取并标准化 MODIS/NetCDF 地球环境产品。

    HDF5/NetCDF 按数据集名称查找，避免依赖 MATLAB 示例中的 SDS 顺序。
    HDF4 需要可选的 pyhdf；缺少依赖时会给出明确错误。
    """

    LAND_TEMP_NAMES = ("LST_Day_CMG", "LST_Day", "Land_Surface_Temperature")
    SEA_TEMP_NAMES = ("SST", "sst", "sea_surface_temperature")
    LAND_TYPE_NAMES = ("Majority_Land_Cover_Type_1", "LC_Type1", "Land_Cover_Type_1")
    CLOUD_FRACTION_NAMES = ("Cloud_Fraction_Mean_Mean", "Cloud_Fraction_Mean")
    CLOUD_TEMP_NAMES = ("Cloud_Top_Temperature_Mean_Mean", "Cloud_Top_Temperature_Mean")
    CLOUD_HEIGHT_NAMES = ("Cloud_Top_Height_Mean_Mean", "Cloud_Top_Height_Mean")
    CLOUD_EFFECTIVE_RADIUS_NAMES = (
        "Cloud_Effective_Radius_Liquid_Mean_Mean",
        "Cloud_Effective_Radius_Liquid_Mean",
    )
    CLOUD_LIQUID_WATER_PATH_NAMES = (
        "Cloud_Water_Path_Liquid_Mean_Mean",
        "Cloud_Water_Path_Liquid_Mean",
    )
    CLOUD_OPTICAL_THICKNESS_NAMES = (
        "Cloud_Optical_Thickness_Liquid_Mean_Mean",
        "Cloud_Optical_Thickness_Liquid_Mean",
    )

    # MCD12 IGBP地表类型的宽带太阳反照率和热红外灰体发射率。
    # 太阳反照率采用工程输入给定的类别代表值；热红外发射率与太阳反照率
    # 分属不同波段，不能简单令epsilon=1-albedo，因此按典型地表经验值设置。
    # 顺序：水、5类森林、2类灌木、2类稀树草原、草原、湿地、耕地、城市、
    #       耕地/自然植被镶嵌、冰雪、荒地、海冰。
    SURFACE_ALBEDO = np.asarray([
        0.07,
        0.15, 0.15, 0.15, 0.15, 0.15,
        0.17, 0.17,
        0.20, 0.20, 0.20,
        0.07,
        0.18,
        0.18,
        0.18,
        0.80,
        0.27,
        0.80,
    ], dtype=float)
    SURFACE_EMISSIVITY = np.asarray([
        0.985,
        0.980, 0.980, 0.980, 0.980, 0.980,
        0.960, 0.950,
        0.970, 0.960, 0.970,
        0.985,
        0.970,
        0.940,
        0.970,
        0.990,
        0.910,
        0.970,
    ], dtype=float)
    SURFACE_TYPE_NAMES = (
        "水体",
        "常绿针叶林",
        "常绿阔叶林",
        "针叶林",
        "落叶针叶林",
        "混交林",
        "封闭灌木",
        "开放灌木",
        "多树草原",
        "热带草原",
        "草原",
        "永久湿地",
        "耕地",
        "城市和建筑",
        "农作物/自然植被镶嵌",
        "冰雪",
        "荒地",
        "海冰",
    )
    OPAC_AEROSOL_TYPE_LABELS = {
        "continental_clean": "大陆清洁型",
        "continental_average": "大陆平均型",
        "urban": "城市型",
        "desert": "沙漠型",
        "maritime_clean": "海洋清洁型",
        "arctic": "北极型",
        "antarctic": "南极型",
    }
    # MCD12 IGBP类型到边界层OPAC气溶胶类型。冰雪和海冰在南半球
    # 进一步切换为antarctic，避免仅由类型编号误判南极环境。
    SURFACE_OPAC_AEROSOL_TYPE = (
        "maritime_clean",
        "continental_clean", "continental_clean", "continental_clean",
        "continental_clean", "continental_clean",
        "continental_average", "continental_average",
        "continental_average", "continental_average", "continental_average",
        "continental_average", "continental_average",
        "urban",
        "continental_average",
        "arctic",
        "desert",
        "arctic",
    )

    def __init__(self) -> None:
        self._grid: EarthEnvironmentGrid | None = None
        self._source_paths: dict[str, str] = {}

    def get_grid(self) -> EarthEnvironmentGrid | None:
        return self._grid

    def clear(self) -> None:
        self._grid = None
        self._source_paths = {}

    def sample_environment_at_subpoint(
        self,
        grid: EarthEnvironmentGrid,
        latitude_deg: float,
        longitude_deg: float,
    ) -> dict[str, Any]:
        """返回星下点最近有效网格的地表和云环境参数。"""
        latitude = np.deg2rad(float(latitude_deg))
        longitude = np.deg2rad(float(longitude_deg))
        grid_latitude = np.deg2rad(np.asarray(grid.latitude, dtype=float))
        grid_longitude = np.deg2rad(np.asarray(grid.longitude, dtype=float))
        angular_similarity = (
            np.sin(latitude) * np.sin(grid_latitude)
            + np.cos(latitude) * np.cos(grid_latitude) * np.cos(grid_longitude - longitude)
        )
        valid = np.asarray(grid.valid_mask, dtype=bool) & np.isfinite(angular_similarity)
        if not valid.any():
            raise RuntimeError("MODIS地球环境网格中没有可用的星下点数据。")
        nearest_flat_index = int(np.argmax(np.where(valid, angular_similarity, -np.inf)))
        index = np.unravel_index(nearest_flat_index, valid.shape)
        cloud_re, cloud_lwp = self._grid_cloud_microphysics(grid)
        surface_type_value = float(np.asarray(grid.surface_type, dtype=float)[index])
        surface_type_code = int(round(surface_type_value)) if np.isfinite(surface_type_value) else -1
        surface_type_name = (
            self.SURFACE_TYPE_NAMES[surface_type_code]
            if 0 <= surface_type_code < len(self.SURFACE_TYPE_NAMES)
            else "未知地表"
        )
        grid_latitude_value = float(np.asarray(grid.latitude, dtype=float)[index])
        opac_type = self.opac_aerosol_type_for_surface(
            surface_type_code, grid_latitude_value
        )
        effective_radius = float(cloud_re[index])
        liquid_water_path = float(cloud_lwp[index])
        cloud_height = float(np.asarray(grid.cloud_top_height_m, dtype=float)[index])
        cloud_optical_thickness = (
            1.5 * liquid_water_path / max(effective_radius, 1e-12)
            if cloud_height > 0.0 else 0.0
        )
        return {
            "environment_grid_latitude_deg": grid_latitude_value,
            "environment_grid_longitude_deg": float(np.asarray(grid.longitude, dtype=float)[index]),
            "subpoint_surface_type_code": surface_type_code,
            "subpoint_surface_type_name": surface_type_name,
            "subpoint_surface_temperature_k": float(np.asarray(grid.surface_temperature_k, dtype=float)[index]),
            "subpoint_surface_albedo": float(np.asarray(grid.surface_albedo, dtype=float)[index]),
            "subpoint_surface_emissivity": float(np.asarray(grid.surface_emissivity, dtype=float)[index]),
            "subpoint_opac_aerosol_type": opac_type,
            "subpoint_opac_aerosol_type_name": self.OPAC_AEROSOL_TYPE_LABELS[opac_type],
            "subpoint_cloud_fraction": float(np.asarray(grid.cloud_fraction, dtype=float)[index]),
            "subpoint_cloud_top_temperature_k": float(np.asarray(grid.cloud_top_temperature_k, dtype=float)[index]),
            "subpoint_cloud_top_height_m": cloud_height,
            "subpoint_cloud_effective_radius_um": effective_radius,
            "subpoint_cloud_liquid_water_path_g_m2": liquid_water_path,
            "subpoint_cloud_optical_thickness": float(cloud_optical_thickness),
            "cloud_effective_radius_source": str(
                grid.metadata.get("cloud_effective_radius_source", "未记录")
            ),
            "cloud_liquid_water_path_source": str(
                grid.metadata.get("cloud_liquid_water_path_source", "未记录")
            ),
        }

    @classmethod
    def opac_aerosol_type_for_surface(cls, surface_type_code: int, latitude_deg: float) -> str:
        if 0 <= int(surface_type_code) < len(cls.SURFACE_OPAC_AEROSOL_TYPE):
            aerosol_type = cls.SURFACE_OPAC_AEROSOL_TYPE[int(surface_type_code)]
        else:
            aerosol_type = "continental_average"
        if aerosol_type == "arctic" and float(latitude_deg) < 0.0:
            return "antarctic"
        return aerosol_type

    @classmethod
    def opac_aerosol_type_grid(
        cls, surface_type: np.ndarray, latitude_deg: np.ndarray
    ) -> np.ndarray:
        surface = np.asarray(surface_type, dtype=float)
        latitude = np.broadcast_to(np.asarray(latitude_deg, dtype=float), surface.shape)
        result = np.full(surface.shape, "continental_average", dtype="<U24")
        finite = np.isfinite(surface)
        codes = np.rint(np.where(finite, surface, -1.0)).astype(int)
        for code, aerosol_type in enumerate(cls.SURFACE_OPAC_AEROSOL_TYPE):
            result[codes == code] = aerosol_type
        polar = np.isin(codes, [15, 17])
        result[polar & (latitude < 0.0)] = "antarctic"
        return result

    def load_products(
        self,
        land_temperature_file: str,
        sea_temperature_file: str,
        cloud_file: str,
        land_type_file: str,
        resolution_deg: float = 2.0,
    ) -> EarthEnvironmentGrid:
        resolution = float(resolution_deg)
        if resolution <= 0 or resolution > 10:
            raise ValueError("地球网格分辨率必须在 0 到 10 度之间。")
        land_t = self._read_named_dataset(Path(land_temperature_file), self.LAND_TEMP_NAMES, "陆地温度")
        sea_t = self._read_named_dataset(Path(sea_temperature_file), self.SEA_TEMP_NAMES, "海表温度")
        land_type = self._read_named_dataset(Path(land_type_file), self.LAND_TYPE_NAMES, "地表类型")
        cloud_fraction = self._read_named_dataset(Path(cloud_file), self.CLOUD_FRACTION_NAMES, "云量")
        cloud_t = self._read_named_dataset(Path(cloud_file), self.CLOUD_TEMP_NAMES, "云顶温度")
        cloud_h = self._read_named_dataset(Path(cloud_file), self.CLOUD_HEIGHT_NAMES, "云顶高度")
        cloud_re_raw = self._read_named_dataset_optional(
            Path(cloud_file), self.CLOUD_EFFECTIVE_RADIUS_NAMES
        )
        cloud_lwp_raw = self._read_named_dataset_optional(
            Path(cloud_file), self.CLOUD_LIQUID_WATER_PATH_NAMES
        )
        cloud_optical_thickness_raw = self._read_named_dataset_optional(
            Path(cloud_file), self.CLOUD_OPTICAL_THICKNESS_NAMES
        )

        shape = (max(1, int(round(180.0 / resolution))), max(1, int(round(360.0 / resolution))))
        land_t = self._to_kelvin(land_t, "land")
        land_t[~np.isfinite(land_t) | (land_t < 150.0) | (land_t > 400.0)] = np.nan
        land_t = self._resample_continuous(land_t, shape)
        sea_t = self._to_kelvin(sea_t, "sea")
        sea_t[~np.isfinite(sea_t) | (sea_t < 268.15) | (sea_t > 323.15)] = np.nan
        # MOD28 NetCDF纬度由南向北排列，统一为第0行对应北纬90度。
        sea_t = self._resample_continuous(np.flipud(sea_t), shape)
        land_type = self._resample(land_type, shape, nearest=True).astype(np.int16)
        cloud_fraction_raw = np.asarray(cloud_fraction, dtype=float)
        cloud_fraction = self._normalise_fraction(cloud_fraction_raw)
        cloud_fraction[~np.isfinite(cloud_fraction_raw) | (cloud_fraction_raw < 0.0)] = np.nan
        cloud_fraction = np.clip(np.nan_to_num(self._resample_continuous(cloud_fraction, shape)), 0.0, 1.0)
        cloud_t = self._cloud_temperature(cloud_t)
        cloud_t[~np.isfinite(cloud_t) | (cloud_t <= 150.0) | (cloud_t >= 330.0)] = np.nan
        cloud_t = self._resample_continuous(cloud_t, shape)
        cloud_h_raw = np.asarray(cloud_h, dtype=float)
        cloud_h = self._cloud_height(cloud_h_raw)
        cloud_h[~np.isfinite(cloud_h_raw) | (cloud_h_raw < 0.0) | (cloud_h > 20_000.0)] = np.nan
        cloud_h = self._resample_continuous(cloud_h, shape)
        cloud_re = self._prepare_cloud_effective_radius(cloud_re_raw, shape)
        cloud_lwp = self._prepare_cloud_liquid_water_path(cloud_lwp_raw, shape)
        cloud_optical_thickness = self._prepare_cloud_optical_thickness(
            cloud_optical_thickness_raw, shape
        )

        land_valid = np.isfinite(land_t)
        sea_valid = np.isfinite(sea_t)
        surface_t = np.where(
            land_valid & sea_valid,
            0.5 * (land_t + sea_t),
            np.where(land_valid, land_t, np.where(sea_valid, sea_t, np.nan)),
        )
        surface_t = self._fill_missing(surface_t, 288.15)
        albedo, emissivity = self._surface_properties(land_type)
        cloud_t_valid = np.isfinite(cloud_t) & (cloud_t > 150.0) & (cloud_t < 330.0)
        # 缺失云顶温度不能用地表温度纹理替代，否则质检图会产生“伪云温”。
        # 计算端会根据云高和标准温度递减率生成临时回退值。
        cloud_t = np.where(cloud_t_valid, cloud_t, np.nan)
        cloud_h = np.clip(np.nan_to_num(cloud_h, nan=0.0), 0.0, 20_000.0)
        estimated_re, estimated_lwp = self.estimate_cloud_microphysics(cloud_t, cloud_h)
        direct_re = np.isfinite(cloud_re)
        direct_lwp = np.isfinite(cloud_lwp)
        cloud_re = np.where(direct_re, cloud_re, estimated_re)
        derived_lwp = (~direct_lwp) & np.isfinite(cloud_optical_thickness)
        cloud_lwp = np.where(
            direct_lwp,
            cloud_lwp,
            np.where(
                derived_lwp,
                cloud_optical_thickness * cloud_re / 1.5,
                estimated_lwp,
            ),
        )

        lat_values = 90.0 - (np.arange(shape[0]) + 0.5) * 180.0 / shape[0]
        lon_values = -180.0 + (np.arange(shape[1]) + 0.5) * 360.0 / shape[1]
        longitude, latitude = np.meshgrid(lon_values, lat_values)
        paths = {
            "land_temperature": str(Path(land_temperature_file).resolve()),
            "sea_temperature": str(Path(sea_temperature_file).resolve()),
            "cloud": str(Path(cloud_file).resolve()),
            "land_type": str(Path(land_type_file).resolve()),
        }
        grid = EarthEnvironmentGrid(
            latitude=latitude, longitude=longitude,
            surface_temperature_k=surface_t, surface_type=land_type,
            surface_albedo=albedo, surface_emissivity=emissivity,
            cloud_fraction=cloud_fraction, cloud_top_temperature_k=cloud_t,
            cloud_top_height_m=cloud_h, valid_mask=np.isfinite(surface_t),
            cloud_effective_radius_um=cloud_re,
            cloud_liquid_water_path_g_m2=cloud_lwp,
            metadata={
                "resolution_deg": resolution,
                "source_paths": paths,
                "loaded_at": datetime.now().isoformat(timespec="seconds"),
                "processing_version": 4,
                "cloud_effective_radius_source": "modis_with_empirical_fallback" if direct_re.any() else "empirical",
                "cloud_liquid_water_path_source": (
                    "modis_with_optical_thickness_and_empirical_fallback"
                    if direct_lwp.any()
                    else "modis_optical_thickness_with_empirical_fallback"
                    if derived_lwp.any()
                    else "empirical"
                ),
            },
        )
        grid.validate()
        self._grid = grid
        self._source_paths = paths
        return grid

    def save_cache(self, path: str | Path) -> str:
        if self._grid is None:
            raise RuntimeError("尚未加载 MODIS 地球环境数据。")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        grid = self._grid
        metadata = dict(grid.metadata)
        metadata["processing_version"] = 4
        cloud_re, cloud_lwp = self._grid_cloud_microphysics(grid)
        np.savez_compressed(
            target, latitude=grid.latitude, longitude=grid.longitude,
            surface_temperature_k=grid.surface_temperature_k, surface_type=grid.surface_type,
            surface_albedo=grid.surface_albedo, surface_emissivity=grid.surface_emissivity,
            cloud_fraction=grid.cloud_fraction, cloud_top_temperature_k=grid.cloud_top_temperature_k,
            cloud_top_height_m=grid.cloud_top_height_m, valid_mask=grid.valid_mask,
            cloud_effective_radius_um=cloud_re,
            cloud_liquid_water_path_g_m2=cloud_lwp,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return str(target)

    def load_cache(self, path: str | Path) -> EarthEnvironmentGrid:
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
            if int(metadata.get("processing_version", 0)) < 3:
                raise ValueError("环境缓存由旧版MODIS温度处理生成，请重新读取原始MODIS数据。")
            fields = {key: np.asarray(data[key]) for key in (
                "latitude", "longitude", "surface_temperature_k", "surface_type",
                "surface_albedo", "surface_emissivity", "cloud_fraction",
                "cloud_top_temperature_k", "cloud_top_height_m", "valid_mask",
            )}
            cloud_re = np.asarray(data["cloud_effective_radius_um"]) if "cloud_effective_radius_um" in data else None
            cloud_lwp = np.asarray(data["cloud_liquid_water_path_g_m2"]) if "cloud_liquid_water_path_g_m2" in data else None
            grid = EarthEnvironmentGrid(
                **fields,
                metadata=metadata,
                cloud_effective_radius_um=cloud_re,
                cloud_liquid_water_path_g_m2=cloud_lwp,
            )
        if grid.cloud_effective_radius_um is None or grid.cloud_liquid_water_path_g_m2 is None:
            estimated_re, estimated_lwp = self.estimate_cloud_microphysics(
                grid.cloud_top_temperature_k, grid.cloud_top_height_m
            )
            grid.cloud_effective_radius_um = estimated_re
            grid.cloud_liquid_water_path_g_m2 = estimated_lwp
        grid.valid_mask = grid.valid_mask.astype(bool)
        grid.validate()
        self._grid = grid
        self._source_paths = dict(metadata.get("source_paths", {}))
        return grid

    def _read_named_dataset(self, path: Path, names: tuple[str, ...], label: str) -> np.ndarray:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label}文件不存在：{path}")
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key = next((name for name in names if name in data), None)
                if key is None:
                    raise KeyError(f"{path.name} 中未找到{label}数据集：{', '.join(names)}")
                return np.asarray(data[key], dtype=float)
        try:
            import h5py  # type: ignore
            with h5py.File(path, "r") as handle:
                found: list[np.ndarray] = []
                def visitor(name: str, obj: Any) -> None:
                    if hasattr(obj, "shape") and name.rsplit("/", 1)[-1] in names:
                        found.append(np.asarray(obj[...], dtype=float))
                handle.visititems(visitor)
                if found:
                    return np.squeeze(found[0])
        except OSError:
            pass
        except ImportError:
            pass
        if path.suffix.lower() == ".hdf":
            try:
                from pyhdf.SD import SD  # type: ignore
                dataset = SD(str(path))
                key = next((name for name in names if name in dataset.datasets()), None)
                if key is not None:
                    return np.asarray(dataset.select(key).get(), dtype=float).squeeze()
            except ImportError as exc:
                raise RuntimeError(f"无法读取{label}。HDF4 产品需要安装 pyhdf。") from exc
        try:
            from netCDF4 import Dataset  # type: ignore
            with Dataset(path) as dataset:
                key = next((name for name in names if name in dataset.variables), None)
                if key is not None:
                    values = dataset.variables[key][:]
                    if np.ma.isMaskedArray(values):
                        values = values.filled(np.nan)
                    return np.asarray(values, dtype=float).squeeze()
        except ImportError as exc:
            raise RuntimeError(f"无法读取{label}。NetCDF 产品需要安装 netCDF4。") from exc
        except OSError:
            pass
        raise KeyError(f"{path.name} 中未找到{label}数据集：{', '.join(names)}")

    def _read_named_dataset_optional(self, path: Path, names: tuple[str, ...]) -> np.ndarray | None:
        try:
            return self._read_named_dataset(path, names, "可选云微物理")
        except (KeyError, RuntimeError):
            return None

    def _resample(self, values: np.ndarray, shape: tuple[int, int], nearest: bool) -> np.ndarray:
        array = np.asarray(values, dtype=float).squeeze()
        if array.ndim != 2:
            raise ValueError(f"环境数据必须是二维数组，当前形状为 {array.shape}。")
        if array.shape == shape:
            return array
        row = np.linspace(0, array.shape[0] - 1, shape[0])
        col = np.linspace(0, array.shape[1] - 1, shape[1])
        if nearest:
            return array[np.rint(row).astype(int)[:, None], np.rint(col).astype(int)[None, :]]
        try:
            from scipy.ndimage import map_coordinates  # type: ignore
            rr, cc = np.meshgrid(row, col, indexing="ij")
            return map_coordinates(array, [rr, cc], order=1, mode="nearest")
        except ImportError:
            # 两次一维线性插值，避免将 scipy 变成强制依赖。
            intermediate = np.vstack([np.interp(col, np.arange(array.shape[1]), line) for line in array])
            return np.vstack([np.interp(row, np.arange(array.shape[0]), intermediate[:, j]) for j in range(shape[1])]).T

    def _resample_continuous(self, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """对含缺失值的连续场进行归一化加权重采样，防止填充值扩散。"""
        array = np.asarray(values, dtype=float)
        if array.shape == shape:
            return array.copy()
        valid = np.isfinite(array)
        numerator = self._resample(np.where(valid, array, 0.0), shape, nearest=False)
        weight = self._resample(valid.astype(float), shape, nearest=False)
        return np.divide(numerator, weight, out=np.full(shape, np.nan), where=weight > 1e-6)

    def _to_kelvin(self, values: np.ndarray, source: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return array
        median = float(np.median(finite))
        if source == "land" and float(np.nanmax(finite)) > 1000.0:
            # MOD11C3 LST_Day_CMG明确规定 scale_factor=0.02 K。
            array = array * 0.02
        elif median < 100.0:
            array = array + 273.15
        return array

    def _normalise_fraction(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        maximum = float(np.nanmax(array)) if np.isfinite(array).any() else 0.0
        if maximum > 100.0:
            array *= 1e-4
        elif maximum > 1.0:
            array *= 0.01
        return np.clip(np.nan_to_num(array), 0.0, 1.0)

    def _cloud_temperature(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array) & (array > -32000.0) & (array < 32000.0)]
        median = float(np.median(finite)) if finite.size else 250.0
        # MOD08_M3 Cloud_Top_Temperature_Mean_Mean：
        # physical_K = (stored_value + 15000) / 100，与参考MATLAB实现一致。
        if abs(median) > 1000.0:
            return (array + 15000.0) / 100.0
        if median < 100.0:
            return array + 273.15
        return array

    def _cloud_height(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        median = float(np.nanmedian(np.abs(array))) if np.isfinite(array).any() else 0.0
        return array * 1000.0 if 0.0 < median < 30.0 else array

    def _prepare_cloud_effective_radius(
        self,
        values: np.ndarray | None,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if values is None:
            return np.full(shape, np.nan)
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array) & (array > 0.0)]
        median = float(np.median(finite)) if finite.size else 0.0
        if median > 100.0:
            array *= 0.01
        elif median > 60.0:
            array *= 0.1
        array[(array < 2.0) | (array > 60.0)] = np.nan
        return self._resample_continuous(array, shape)

    def _prepare_cloud_liquid_water_path(
        self,
        values: np.ndarray | None,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if values is None:
            return np.full(shape, np.nan)
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array) & (array > 0.0)]
        median = float(np.median(finite)) if finite.size else 0.0
        if median > 5000.0:
            array *= 0.01
        array[(array < 1.0) | (array > 3000.0)] = np.nan
        return self._resample_continuous(array, shape)

    def _prepare_cloud_optical_thickness(
        self,
        values: np.ndarray | None,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if values is None:
            return np.full(shape, np.nan)
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array) & (array > 0.0)]
        median = float(np.median(finite)) if finite.size else 0.0
        if median > 200.0:
            array *= 0.01
        array[(array <= 0.0) | (array > 200.0)] = np.nan
        return self._resample_continuous(array, shape)

    @staticmethod
    def estimate_cloud_microphysics(
        cloud_temperature_k: np.ndarray,
        cloud_height_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据云顶温度和云高估算水云有效半径与液态水路径。

        缺少MODIS直接反演时，以10 μm和60 g/m²（ARTESolver示例值）
        为基准，随云高和云顶冷却程度平滑调整。
        """
        temperature = np.asarray(cloud_temperature_k, dtype=float)
        height_km = np.clip(np.asarray(cloud_height_m, dtype=float) / 1000.0, 0.0, 20.0)
        temperature = np.where(np.isfinite(temperature), temperature, 273.15 - 6.5 * height_km)
        coldness = np.clip((273.15 - temperature) / 40.0, 0.0, 1.5)
        effective_radius = np.clip(9.0 + 0.35 * height_km + 2.0 * coldness, 6.0, 20.0)
        liquid_water_path = np.clip(45.0 + 7.5 * height_km + 25.0 * coldness, 20.0, 250.0)
        return effective_radius, liquid_water_path

    def _grid_cloud_microphysics(self, grid: EarthEnvironmentGrid) -> tuple[np.ndarray, np.ndarray]:
        estimated_re, estimated_lwp = self.estimate_cloud_microphysics(
            grid.cloud_top_temperature_k, grid.cloud_top_height_m
        )
        cloud_re = (
            np.asarray(grid.cloud_effective_radius_um, dtype=float)
            if grid.cloud_effective_radius_um is not None else estimated_re
        )
        cloud_lwp = (
            np.asarray(grid.cloud_liquid_water_path_g_m2, dtype=float)
            if grid.cloud_liquid_water_path_g_m2 is not None else estimated_lwp
        )
        cloud_re = np.where(np.isfinite(cloud_re) & (cloud_re > 0.0), cloud_re, estimated_re)
        cloud_lwp = np.where(np.isfinite(cloud_lwp) & (cloud_lwp > 0.0), cloud_lwp, estimated_lwp)
        return np.clip(cloud_re, 2.0, 60.0), np.clip(cloud_lwp, 1.0, 3000.0)

    def _fill_missing(self, values: np.ndarray, default: float) -> np.ndarray:
        array = np.asarray(values, dtype=float).copy()
        valid = np.isfinite(array) & (array > 0.0)
        fill = float(np.nanmean(array[valid])) if valid.any() else default
        array[~valid] = fill
        return array

    def _surface_properties(self, land_type: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        surface_type = np.asarray(land_type)
        albedo = np.full(surface_type.shape, 0.18, dtype=float)
        emissivity = np.full(surface_type.shape, 0.96, dtype=float)
        valid = (
            np.isfinite(surface_type)
            & (surface_type >= 0)
            & (surface_type < self.SURFACE_ALBEDO.size)
        )
        indices = surface_type[valid].astype(np.intp)
        albedo[valid] = self.SURFACE_ALBEDO[indices]
        emissivity[valid] = self.SURFACE_EMISSIVITY[indices]
        return albedo, emissivity


class LayeredAtmosphereSolver:
    """35层宽带辐射传输核。

    采用Beer-Lambert透过率、层源函数和δ-Eddington型散射增强。
    接口保留光谱扩展能力；参数均使用SI单位。
    """

    LAYER_HEIGHT_KM = np.asarray([1] * 10 + [2] * 6 + [3] * 10 + [5] * 9, dtype=float)
    # OPAC在RH=80%时的代表光学锚点（Hess et al., 1998, Table 3）：
    # (SSA550, g550, Angstrom 0.5-0.8 μm, 代表模态半径μm, 几何标准差)。
    OPAC_TYPE_PROPERTIES = {
        "continental_clean": (0.972, 0.709, 1.42, 0.12, 2.00),
        "continental_average": (0.925, 0.703, 1.42, 0.11, 2.05),
        "urban": (0.817, 0.689, 1.43, 0.08, 1.95),
        "desert": (0.888, 0.729, 0.17, 0.55, 2.20),
        "maritime_clean": (0.997, 0.772, 0.08, 0.65, 2.15),
        "arctic": (0.887, 0.721, 0.89, 0.18, 2.00),
        "antarctic": (1.000, 0.784, 0.73, 0.20, 2.05),
    }
    _solar_reference_wavenumber_cm: np.ndarray | None = None
    _solar_reference_irradiance_per_cm: np.ndarray | None = None

    def __init__(self) -> None:
        bottom = np.r_[0.0, np.cumsum(self.LAYER_HEIGHT_KM[:-1])]
        self.altitude_mid_km = bottom + 0.5 * self.LAYER_HEIGHT_KM
        self._standard_temperature_k = np.interp(
            self.altitude_mid_km,
            [0, 11, 20, 32, 47, 51, 71, 86, 100],
            [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87, 195.08],
        )
        self.temperature_k = self._standard_temperature_k.copy()
        self._upper_atmosphere_temperature_offset_k = 0.0
        self._absorption_wavenumber: np.ndarray | None = None
        self._absorption_tau: np.ndarray | None = None
        self._absorption_tau_original: np.ndarray | None = None
        self._absorption_sources: list[str] = []
        self._optical_depth_corrections: list[dict[str, float]] = []
        self._aerosol_mie_cache: dict[tuple[float, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cloud_mie_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def set_upper_atmosphere_temperature_offset(
        self, offset_k: float, minimum_altitude_km: float = 10.0
    ) -> float:
        """从标准廓线重建温度，并只偏移指定高度以上的大气层。"""
        offset = float(np.clip(float(offset_k), -15.0, 15.0))
        self.temperature_k = self._standard_temperature_k.copy()
        self.temperature_k[self.altitude_mid_km >= float(minimum_altitude_km)] += offset
        self._upper_atmosphere_temperature_offset_k = offset
        return offset

    def load_absorption_optical_depth(self, file_path: str | Path) -> None:
        """加载 ARTESolver 使用的“波数 + 每层总光学厚度”CSV。

        文件可以包含35层或其他层数；层数不一致时按归一化高度插值到35层。
        每列已经是所有气体组分合成后的总量，数据按从地面到大气顶保存。
        """
        source = Path(file_path).expanduser().resolve()
        wavenumber, tau = self._read_absorption_csv(source)
        self._absorption_wavenumber = wavenumber
        self._absorption_tau_original = tau.copy()
        self._absorption_tau = tau.copy()
        self._absorption_sources = [str(source)]
        self._optical_depth_corrections = []

    @staticmethod
    def normalize_optical_depth_corrections(
        corrections: Any,
    ) -> list[dict[str, float]]:
        """校验并标准化分波段总光学厚度倍率。"""
        if corrections in (None, ""):
            return []
        if not isinstance(corrections, (list, tuple)):
            raise ValueError("总光学厚度修正参数必须是波段列表。")
        normalized: list[dict[str, float]] = []
        for index, correction in enumerate(corrections, start=1):
            if not isinstance(correction, dict):
                raise ValueError(f"第{index}个光学厚度修正波段格式无效。")
            try:
                wavelength_min = float(correction["wavelength_min_um"])
                wavelength_max = float(correction["wavelength_max_um"])
                factor = float(correction["factor"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"第{index}个光学厚度修正波段缺少有效参数。") from exc
            if not np.isfinite([wavelength_min, wavelength_max, factor]).all():
                raise ValueError(f"第{index}个光学厚度修正波段含非有限数值。")
            if wavelength_min <= 0.0 or wavelength_max <= wavelength_min:
                raise ValueError(f"第{index}个光学厚度修正波段的波长范围无效。")
            if factor <= 0.0 or factor > 100.0:
                raise ValueError(f"第{index}个光学厚度修正倍率必须在0～100之间。")
            normalized.append({
                "wavelength_min_um": wavelength_min,
                "wavelength_max_um": wavelength_max,
                "factor": factor,
            })
        normalized.sort(key=lambda item: item["wavelength_min_um"])
        for previous, current in zip(normalized, normalized[1:]):
            if current["wavelength_min_um"] < previous["wavelength_max_um"]:
                raise ValueError("总光学厚度修正波段不能相互重叠。")
        return normalized

    def apply_optical_depth_corrections(self, corrections: Any) -> list[dict[str, float]]:
        """从原始输入重新生成修正后的逐层总光学厚度。"""
        if self._absorption_wavenumber is None or self._absorption_tau_original is None:
            raise RuntimeError("请先导入总气体分子光学厚度文件。")
        normalized = self.normalize_optical_depth_corrections(corrections)
        if normalized == self._optical_depth_corrections and self._absorption_tau is not None:
            return [dict(item) for item in normalized]
        corrected = self._absorption_tau_original.copy()
        correction_factor = np.ones(self._absorption_wavenumber.shape, dtype=float)
        applied_count = 0
        for correction in normalized:
            spectral_slice = self._correction_wavenumber_slice(correction)
            if spectral_slice.start == spectral_slice.stop:
                # 卫星通道可能比输入光学厚度网格更密；无对应源波数时保持原值。
                continue
            correction_factor[spectral_slice] = correction["factor"]
            applied_count += 1
        if normalized and applied_count == 0:
            raise ValueError("总光学厚度数据未覆盖任何待修正卫星通道。")
        corrected *= correction_factor[None, :]
        self._absorption_tau = corrected
        self._optical_depth_corrections = normalized
        return [dict(item) for item in normalized]

    def clear_optical_depth_corrections(self) -> None:
        if not self._optical_depth_corrections:
            return
        if self._absorption_tau_original is not None:
            self._absorption_tau = self._absorption_tau_original.copy()
        self._optical_depth_corrections = []

    def get_optical_depth_corrections(self) -> list[dict[str, float]]:
        return [dict(item) for item in self._optical_depth_corrections]

    def _correction_wavenumber_slice(self, correction: dict[str, float]) -> slice:
        if self._absorption_wavenumber is None:
            return slice(0, 0)
        lower = 10_000.0 / correction["wavelength_max_um"]
        upper = 10_000.0 / correction["wavelength_min_um"]
        tolerance = 1.0e-9 * max(abs(lower), abs(upper), 1.0)
        start = int(np.searchsorted(self._absorption_wavenumber, lower - tolerance, side="left"))
        stop = int(np.searchsorted(self._absorption_wavenumber, upper + tolerance, side="right"))
        return slice(start, stop)

    def save_corrected_absorption_optical_depth(self, file_path: str | Path) -> Path:
        if self._absorption_wavenumber is None or self._absorption_tau is None:
            raise RuntimeError("尚未加载可保存的总气体分子光学厚度。")
        target = Path(file_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        values = np.column_stack((self._absorption_wavenumber, self._absorption_tau.T))
        header = ",".join(
            ["wavenumber_cm"]
            + [f"layer_{index + 1:02d}_tau" for index in range(self._absorption_tau.shape[0])]
        )
        np.savetxt(target, values, delimiter=",", header=header, comments="", fmt="%.12e")
        return target

    def load_corrected_absorption_optical_depth(
        self, file_path: str | Path, corrections: Any
    ) -> None:
        """加载项目内修正版，同时恢复可重复应用倍率的基准光学厚度。"""
        self.load_absorption_optical_depth(file_path)
        normalized = self.normalize_optical_depth_corrections(corrections)
        if self._absorption_wavenumber is None or self._absorption_tau is None:
            raise RuntimeError("项目内修正光学厚度加载失败。")
        original = self._absorption_tau.copy()
        correction_factor = np.ones(self._absorption_wavenumber.shape, dtype=float)
        applied_count = 0
        for correction in normalized:
            spectral_slice = self._correction_wavenumber_slice(correction)
            if spectral_slice.start == spectral_slice.stop:
                continue
            correction_factor[spectral_slice] = correction["factor"]
            applied_count += 1
        if normalized and applied_count == 0:
            raise ValueError("项目内修正光学厚度与保存的修正通道不匹配。")
        original /= correction_factor[None, :]
        self._absorption_tau_original = original
        self.apply_optical_depth_corrections(normalized)

    def get_absorption_visualization_data(self) -> dict[str, Any]:
        if self._absorption_wavenumber is None or self._absorption_tau is None:
            raise RuntimeError("尚未导入总气体分子光学厚度文件。")
        wavenumber = np.asarray(self._absorption_wavenumber, dtype=float)
        solar_irradiance = np.full(wavenumber.shape, np.nan, dtype=float)
        solar_band = (
            (wavenumber >= SPECTRAL_WAVENUMBER_MIN_CM)
            & (wavenumber <= SPECTRAL_WAVENUMBER_MAX_CM)
        )
        if np.any(solar_band):
            solar_irradiance[solar_band] = self._solar_spectral_irradiance(
                wavenumber[solar_band]
            )
        return {
            "wavenumber_cm": wavenumber,
            "total_tau_layers": self._absorption_tau,
            "sources": list(self._absorption_sources),
            "optical_depth_corrections": self.get_optical_depth_corrections(),
            "solar_spectral_irradiance_w_m2_per_cm": solar_irradiance,
            "solar_spectrum_source": SOLAR_REFERENCE_SOURCE,
        }

    def _read_absorption_csv(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        with source.open("r", encoding="utf-8-sig", errors="ignore") as stream:
            first_line = stream.readline().strip()
        first_token = first_line.split(",", 1)[0].strip()
        try:
            float(first_token)
            skip_rows = 0
        except ValueError:
            skip_rows = 1
        try:
            raw = np.loadtxt(source, delimiter=",", ndmin=2, skiprows=skip_rows)
        except ValueError:
            raw = np.loadtxt(source, ndmin=2, skiprows=skip_rows)
        if raw.shape[1] < 2 or raw.shape[0] < 2:
            raise ValueError("总气体分子光学厚度文件至少需要两行和两列。")
        wavenumber = np.asarray(raw[:, 0], dtype=float)
        tau = np.maximum(np.asarray(raw[:, 1:], dtype=float).T, 0.0)
        order = np.argsort(wavenumber)
        wavenumber = wavenumber[order]
        tau = tau[:, order]
        if tau.shape[0] != len(self.LAYER_HEIGHT_KM):
            old_height = np.linspace(0.0, 1.0, tau.shape[0])
            new_height = np.linspace(0.0, 1.0, len(self.LAYER_HEIGHT_KM))
            tau = np.vstack([np.interp(new_height, old_height, tau[:, index]) for index in range(tau.shape[1])]).T
        return wavenumber, tau


    def spectral_radiance(
        self,
        wavenumber_cm: np.ndarray,
        surface_temperature_k: float,
        surface_albedo: float,
        surface_emissivity: float,
        cloud_fraction: float,
        cloud_temperature_k: float,
        cloud_height_m: float,
        solar_mu: float,
        view_mu: float,
        visibility_km: float,
        aerosol_optical_depth_550: float | None = None,
        aerosol_type: str = "continental_average",
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算大气顶热辐射及反射太阳光谱辐亮度。

        输出单位为 W/(m²·sr·cm⁻¹)。反射太阳项由单次散射、δ-Eddington
        多次散射和经大气消光后的地表反射三部分组成。
        """
        result = self.spectral_radiance_batch(
            wavenumber_cm,
            np.asarray([surface_temperature_k]),
            np.asarray([surface_albedo]),
            np.asarray([surface_emissivity]),
            np.asarray([cloud_fraction]),
            np.asarray([cloud_temperature_k]),
            np.asarray([cloud_height_m]),
            np.asarray([solar_mu]),
            np.asarray([view_mu]),
            visibility_km,
            enable_scattering=True,
            aerosol_optical_depth_550=aerosol_optical_depth_550,
            aerosol_type=aerosol_type,
        )
        return result[0][0], result[1][0]

    def spectral_radiance_batch(
        self,
        wavenumber_cm: np.ndarray,
        surface_temperature_k: np.ndarray,
        surface_albedo: np.ndarray,
        surface_emissivity: np.ndarray,
        cloud_fraction: np.ndarray,
        cloud_temperature_k: np.ndarray,
        cloud_height_m: np.ndarray,
        solar_mu: np.ndarray,
        view_mu: np.ndarray,
        visibility_km: float,
        enable_scattering: bool = True,
        scattering_cosine: np.ndarray | None = None,
        return_solar_components: bool = False,
        cloud_effective_radius_um: np.ndarray | None = None,
        cloud_liquid_water_path_g_m2: np.ndarray | None = None,
        aerosol_optical_depth_550: np.ndarray | float | None = None,
        aerosol_type: np.ndarray | str | None = None,
    ) -> tuple[np.ndarray, ...]:
        """批量计算地球网格的大气顶光谱辐亮度。

        输出形状为 ``(网格数, 波数点数)``，单位为
        W/(m²·sr·cm⁻¹)。该接口与 :meth:`spectral_radiance` 使用相同的
        35层传输模型，但共享气体、气溶胶和瑞利光学厚度以提高预览速度。
        ``return_solar_components=True`` 时额外返回单次散射、多次散射和
        经大气吸收后的地表反射太阳辐亮度。
        """
        wavenumber = np.asarray(wavenumber_cm, dtype=float).reshape(-1)
        if wavenumber.size < 2 or np.any(~np.isfinite(wavenumber)) or np.any(wavenumber <= 0.0):
            raise ValueError("光谱波数至少需要两个有限正值。")
        arrays = np.broadcast_arrays(
            np.asarray(surface_temperature_k, dtype=float),
            np.asarray(surface_albedo, dtype=float),
            np.asarray(surface_emissivity, dtype=float),
            np.asarray(cloud_fraction, dtype=float),
            np.asarray(cloud_temperature_k, dtype=float),
            np.asarray(cloud_height_m, dtype=float),
            np.asarray(solar_mu, dtype=float),
            np.asarray(view_mu, dtype=float),
        )
        temperature, albedo, emissivity, cloud, cloud_temperature, cloud_height, mu0, mu = (
            np.asarray(value, dtype=float).reshape(-1) for value in arrays
        )
        if scattering_cosine is None:
            cosine = (
                mu0 * mu
                + np.sqrt(np.maximum(0.0, 1.0 - mu0**2))
                * np.sqrt(np.maximum(0.0, 1.0 - mu**2))
            )
        else:
            cosine = np.broadcast_to(np.asarray(scattering_cosine, dtype=float), arrays[0].shape).reshape(-1)
        cosine = np.clip(cosine, -1.0, 1.0)
        mu = np.clip(mu, 0.03, 1.0)
        positive_mu0 = np.clip(mu0, 0.0, 1.0)
        cloud = np.clip(cloud, 0.0, 1.0)
        cloud_height = np.maximum(cloud_height, 0.0)
        fallback_cloud_temperature = np.clip(temperature - 6.5 * cloud_height / 1000.0, 180.0, 330.0)
        cloud_temperature = np.where(np.isfinite(cloud_temperature), cloud_temperature, fallback_cloud_temperature)
        missing_cloud_height = (cloud > 0.0) & (cloud_height <= 0.0)
        diagnosed_height = np.clip(
            (temperature - cloud_temperature) / 6.5 * 1000.0,
            500.0,
            12_000.0,
        )
        cloud_height = np.where(missing_cloud_height, diagnosed_height, cloud_height)
        estimated_re, estimated_lwp = ModisDataManager.estimate_cloud_microphysics(
            cloud_temperature, cloud_height
        )
        if cloud_effective_radius_um is None:
            cloud_re = estimated_re.reshape(-1)
        else:
            cloud_re = np.broadcast_to(
                np.asarray(cloud_effective_radius_um, dtype=float), arrays[0].shape
            ).reshape(-1)
            cloud_re = np.where(np.isfinite(cloud_re) & (cloud_re > 0.0), cloud_re, estimated_re)
        if cloud_liquid_water_path_g_m2 is None:
            cloud_lwp = estimated_lwp.reshape(-1)
        else:
            cloud_lwp = np.broadcast_to(
                np.asarray(cloud_liquid_water_path_g_m2, dtype=float), arrays[0].shape
            ).reshape(-1)
            cloud_lwp = np.where(np.isfinite(cloud_lwp) & (cloud_lwp > 0.0), cloud_lwp, estimated_lwp)
        cloud_re = np.clip(cloud_re, 2.0, 60.0)
        cloud_lwp = np.clip(cloud_lwp, 1.0, 3000.0)

        wavelength_um = 10_000.0 / wavenumber
        if self._absorption_tau is None or self._absorption_wavenumber is None:
            gas_total = (
                0.03
                + 0.25 * np.exp(-((wavelength_um - 6.3) / 1.2) ** 2)
                + 0.35 * np.exp(-((wavelength_um - 15.0) / 2.0) ** 2)
            )
            density_weight = np.exp(-self.altitude_mid_km / 7.5) * self.LAYER_HEIGHT_KM
            density_weight /= density_weight.sum()
            gas_tau = density_weight[:, None] * gas_total[None, :]
        else:
            gas_tau = np.vstack([
                np.interp(wavenumber, self._absorption_wavenumber, layer, left=layer[0], right=layer[-1])
                for layer in self._absorption_tau
            ])
        visibility = max(float(visibility_km), 1.0)
        if aerosol_type is None:
            aerosol_types = np.full(temperature.size, "continental_average", dtype="<U24")
        else:
            aerosol_types = np.broadcast_to(
                np.asarray(aerosol_type, dtype=str), arrays[0].shape
            ).reshape(-1)
        supported_types = set(self.OPAC_TYPE_PROPERTIES)
        aerosol_types = np.asarray([
            value if value in supported_types else "continental_average"
            for value in aerosol_types
        ], dtype="<U24")
        aerosol_extinction_batch = np.empty(
            (temperature.size, len(self.LAYER_HEIGHT_KM), wavenumber.size), dtype=float
        )
        aerosol_scattering_batch = np.empty_like(aerosol_extinction_batch)
        aerosol_asymmetry_batch = np.empty((temperature.size, wavenumber.size), dtype=float)
        rayleigh_tau: np.ndarray | None = None
        for type_name in np.unique(aerosol_types):
            extinction, scattering, asymmetry, type_rayleigh = self._arte_scattering_properties(
                wavenumber, visibility, str(type_name)
            )
            selected_type = aerosol_types == type_name
            aerosol_extinction_batch[selected_type] = extinction
            aerosol_scattering_batch[selected_type] = scattering
            aerosol_asymmetry_batch[selected_type] = asymmetry
            if rayleigh_tau is None:
                rayleigh_tau = type_rayleigh
        if rayleigh_tau is None:
            raise RuntimeError("未能构建OPAC气溶胶光学性质。")
        estimated_aod_550 = 3.912 / visibility
        if aerosol_optical_depth_550 is None:
            aerosol_scale = np.ones(temperature.size, dtype=float)
        else:
            requested_aod = np.broadcast_to(
                np.asarray(aerosol_optical_depth_550, dtype=float), arrays[0].shape
            ).reshape(-1)
            aerosol_scale = np.divide(
                np.where(np.isfinite(requested_aod) & (requested_aod >= 0.0), requested_aod, estimated_aod_550),
                estimated_aod_550,
            )
        aerosol_extinction_batch *= aerosol_scale[:, None, None]
        aerosol_scattering_batch *= aerosol_scale[:, None, None]

        cloud_layer = np.argmin(
            np.abs(self.altitude_mid_km[None, :] * 1000.0 - cloud_height[:, None]), axis=1
        )
        layer_count = len(self.LAYER_HEIGHT_KM)
        row_index = np.arange(temperature.size)
        clear_tau_ext = (
            gas_tau[None, :, :] + aerosol_extinction_batch + rayleigh_tau[None, :, :]
        )
        clear_tau_sca = aerosol_scattering_batch + rayleigh_tau[None, :, :]
        clear_asymmetry = np.broadcast_to(
            aerosol_asymmetry_batch[:, None, :], clear_tau_ext.shape
        ).copy()
        rayleigh_tau_batch = np.broadcast_to(rayleigh_tau[None, :, :], clear_tau_ext.shape)

        cloudy_tau_ext = clear_tau_ext.copy()
        cloudy_tau_sca = clear_tau_sca.copy()
        cloudy_asymmetry = clear_asymmetry.copy()
        if np.any(cloud > 0.0):
            cloud_extinction_tau, cloud_scattering_tau, cloud_asymmetry = self._cloud_mie_properties(
                wavenumber,
                np.where(cloud > 0.0, cloud_re, 10.0),
                np.where(cloud > 0.0, cloud_lwp, 60.0),
            )
        else:
            cloud_extinction_tau = np.zeros((temperature.size, wavenumber.size), dtype=float)
            cloud_scattering_tau = np.zeros_like(cloud_extinction_tau)
            cloud_asymmetry = np.broadcast_to(
                aerosol_asymmetry_batch, cloud_extinction_tau.shape
            ).copy()
        cloudy_tau_ext[row_index, cloud_layer, :] += cloud_extinction_tau
        cloudy_tau_sca[row_index, cloud_layer, :] += cloud_scattering_tau
        clear_scattering_at_cloud = clear_tau_sca[row_index, cloud_layer, :]
        cloudy_asymmetry[row_index, cloud_layer, :] = np.divide(
            clear_asymmetry[row_index, cloud_layer, :] * clear_scattering_at_cloud
            + cloud_asymmetry * cloud_scattering_tau,
            clear_scattering_at_cloud + cloud_scattering_tau,
            out=clear_asymmetry[row_index, cloud_layer, :].copy(),
            where=(clear_scattering_at_cloud + cloud_scattering_tau) > 1e-14,
        )

        clear_thermal = self._arte_thermal_multiple_scattering(
            wavenumber,
            clear_tau_ext,
            clear_tau_sca,
            clear_asymmetry,
            temperature,
            emissivity,
            mu,
        )
        cloudy_thermal = self._arte_thermal_multiple_scattering(
            wavenumber,
            cloudy_tau_ext,
            cloudy_tau_sca,
            cloudy_asymmetry,
            temperature,
            emissivity,
            mu,
        )

        clear_solar = self._solar_reflection_components(
            self._solar_spectral_irradiance(wavenumber),
            clear_tau_ext,
            clear_tau_sca,
            rayleigh_tau_batch,
            clear_asymmetry,
            albedo,
            positive_mu0,
            mu,
            cosine,
            enable_scattering,
        )
        cloudy_solar = self._solar_reflection_components(
            self._solar_spectral_irradiance(wavenumber),
            cloudy_tau_ext,
            cloudy_tau_sca,
            rayleigh_tau_batch,
            cloudy_asymmetry,
            albedo,
            positive_mu0,
            mu,
            cosine,
            enable_scattering,
        )
        cloud_weight = cloud[:, None]
        thermal = (1.0 - cloud_weight) * clear_thermal + cloud_weight * cloudy_thermal
        single_scattering, multiple_scattering, surface_reflection = (
            (1.0 - cloud_weight) * clear_component + cloud_weight * cloudy_component
            for clear_component, cloudy_component in zip(clear_solar, cloudy_solar)
        )
        reflected = single_scattering + multiple_scattering + surface_reflection
        if return_solar_components:
            return (
                np.maximum(thermal, 0.0),
                np.maximum(reflected, 0.0),
                single_scattering,
                multiple_scattering,
                surface_reflection,
            )
        return np.maximum(thermal, 0.0), np.maximum(reflected, 0.0)

    def _arte_thermal_multiple_scattering(
        self,
        wavenumber_cm: np.ndarray,
        tau_extinction: np.ndarray,
        tau_scattering: np.ndarray,
        asymmetry: np.ndarray,
        surface_temperature_k: np.ndarray,
        surface_emissivity: np.ndarray,
        view_mu: np.ndarray,
    ) -> np.ndarray:
        """移植ARTESolver.theramlRadMulScatter的热辐射Adding递推。"""
        tau = np.maximum(tau_extinction[:, ::-1, :], 0.0)
        scattering = np.clip(tau_scattering[:, ::-1, :], 0.0, tau)
        g = np.clip(asymmetry[:, ::-1, :], 0.0, 0.98)
        omega = np.clip(
            np.divide(scattering, tau, out=np.zeros_like(scattering), where=tau > 1e-14),
            0.0,
            1.0 - 1e-6,
        )
        wavenumber = np.asarray(wavenumber_cm, dtype=float)
        surface_temperature = np.maximum(np.asarray(surface_temperature_k, dtype=float), 1.0)
        emissivity = np.clip(np.asarray(surface_emissivity, dtype=float), 0.0, 1.0)
        thermal_albedo = 1.0 - emissivity
        first_radiation_constant = 1.190956e-12
        second_radiation_constant = 1.438786
        earth_planck = (
            first_radiation_constant
            * wavenumber[None, :] ** 3
            / np.expm1(second_radiation_constant * wavenumber[None, :] / surface_temperature[:, None])
        )
        layer_temperature = self.temperature_k[::-1]
        layer_planck = (
            first_radiation_constant
            * wavenumber[None, :] ** 3
            / np.expm1(second_radiation_constant * wavenumber[None, :] / layer_temperature[:, None])
        )

        coefficient_k = 1.0 / np.sqrt(3.0) * np.sqrt(
            np.maximum((1.0 - omega * g) * (1.0 - omega), 0.0)
        )
        coefficient_a = np.sqrt(
            np.divide(1.0 - omega, np.maximum(1.0 - omega * g, 1e-30))
        )
        xi = coefficient_k * tau
        xi_safe = np.minimum(xi, 500.0)
        exp_positive = np.exp(xi_safe)
        exp_negative = np.exp(-xi_safe)
        overflow = xi > 500.0
        denominator_rt = (
            (1.0 + coefficient_a) ** 2 * exp_positive
            - (1.0 - coefficient_a) ** 2 * exp_negative
        )
        denominator_rt = self._signed_floor(denominator_rt, 1e-30)
        layer_reflection = (
            (1.0 - coefficient_a**2) * exp_positive
            - (1.0 - coefficient_a**2) * exp_negative
        ) / denominator_rt
        layer_transmission = 4.0 * coefficient_a / denominator_rt
        layer_reflection[overflow] = 1.0
        layer_transmission[overflow] = 0.0
        layer_reflection = np.clip(layer_reflection, 0.0, 1.0)
        layer_transmission = np.maximum(layer_transmission, 0.0)

        denominator_uv = self._signed_floor(xi_safe * exp_positive, 1e-30)
        numerator_u = (
            (coefficient_a + 1.0) * exp_positive
            + (coefficient_a - 1.0) * exp_negative
            - 2.0 * xi_safe
            - 2.0 * coefficient_a
        )
        numerator_v = (
            (coefficient_a + 1.0) * (xi_safe - 1.0) * exp_positive
            + (1.0 - coefficient_a) * (xi_safe + 1.0) * exp_negative
            + 2.0 * coefficient_a
        )
        source_u = numerator_u / denominator_uv
        source_v = numerator_v / denominator_uv
        thin = xi < 1e-4
        source_u[thin] = coefficient_a[thin] * xi[thin] * (1.0 / 3.0 - xi[thin] / 12.0)
        source_v[thin] = coefficient_a[thin] * xi[thin] * (1.0 / 3.0 + xi[thin] / 12.0)
        source_u[overflow] = coefficient_a[overflow] - 1.0
        source_v[overflow] = coefficient_a[overflow] + 1.0

        batch_count, layer_count, spectral_count = tau.shape
        upward_reflectance = np.broadcast_to(
            thermal_albedo[:, None, None], (batch_count, layer_count, spectral_count)
        ).copy()
        upward_transmission = np.zeros_like(tau)
        downward_reflectance = np.zeros_like(tau)
        downward_transmission = np.zeros_like(tau)
        for layer in range(layer_count - 1, 0, -1):
            target = layer - 1
            denominator_up = self._signed_floor(
                1.0 - upward_reflectance[:, layer, :] * layer_reflection[:, target, :], 1e-30
            )
            upward_transmission[:, target, :] = layer_transmission[:, target, :] / denominator_up
            upward_reflectance[:, target, :] = (
                layer_reflection[:, target, :]
                + upward_transmission[:, target, :]
                * layer_transmission[:, target, :]
                * upward_reflectance[:, layer, :]
            )
            denominator_down = self._signed_floor(
                1.0 - downward_reflectance[:, layer, :] * layer_reflection[:, target, :], 1e-30
            )
            downward_transmission[:, target, :] = layer_transmission[:, target, :] / denominator_down
            downward_reflectance[:, target, :] = (
                layer_reflection[:, target, :]
                + downward_transmission[:, target, :]
                * layer_transmission[:, target, :]
                * downward_reflectance[:, layer, :]
            )

        denominator_source = self._signed_floor(
            (coefficient_a + 1.0) ** 2 * exp_positive
            + (coefficient_a - 1.0) ** 2 * exp_negative,
            1e-30,
        )
        downward_source = np.empty_like(tau)
        upward_source = np.empty_like(tau)
        for layer in range(layer_count - 1, -1, -1):
            if layer > 0:
                lower_planck = layer_planck[layer - 1]
                upper_planck = layer_planck[layer]
            else:
                lower_planck = earth_planck
                upper_planck = layer_planck[layer]
            downward_source[:, layer, :] = (
                2.0
                * coefficient_a[:, layer, :]
                * (source_u[:, layer, :] * lower_planck + source_v[:, layer, :] * upper_planck)
                / denominator_source[:, layer, :]
            )
            upward_source[:, layer, :] = (
                2.0
                * coefficient_a[:, layer, :]
                * (source_u[:, layer, :] * upper_planck + source_v[:, layer, :] * lower_planck)
                / denominator_source[:, layer, :]
            )

        downward_boundary = np.zeros_like(tau)
        upward_boundary = np.zeros_like(tau)
        upward_boundary[:, -1, :] = emissivity[:, None] * earth_planck
        for layer in range(layer_count - 2, -1, -1):
            upward_boundary[:, layer, :] = (
                upward_source[:, layer, :]
                + upward_transmission[:, layer, :]
                * (
                    upward_boundary[:, layer + 1, :]
                    + upward_reflectance[:, layer + 1, :] * downward_source[:, layer, :]
                )
            )
        for layer in range(1, layer_count):
            downward_boundary[:, layer, :] = (
                downward_source[:, layer - 1, :]
                + downward_transmission[:, layer - 1, :]
                * (
                    downward_boundary[:, layer - 1, :]
                    + downward_reflectance[:, layer - 1, :] * upward_source[:, layer - 1, :]
                )
            )

        upward_flux = np.zeros_like(tau)
        downward_flux = np.zeros_like(tau)
        for layer in range(1, layer_count):
            denominator_combined = self._signed_floor(
                1.0 - downward_reflectance[:, layer - 1, :] * layer_reflection[:, layer - 1, :],
                1e-30,
            )
            if layer < layer_count - 1:
                upward_flux[:, layer, :] = (
                    upward_boundary[:, layer, :]
                    + downward_boundary[:, layer, :] * upward_reflectance[:, layer, :]
                ) / denominator_combined
                downward_flux[:, layer, :] = (
                    downward_boundary[:, layer, :]
                    + upward_boundary[:, layer, :] * downward_reflectance[:, layer, :]
                ) / denominator_combined
            else:
                downward_flux[:, layer, :] = (
                    downward_boundary[:, layer - 1, :]
                    + upward_boundary[:, layer, :] * downward_reflectance[:, layer - 1, :]
                ) / denominator_combined
                upward_flux[:, layer, :] = (
                    upward_boundary[:, layer, :]
                    + downward_flux[:, layer, :] * upward_reflectance[:, layer, :]
                )
        target_radiance = (
            upward_flux[:, 1, :] - downward_flux[:, 1, :]
        ) * np.abs(np.asarray(view_mu, dtype=float))[:, None]
        return np.maximum(target_radiance, 0.0) * 1e4

    def _solar_reflection_components(
        self,
        solar_spectrum: np.ndarray,
        tau_extinction: np.ndarray,
        tau_scattering: np.ndarray,
        rayleigh_tau: np.ndarray,
        asymmetry: np.ndarray,
        surface_albedo: np.ndarray,
        solar_mu: np.ndarray,
        view_mu: np.ndarray,
        scattering_cosine: np.ndarray,
        enable_scattering: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按ARTESolver的三分量结构计算反射太阳辐亮度。"""
        # Python内部大气层由地面向上排列；参考算法按大气顶向地面递推。
        tau = np.maximum(tau_extinction[:, ::-1, :], 0.0)
        tau_sca = np.clip(tau_scattering[:, ::-1, :], 0.0, tau)
        rayleigh = np.clip(rayleigh_tau[:, ::-1, :], 0.0, tau_sca)
        g = np.clip(asymmetry[:, ::-1, :], 0.0, 0.98)
        mu0 = np.clip(np.asarray(solar_mu, dtype=float), 0.0, 1.0)
        mu0_safe = np.maximum(mu0, 1e-4)[:, None, None]
        mu1 = np.clip(np.asarray(view_mu, dtype=float), 0.03, 1.0)[:, None, None]
        solar = np.asarray(solar_spectrum, dtype=float)[None, None, :]
        albedo = np.clip(np.asarray(surface_albedo, dtype=float), 0.0, 1.0)
        night = mu0 <= 0.0

        tau_total = np.sum(tau, axis=1)
        surface_reflection = (
            solar[:, 0, :]
            * np.exp(-tau_total / mu0_safe[:, 0, :])
            * albedo[:, None]
            * np.exp(-tau_total / mu1[:, 0, :])
            / np.pi
        )
        surface_reflection[night] = 0.0
        if not enable_scattering:
            zeros = np.zeros_like(surface_reflection)
            return zeros, zeros, np.maximum(surface_reflection, 0.0)

        omega = np.divide(tau_sca, tau, out=np.zeros_like(tau_sca), where=tau > 1e-14)
        forward_fraction = g**2
        tau_effective = np.maximum((1.0 - omega * forward_fraction) * tau, 0.0)
        omega_effective = np.divide(
            (1.0 - forward_fraction) * omega,
            np.maximum(1.0 - omega * forward_fraction, 1e-12),
        )
        g_effective = g / (1.0 + g)
        cumulative_tau = np.cumsum(tau_effective, axis=1)
        total_effective_tau = cumulative_tau[:, -1:, :]
        transmission_sun = np.exp(-cumulative_tau / mu0_safe)
        transmission_target = np.exp(-(total_effective_tau - cumulative_tau) / mu1)
        cosine = np.clip(np.asarray(scattering_cosine, dtype=float), -1.0, 1.0)[:, None, None]
        phase_rayleigh = (3.0 / (16.0 * np.pi)) * (1.0 + cosine**2)
        phase_hg = (
            (1.0 - g_effective**2)
            / (
                4.0 * np.pi
                * np.maximum(1.0 + g_effective**2 - 2.0 * g_effective * cosine, 1e-12) ** 1.5
            )
        )
        omega_rayleigh = np.divide(rayleigh, tau, out=np.zeros_like(rayleigh), where=tau > 1e-14)
        omega_non_rayleigh = np.maximum(omega_effective - omega_rayleigh, 0.0)
        scattering_source = omega_rayleigh * phase_rayleigh + omega_non_rayleigh * phase_hg
        single_scattering = np.sum(
            solar
            * transmission_sun / mu0_safe
            * transmission_target / mu1
            * scattering_source
            * tau_effective,
            axis=1,
        ) / np.pi * mu0[:, None] * mu1[:, 0, :]

        multiple_scattering = self._delta_eddington_multiple_scattering(
            solar_spectrum,
            tau,
            omega,
            g,
            albedo,
            mu0,
            np.asarray(view_mu, dtype=float),
        )
        single_scattering[night] = 0.0
        multiple_scattering[night] = 0.0
        return (
            np.maximum(single_scattering, 0.0),
            np.maximum(multiple_scattering, 0.0),
            np.maximum(surface_reflection, 0.0),
        )

    def _delta_eddington_multiple_scattering(
        self,
        solar_spectrum: np.ndarray,
        tau: np.ndarray,
        omega: np.ndarray,
        asymmetry: np.ndarray,
        surface_albedo: np.ndarray,
        solar_mu: np.ndarray,
        view_mu: np.ndarray,
    ) -> np.ndarray:
        """移植ARTESolver.solarMulScatter的δ-Eddington二流递推。"""
        mu0 = np.maximum(np.asarray(solar_mu, dtype=float), 1e-4)[:, None, None]
        mu1 = np.clip(np.asarray(view_mu, dtype=float), 0.03, 1.0)[:, None]
        omega = np.clip(omega, 0.0, 1.0 - 1e-6)
        g = np.clip(asymmetry, 0.0, 0.98)
        gamma1 = (7.0 - omega * (4.0 + 3.0 * g)) / 4.0
        gamma2 = -(1.0 - omega * (4.0 - 3.0 * g)) / 4.0
        gamma3 = (2.0 - 3.0 * g * mu0) / 4.0
        gamma4 = 1.0 - gamma3
        coefficient = np.sqrt(np.maximum(gamma1**2 - gamma2**2, 0.0))
        xi = coefficient * tau
        exp_positive = np.exp(np.minimum(xi, 500.0))
        exp_negative = np.exp(-np.minimum(xi, 500.0))
        direct_exponent = np.minimum(tau / mu0, 500.0)
        direct_transmission = np.exp(-direct_exponent)
        denominator = (
            (1.0 - (coefficient * mu0) ** 2)
            * ((coefficient + gamma1) * exp_positive + (coefficient - gamma1) * exp_negative)
        )
        denominator = self._signed_floor(denominator, 1e-12)
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4
        numerator_reflection = omega * (
            (1.0 - coefficient * mu0) * (alpha2 + coefficient * gamma3) * exp_positive
            - (1.0 + coefficient * mu0) * (alpha2 - coefficient * gamma3) * exp_negative
            - 2.0 * coefficient * (gamma3 - alpha2 * mu0) * direct_transmission
        )
        numerator_transmission = omega * (
            (1.0 + coefficient * mu0) * (alpha1 + coefficient * gamma4) * exp_positive
            - (1.0 - coefficient * mu0) * (alpha1 - coefficient * gamma4) * exp_negative
            - 2.0 * coefficient * (gamma4 + alpha1 * mu0) * np.exp(direct_exponent)
        )
        layer_reflection = np.clip(numerator_reflection / denominator, 0.0, 1.0)
        layer_transmission = np.clip(
            direct_transmission * (1.0 - numerator_transmission / denominator), 0.0, 1.0
        )

        batch_count, layer_count, spectral_count = tau.shape
        upward_reflectance = np.zeros((batch_count, layer_count, spectral_count), dtype=float)
        upward_reflectance[:, -1, :] = np.clip(surface_albedo, 0.0, 1.0)[:, None]
        for layer in range(layer_count - 2, -1, -1):
            denominator_up = self._signed_floor(
                1.0 - layer_reflection[:, layer, :] * upward_reflectance[:, layer + 1, :], 1e-6
            )
            upward_reflectance[:, layer, :] = (
                layer_reflection[:, layer, :]
                + layer_transmission[:, layer, :] ** 2
                * upward_reflectance[:, layer + 1, :]
                / denominator_up
            )

        cumulative_tau = np.cumsum(tau, axis=1)
        direct_irradiance = (
            np.asarray(solar_spectrum, dtype=float)[None, None, :]
            * np.exp(-np.minimum(cumulative_tau / mu0, 500.0))
            * mu0
        )
        upward_source = direct_irradiance * layer_reflection
        upward_flux = np.zeros_like(tau)
        upward_flux[:, -1, :] = (
            np.clip(surface_albedo, 0.0, 1.0)[:, None] * direct_irradiance[:, -1, :]
        )
        for layer in range(layer_count - 2, -1, -1):
            denominator_up = self._signed_floor(
                1.0 - upward_reflectance[:, layer + 1, :] * layer_reflection[:, layer, :], 1e-6
            )
            upward_flux[:, layer, :] = (
                upward_flux[:, layer + 1, :] * layer_transmission[:, layer, :]
                + upward_source[:, layer, :]
            ) / denominator_up
        return np.maximum(upward_flux[:, 0, :], 0.0) / np.pi * mu0[:, 0, :] * mu1

    def _signed_floor(self, values: np.ndarray, minimum: float) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.where(array >= 0.0, np.maximum(array, minimum), -np.maximum(-array, minimum))

    def _arte_scattering_properties(
        self,
        wavenumber_cm: np.ndarray,
        visibility_km: float,
        aerosol_type: str = "continental_average",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按选定OPAC类型构建气溶胶和Rayleigh逐层光学属性。"""
        wavelength_table, beta_ext, beta_sca, aerosol_asymmetry = self._arte_aerosol_mie_table(
            visibility_km, aerosol_type
        )
        table_wavenumber = 10_000.0 / wavelength_table
        order = np.argsort(table_wavenumber)
        target = np.asarray(wavenumber_cm, dtype=float)
        aerosol_ext_total = self._pchip_interpolate(table_wavenumber[order], beta_ext[order], target)
        aerosol_sca_total = self._pchip_interpolate(table_wavenumber[order], beta_sca[order], target)
        aerosol_g = self._pchip_interpolate(table_wavenumber[order], aerosol_asymmetry[order], target)
        aerosol_ext_total = np.maximum(aerosol_ext_total, 0.0)
        aerosol_sca_total = np.clip(aerosol_sca_total, 0.0, aerosol_ext_total)
        aerosol_g = np.clip(aerosol_g, -1.0, 1.0)
        aerosol_profile = (
            np.exp(-self.altitude_mid_km / 1.0) * self.LAYER_HEIGHT_KM
        )[:, None]
        aerosol_profile /= max(float(np.sum(aerosol_profile)), 1e-30)
        aerosol_extinction = aerosol_profile * aerosol_ext_total[None, :]
        aerosol_scattering = aerosol_profile * aerosol_sca_total[None, :]

        layer_mid_m = self.altitude_mid_km * 1000.0
        number_density_surface = 2.547e19
        number_density = number_density_surface * np.exp(-layer_mid_m / 8500.0)
        wavelength_um = 10_000.0 / target
        wavelength_cm = wavelength_um * 1e-4
        inverse_wavelength_squared = (1.0 / wavelength_um) ** 2
        refractivity = 1e-8 * (
            8342.13
            + 2_406_030.0 / (130.0 - inverse_wavelength_squared)
            + 15_997.0 / (38.9 - inverse_wavelength_squared)
        )
        refractive_index = 1.0 + refractivity
        rayleigh_cross_section = (
            24.0 * np.pi**3
            / (wavelength_cm**4 * number_density_surface**2)
            * (refractive_index**2 - 1.0) ** 2
            * 1.055
        )
        rayleigh_tau = (
            number_density[:, None]
            * (self.LAYER_HEIGHT_KM * 1e5)[:, None]
            * rayleigh_cross_section[None, :]
        )
        return aerosol_extinction, aerosol_scattering, aerosol_g, rayleigh_tau

    def _cloud_mie_properties(
        self,
        wavenumber_cm: np.ndarray,
        effective_radius_um: np.ndarray,
        liquid_water_path_g_m2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按ARTESolver.calcCloudMie计算每个有云子柱的Mie光学性质。

        有效半径以0.5 μm分档并缓存单位LWP光谱；光学厚度随LWP
        线性缩放。这保留了修正Gamma粒径分布和Hale–Querry水复折射率，
        同时避免地球盘中相近网格重复求解Mie级数。
        """
        wavenumber = np.asarray(wavenumber_cm, dtype=float)
        radius = np.clip(np.asarray(effective_radius_um, dtype=float).reshape(-1), 2.0, 60.0)
        lwp = np.clip(np.asarray(liquid_water_path_g_m2, dtype=float).reshape(-1), 1.0, 3000.0)
        radius_bin = np.maximum(2.0, np.round(radius / 2.0) * 2.0)
        extinction = np.empty((radius.size, wavenumber.size), dtype=float)
        scattering = np.empty_like(extinction)
        asymmetry = np.empty_like(extinction)
        for value in np.unique(radius_bin):
            wavelength, tau_ext_per_lwp, tau_sca_per_lwp, table_g = self._cloud_mie_table(float(value))
            table_wavenumber = 10_000.0 / wavelength
            order = np.argsort(table_wavenumber)
            ext_base = np.maximum(
                self._pchip_interpolate(table_wavenumber[order], tau_ext_per_lwp[order], wavenumber),
                0.0,
            )
            sca_base = np.clip(
                self._pchip_interpolate(table_wavenumber[order], tau_sca_per_lwp[order], wavenumber),
                0.0,
                ext_base,
            )
            g_spectrum = np.clip(
                self._pchip_interpolate(table_wavenumber[order], table_g[order], wavenumber),
                -1.0,
                1.0,
            )
            selected = radius_bin == value
            # Mie光谱形状按最近2 μm半径档求解；τ≈LWP/r_eff的
            # 连续尺度因子用实际有效半径恢复。
            optical_scale = lwp[selected] * value / radius[selected]
            extinction[selected] = optical_scale[:, None] * ext_base[None, :]
            scattering[selected] = optical_scale[:, None] * sca_base[None, :]
            asymmetry[selected] = g_spectrum[None, :]
        return extinction, np.minimum(scattering, extinction), asymmetry

    def _cloud_mie_table(
        self,
        effective_radius_um: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        radius_key = max(2.0, round(float(effective_radius_um) / 2.0) * 2.0)
        cached = self._cloud_mie_cache.get(radius_key)
        if cached is not None:
            return cached
        wavelength_water = np.asarray([
            0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.40,
            2.60, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00, 4.50, 5.00, 6.00,
            7.00, 8.00, 9.00, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
        ])
        refractive_real = np.asarray([
            1.396, 1.362, 1.349, 1.343, 1.339, 1.337, 1.335, 1.333, 1.332, 1.331, 1.331,
            1.330, 1.329, 1.328, 1.327, 1.324, 1.321, 1.317, 1.312, 1.306, 1.296, 1.279,
            1.242, 1.219, 1.218, 1.310, 1.430, 1.460, 1.420, 1.400, 1.351, 1.300, 1.265,
            1.317, 1.291, 1.262, 1.218, 1.144, 1.200, 1.482, 1.880, 1.820,
        ])
        refractive_imaginary = np.asarray([
            1.1e-7, 3.35e-8, 1.6e-8, 6.0e-9, 1.86e-9, 1.0e-9, 1.0e-9, 1.96e-9,
            1.09e-8, 1.39e-8, 3.11e-8, 8.6e-8, 1.63e-7, 3.68e-7, 2.4e-6, 6.0e-6,
            1.4e-4, 2.4e-4, 1.25e-3, 9.16e-4, 1.16e-3, 1.1e-2, 2.89e-2, 9.6e-2,
            2.0e-1, 1.79e-1, 1.18e-1, 1.34e-1, 1.7e-1, 2.0e-1, 2.25e-1, 2.37e-1,
            3.65e-1, 3.46e-1, 3.07e-1, 3.27e-1, 3.98e-1, 3.46e-1, 2.83e-1, 3.50e-1,
            3.75e-1, 3.90e-1,
        ])
        wavelength = np.arange(0.3, 20.0 + 0.05, 0.1)
        interpolated_real = self._pchip_interpolate(wavelength_water, refractive_real, wavelength)
        interpolated_imaginary = np.maximum(
            10.0 ** self._pchip_interpolate(
                wavelength_water, np.log10(refractive_imaginary), wavelength
            ),
            1e-10,
        )

        effective_variance = 0.1
        droplet_radius = np.logspace(np.log10(0.1), np.log10(radius_key * 10.0), 300)
        distribution = droplet_radius ** ((1.0 - 3.0 * effective_variance) / effective_variance)
        distribution *= np.exp(-droplet_radius / (radius_key * effective_variance))
        distribution /= np.trapezoid(distribution, droplet_radius)
        volume_integral = np.trapezoid(
            (4.0 / 3.0) * np.pi * droplet_radius**3 * distribution,
            droplet_radius,
        )
        water_density_g_m3 = 1.0e6
        column_number_per_lwp = 1.0 / (water_density_g_m3 * volume_integral * 1e-18)
        tau_extinction_per_lwp = np.empty(wavelength.shape, dtype=float)
        tau_scattering_per_lwp = np.empty(wavelength.shape, dtype=float)
        bulk_asymmetry = np.empty(wavelength.shape, dtype=float)
        cross_section = np.pi * droplet_radius**2 * distribution
        for index, wavelength_value in enumerate(wavelength):
            size = 2.0 * np.pi * droplet_radius / wavelength_value
            mie_extinction, mie_scattering, mie_asymmetry = self._mie_efficiencies(
                size,
                complex(interpolated_real[index], -interpolated_imaginary[index]),
            )
            extinction_integral = np.trapezoid(
                cross_section * mie_extinction, droplet_radius
            )
            scattering_integral = np.trapezoid(
                cross_section * mie_scattering, droplet_radius
            )
            asymmetry_integral = np.trapezoid(
                cross_section * mie_scattering * mie_asymmetry, droplet_radius
            )
            tau_extinction_per_lwp[index] = column_number_per_lwp * extinction_integral * 1e-12
            tau_scattering_per_lwp[index] = column_number_per_lwp * scattering_integral * 1e-12
            bulk_asymmetry[index] = asymmetry_integral / max(scattering_integral, 1e-30)
        result = (
            wavelength,
            np.maximum(tau_extinction_per_lwp, 0.0),
            np.clip(tau_scattering_per_lwp, 0.0, tau_extinction_per_lwp),
            np.clip(bulk_asymmetry, -1.0, 1.0),
        )
        self._cloud_mie_cache[radius_key] = result
        return result

    def _arte_aerosol_mie_table(
        self,
        visibility_km: float,
        aerosol_type: str = "continental_average",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        visibility = float(visibility_km)
        type_name = (
            str(aerosol_type)
            if str(aerosol_type) in self.OPAC_TYPE_PROPERTIES
            else "continental_average"
        )
        target_ssa, target_g, target_angstrom, modal_radius, geometric_sigma = (
            self.OPAC_TYPE_PROPERTIES[type_name]
        )
        cache_key = (round(visibility, 8), type_name)
        cached = self._aerosol_mie_cache.get(cache_key)
        if cached is not None:
            return cached
        wavelength_opac = np.asarray([
            0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00,
            3.20, 3.39, 3.50, 3.75, 4.00, 4.50, 5.00, 5.50, 6.00, 6.20,
            6.50, 7.20, 7.90, 8.20, 8.50, 8.70, 9.00, 9.20, 9.50, 9.80,
            10.0, 10.6, 11.0, 11.5, 12.5, 13.0, 14.0, 14.8, 15.0,
            16.4, 17.2, 18.0, 18.5, 20.0, 21.3, 22.5, 25.0, 30.0, 40.0,
        ])
        refractive_real = np.asarray([
            1.500, 1.480, 1.480, 1.480, 1.480, 1.480, 1.475, 1.470, 1.465, 1.460,
            1.455, 1.450, 1.440, 1.430, 1.420, 1.410, 1.400, 1.390, 1.360, 1.310,
            1.310, 1.300, 1.300, 1.300, 1.300, 1.290, 1.280, 1.260, 1.250, 1.240,
            1.235, 1.210, 1.170, 1.145, 1.100, 1.080, 1.060, 1.040, 1.020, 1.005,
            0.990, 0.965, 0.950, 0.935, 0.910, 0.895, 0.870, 0.855, 0.850,
            0.830, 0.825, 0.820, 0.820, 0.800, 0.790, 0.780, 0.760, 0.740, 0.720,
        ])
        refractive_imaginary = np.asarray([
            1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8,
            1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-7, 1e-6,
            2e-4, 1.97e-1, 6.69e-2, 1.51e-2, 7.17e-3, 2.90e-3, 3.69e-3,
            9.97e-3, 9.57e-3, 4.26e-2, 1.15e-1, 1.73e-1, 2.40e-1, 2.95e-1,
            2.89e-1, 2.83e-1, 2.90e-1, 3.20e-1, 3.70e-1, 4.00e-1, 4.20e-1,
            4.30e-1, 4.30e-1, 3.90e-1, 3.70e-1, 3.60e-1, 3.30e-1, 3.20e-1,
            3.20e-1, 3.20e-1, 3.20e-1, 3.10e-1, 3.00e-1, 3.00e-1, 3.00e-1,
            3.00e-1, 3.00e-1, 3.00e-1, 3.00e-1, 3.00e-1, 3.00e-1,
        ])
        wavelength = np.arange(0.3, 20.0 + 0.05, 0.1)
        interpolated_real = self._pchip_interpolate(wavelength_opac, refractive_real, wavelength)
        interpolated_imaginary = 10.0 ** self._pchip_interpolate(
            wavelength_opac, np.log10(refractive_imaginary), wavelength
        )
        radius = np.logspace(np.log10(0.001), np.log10(20.0), 500)
        logarithmic_sigma = np.log(geometric_sigma)
        number_distribution = (
            1.0
            / (radius * logarithmic_sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-(np.log(radius) - np.log(modal_radius)) ** 2 / (2.0 * logarithmic_sigma**2))
        )
        number_distribution /= np.trapezoid(number_distribution, radius)
        visible_real = float(self._pchip_interpolate(wavelength_opac, refractive_real, np.asarray([0.55]))[0])
        visible_imaginary = float(10.0 ** self._pchip_interpolate(
            wavelength_opac, np.log10(refractive_imaginary), np.asarray([0.55])
        )[0])
        visible_size = 2.0 * np.pi * radius / 0.55
        visible_extinction, visible_scattering, visible_asymmetry = self._mie_efficiencies(
            visible_size, complex(visible_real, -visible_imaginary)
        )
        visible_cross_section = np.pi * radius**2 * number_distribution
        visible_integral = np.trapezoid(
            visible_cross_section * visible_extinction, radius
        )
        visible_scattering_integral = np.trapezoid(
            visible_cross_section * visible_scattering, radius
        )
        visible_asymmetry_value = np.trapezoid(
            visible_cross_section * visible_scattering * visible_asymmetry, radius
        ) / max(visible_scattering_integral, 1e-30)
        number_scale = (3.912 / visibility) / (visible_integral * 1e12)

        angstrom_integrals: list[float] = []
        for visible_wavelength in (0.5, 0.8):
            visible_n = float(self._pchip_interpolate(
                wavelength_opac, refractive_real, np.asarray([visible_wavelength])
            )[0])
            visible_k = float(10.0 ** self._pchip_interpolate(
                wavelength_opac, np.log10(refractive_imaginary), np.asarray([visible_wavelength])
            )[0])
            extinction_at_wavelength, _, _ = self._mie_efficiencies(
                2.0 * np.pi * radius / visible_wavelength,
                complex(visible_n, -visible_k),
            )
            angstrom_integrals.append(float(np.trapezoid(
                visible_cross_section * extinction_at_wavelength, radius
            )))
        modeled_angstrom = -np.log(
            max(angstrom_integrals[1], 1e-30) / max(angstrom_integrals[0], 1e-30)
        ) / np.log(0.8 / 0.5)

        beta_extinction = np.empty(wavelength.shape, dtype=float)
        beta_scattering = np.empty(wavelength.shape, dtype=float)
        beta_asymmetry = np.empty(wavelength.shape, dtype=float)
        for index, wavelength_value in enumerate(wavelength):
            size = 2.0 * np.pi * radius / wavelength_value
            extinction, scattering, asymmetry = self._mie_efficiencies(
                size, complex(interpolated_real[index], -max(interpolated_imaginary[index], 1e-10))
            )
            cross_section = np.pi * radius**2 * number_distribution
            beta_extinction[index] = number_scale * np.trapezoid(cross_section * extinction, radius) * 1e12
            beta_scattering[index] = number_scale * np.trapezoid(cross_section * scattering, radius) * 1e12
            beta_asymmetry[index] = number_scale * np.trapezoid(
                cross_section * scattering * asymmetry, radius
            ) * 1e12
        bulk_asymmetry = np.divide(
            beta_asymmetry,
            np.maximum(beta_scattering, 1e-30),
        )
        # 保留逐波长Mie结构，同时用OPAC类型在550 nm的SSA、g和
        # 0.5–0.8 μm Ångström指数校准类型差异。
        angstrom_influence = np.exp(-((wavelength - 0.55) / 1.5) ** 2)
        spectral_tilt = (wavelength / 0.55) ** (
            -(target_angstrom - modeled_angstrom) * angstrom_influence
        )
        beta_extinction *= spectral_tilt
        beta_scattering *= spectral_tilt
        modeled_visible_ssa = visible_scattering_integral / max(visible_integral, 1e-30)
        beta_scattering = np.minimum(
            beta_extinction,
            beta_scattering * target_ssa / max(modeled_visible_ssa, 1e-30),
        )
        bulk_asymmetry = np.clip(
            bulk_asymmetry + (target_g - visible_asymmetry_value), -1.0, 1.0
        )
        result = wavelength, beta_extinction, beta_scattering, bulk_asymmetry
        self._aerosol_mie_cache[cache_key] = result
        return result

    def _mie_efficiencies(
        self,
        size_parameter: np.ndarray,
        refractive_index: complex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bohren-Huffman Mie效率，向量化复现ARTESolver.mieQ。"""
        size = np.asarray(size_parameter, dtype=float)
        maximum_order_by_size = np.floor(2.0 + size + 4.0 * np.cbrt(size) + 0.5).astype(int)
        maximum_order = int(np.max(maximum_order_by_size))
        product = refractive_index * size
        logarithmic_derivative = np.zeros((maximum_order + 2, size.size), dtype=complex)
        for order in range(maximum_order - 1, 0, -1):
            active = order < maximum_order_by_size
            ratio = order / product[active]
            logarithmic_derivative[order, active] = (
                ratio - 1.0 / (logarithmic_derivative[order + 1, active] + ratio)
            )

        extinction_sum = np.zeros(size.shape, dtype=float)
        scattering_sum = np.zeros(size.shape, dtype=float)
        asymmetry_sum = np.zeros(size.shape, dtype=float)
        previous_a = np.zeros(size.shape, dtype=complex)
        previous_b = np.zeros(size.shape, dtype=complex)
        psi_previous = np.sin(size)
        psi_current = np.sin(size) / size - np.cos(size)
        xi_previous = np.sin(size) + 1j * np.cos(size)
        xi_current = np.sin(size) / size - np.cos(size) + 1j * (np.cos(size) / size + np.sin(size))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for order in range(1, maximum_order + 1):
                psi_next = (2.0 * order + 1.0) / size * psi_current - psi_previous
                xi_next = (2.0 * order + 1.0) / size * xi_current - xi_previous
                derivative = logarithmic_derivative[order]
                coefficient_a = (
                    (derivative / refractive_index + order / size) * psi_current - psi_previous
                ) / ((derivative / refractive_index + order / size) * xi_current - xi_previous)
                coefficient_b = (
                    (refractive_index * derivative + order / size) * psi_current - psi_previous
                ) / ((refractive_index * derivative + order / size) * xi_current - xi_previous)
                active = order <= maximum_order_by_size
                extinction_sum += np.where(
                    active, (2.0 * order + 1.0) * np.real(coefficient_a + coefficient_b), 0.0
                )
                scattering_sum += np.where(
                    active, (2.0 * order + 1.0) * (np.abs(coefficient_a) ** 2 + np.abs(coefficient_b) ** 2), 0.0
                )
                asymmetry_term = (
                    order * (order + 2.0) / (order + 1.0)
                    * np.real(previous_a * np.conj(coefficient_a) + previous_b * np.conj(coefficient_b))
                    + (2.0 * order - 1.0) / order * np.real(previous_a * np.conj(previous_b))
                )
                asymmetry_sum += np.where(active, asymmetry_term, 0.0)
                previous_a = coefficient_a
                previous_b = coefficient_b
                psi_previous, psi_current = psi_current, psi_next
                xi_previous, xi_current = xi_current, xi_next
        extinction = 2.0 / size**2 * extinction_sum
        scattering = 2.0 / size**2 * scattering_sum
        asymmetry = np.clip(
            4.0 / size**2 * asymmetry_sum / np.maximum(scattering, 1e-30), -1.0, 1.0
        )
        return extinction, scattering, asymmetry

    def _pchip_interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        query: np.ndarray,
    ) -> np.ndarray:
        """无SciPy依赖的单调分段三次Hermite插值，兼容MATLAB pchip。"""
        x_values = np.asarray(x, dtype=float)
        y_values = np.asarray(y, dtype=float)
        query_values = np.asarray(query, dtype=float)
        interval = np.diff(x_values)
        slope = np.diff(y_values) / interval
        derivative = np.zeros_like(y_values)
        if x_values.size == 2:
            derivative[:] = slope[0]
        else:
            same_sign = slope[:-1] * slope[1:] > 0.0
            weight_left = 2.0 * interval[1:] + interval[:-1]
            weight_right = interval[1:] + 2.0 * interval[:-1]
            derivative[1:-1] = np.where(
                same_sign,
                (weight_left + weight_right)
                / (weight_left / np.where(slope[:-1] != 0.0, slope[:-1], 1.0)
                   + weight_right / np.where(slope[1:] != 0.0, slope[1:], 1.0)),
                0.0,
            )
            derivative[0] = self._pchip_endpoint(interval[0], interval[1], slope[0], slope[1])
            derivative[-1] = self._pchip_endpoint(interval[-1], interval[-2], slope[-1], slope[-2])
        index = np.searchsorted(x_values, query_values, side="right") - 1
        index = np.clip(index, 0, x_values.size - 2)
        local_interval = x_values[index + 1] - x_values[index]
        normalized = (query_values - x_values[index]) / local_interval
        h00 = 2.0 * normalized**3 - 3.0 * normalized**2 + 1.0
        h10 = normalized**3 - 2.0 * normalized**2 + normalized
        h01 = -2.0 * normalized**3 + 3.0 * normalized**2
        h11 = normalized**3 - normalized**2
        return (
            h00 * y_values[index]
            + h10 * local_interval * derivative[index]
            + h01 * y_values[index + 1]
            + h11 * local_interval * derivative[index + 1]
        )

    def _pchip_endpoint(
        self,
        interval: float,
        adjacent_interval: float,
        slope: float,
        adjacent_slope: float,
    ) -> float:
        derivative = ((2.0 * interval + adjacent_interval) * slope - interval * adjacent_slope) / (
            interval + adjacent_interval
        )
        if np.sign(derivative) != np.sign(slope):
            return 0.0
        if np.sign(slope) != np.sign(adjacent_slope) and abs(derivative) > abs(3.0 * slope):
            return 3.0 * slope
        return float(derivative)

    def _planck_wavenumber(self, wavenumber_cm: np.ndarray, temperature_k: float) -> np.ndarray:
        h = 6.62607015e-34
        c = 299_792_458.0
        k = 1.380649e-23
        sigma_m = np.asarray(wavenumber_cm, dtype=float) * 100.0
        exponent = np.clip(h * c * sigma_m / (k * max(temperature_k, 1.0)), 0.0, 700.0)
        return 2.0 * h * c**2 * sigma_m**3 / np.expm1(exponent) * 100.0

    def _planck_wavenumber_batch(self, wavenumber_cm: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
        h = 6.62607015e-34
        c = 299_792_458.0
        k = 1.380649e-23
        sigma_m = np.asarray(wavenumber_cm, dtype=float).reshape(1, -1) * 100.0
        temperature = np.maximum(np.asarray(temperature_k, dtype=float).reshape(-1, 1), 1.0)
        exponent = np.clip(h * c * sigma_m / (k * temperature), 0.0, 700.0)
        return 2.0 * h * c**2 * sigma_m**3 / np.expm1(exponent) * 100.0

    def _solar_spectral_irradiance(self, wavenumber_cm: np.ndarray) -> np.ndarray:
        """插值TSIS-1 HSRS大气层外太阳光谱，单位W/(m²·cm⁻¹)。"""
        reference_wavenumber, reference_irradiance = self._load_solar_reference_spectrum()
        wavenumber = np.asarray(wavenumber_cm, dtype=float)
        if np.any(~np.isfinite(wavenumber)) or np.any(wavenumber <= 0.0):
            raise ValueError("太阳光谱插值波数必须为有限正值。")
        tolerance = 1e-9
        if (
            float(np.min(wavenumber)) < float(reference_wavenumber[0]) - tolerance
            or float(np.max(wavenumber)) > float(reference_wavenumber[-1]) + tolerance
        ):
            raise RuntimeError(
                "TSIS-1太阳参考光谱未覆盖请求波段："
                f"{float(np.min(wavenumber)):.3f}–{float(np.max(wavenumber)):.3f} cm⁻¹。"
            )
        return np.interp(wavenumber, reference_wavenumber, reference_irradiance)

    def solar_spectral_irradiance(self, wavenumber_cm: np.ndarray) -> np.ndarray:
        """返回1 AU处TSIS-1大气层外太阳波数谱，单位W/(m²·cm⁻¹)。"""
        return self._solar_spectral_irradiance(wavenumber_cm)

    @classmethod
    def _load_solar_reference_spectrum(cls) -> tuple[np.ndarray, np.ndarray]:
        cached_wavenumber = cls._solar_reference_wavenumber_cm
        cached_irradiance = cls._solar_reference_irradiance_per_cm
        if cached_wavenumber is not None and cached_irradiance is not None:
            return cached_wavenumber, cached_irradiance
        if not SOLAR_REFERENCE_PATH.is_file():
            raise RuntimeError(f"缺少TSIS-1太阳参考光谱资源：{SOLAR_REFERENCE_PATH}")
        try:
            from netCDF4 import Dataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError("读取TSIS-1太阳参考光谱需要安装netCDF4。") from exc
        try:
            with Dataset(SOLAR_REFERENCE_PATH, "r") as dataset:
                wavelength_nm = np.asarray(dataset.variables["Vacuum Wavelength"][:], dtype=float)
                irradiance_per_nm = np.asarray(dataset.variables["SSI"][:], dtype=float)
        except (KeyError, OSError) as exc:
            raise RuntimeError(f"TSIS-1太阳参考光谱读取失败：{exc}") from exc
        valid = (
            np.isfinite(wavelength_nm)
            & np.isfinite(irradiance_per_nm)
            & (wavelength_nm > 0.0)
            & (irradiance_per_nm >= 0.0)
        )
        if np.count_nonzero(valid) < 2:
            raise RuntimeError("TSIS-1太阳参考光谱不包含足够的有效数据。")
        wavelength_nm = wavelength_nm[valid]
        irradiance_per_nm = irradiance_per_nm[valid]
        # λ[nm] = 1e7 / ν~[cm⁻¹]，因此
        # E_ν~ = E_λ * |dλ/dν~| = E_λ * 1e7 / ν~²。
        wavenumber_cm = 1.0e7 / wavelength_nm
        irradiance_per_cm = irradiance_per_nm * 1.0e7 / wavenumber_cm**2
        order = np.argsort(wavenumber_cm)
        wavenumber_cm = np.asarray(wavenumber_cm[order], dtype=float)
        irradiance_per_cm = np.asarray(irradiance_per_cm[order], dtype=float)
        if (
            wavenumber_cm[0] > SPECTRAL_WAVENUMBER_MIN_CM
            or wavenumber_cm[-1] < SPECTRAL_WAVENUMBER_MAX_CM
        ):
            raise RuntimeError("TSIS-1太阳参考光谱未完整覆盖500–33300 cm⁻¹。")
        wavenumber_cm.setflags(write=False)
        irradiance_per_cm.setflags(write=False)
        cls._solar_reference_wavenumber_cm = wavenumber_cm
        cls._solar_reference_irradiance_per_cm = irradiance_per_cm
        return wavenumber_cm, irradiance_per_cm

    def broadband_radiance(
        self,
        surface_temperature_k: np.ndarray,
        surface_albedo: np.ndarray,
        surface_emissivity: np.ndarray,
        cloud_fraction: np.ndarray,
        cloud_temperature_k: np.ndarray,
        cloud_height_m: np.ndarray,
        mu0: np.ndarray,
        mu_view: np.ndarray,
        visibility_km: float,
        enable_scattering: bool = True,
        cloud_effective_radius_um: np.ndarray | None = None,
        cloud_liquid_water_path_g_m2: np.ndarray | None = None,
        aerosol_optical_depth_550: np.ndarray | float | None = None,
        aerosol_type: np.ndarray | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        mu = np.clip(mu_view, 0.03, 1.0)
        solar_mu = np.clip(mu0, 0.0, 1.0)
        visibility = max(float(visibility_km), 1.0)
        estimated_aerosol_tau = 3.912 / visibility
        if aerosol_optical_depth_550 is None:
            aerosol_tau = estimated_aerosol_tau
        else:
            requested_aod = np.asarray(aerosol_optical_depth_550, dtype=float)
            aerosol_tau = np.where(
                np.isfinite(requested_aod) & (requested_aod >= 0.0),
                requested_aod,
                estimated_aerosol_tau,
            )
        if aerosol_type is None:
            aerosol_types = np.full(
                np.asarray(surface_temperature_k).shape, "continental_average", dtype="<U24"
            )
        else:
            aerosol_types = np.broadcast_to(
                np.asarray(aerosol_type, dtype=str), np.asarray(surface_temperature_k).shape
            )
        aerosol_ssa = np.full(aerosol_types.shape, self.OPAC_TYPE_PROPERTIES["continental_average"][0])
        aerosol_g = np.full(aerosol_types.shape, self.OPAC_TYPE_PROPERTIES["continental_average"][1])
        for type_name, properties in self.OPAC_TYPE_PROPERTIES.items():
            selected = aerosol_types == type_name
            aerosol_ssa[selected] = properties[0]
            aerosol_g[selected] = properties[1]
        rayleigh_tau = 0.10
        gas_tau_thermal = 0.22
        estimated_re, estimated_lwp = ModisDataManager.estimate_cloud_microphysics(
            cloud_temperature_k, cloud_height_m
        )
        cloud_re = (
            estimated_re if cloud_effective_radius_um is None
            else np.where(
                np.isfinite(cloud_effective_radius_um) & (cloud_effective_radius_um > 0.0),
                cloud_effective_radius_um,
                estimated_re,
            )
        )
        cloud_lwp = (
            estimated_lwp if cloud_liquid_water_path_g_m2 is None
            else np.where(
                np.isfinite(cloud_liquid_water_path_g_m2) & (cloud_liquid_water_path_g_m2 > 0.0),
                cloud_liquid_water_path_g_m2,
                estimated_lwp,
            )
        )
        # 几何光学极限Qext≈2时，calcCloudMie化简为
        # tau≈3*LWP/(2*rho_water*r_eff)=1.5*LWP[g/m²]/r_eff[μm]。
        cloud_tau = np.where(
            cloud_height_m > 0.0,
            1.5 * np.clip(cloud_lwp, 1.0, 3000.0) / np.clip(cloud_re, 2.0, 60.0),
            0.0,
        )
        thermal_aerosol_factor = 0.08 + 0.50 * (1.0 - aerosol_ssa)
        clear_trans_thermal = np.exp(
            -(gas_tau_thermal + thermal_aerosol_factor * aerosol_tau) / mu
        )
        cloud_trans_thermal = np.exp(-(gas_tau_thermal + cloud_tau) / mu)
        surface_radiance = surface_emissivity * SIGMA * surface_temperature_k**4 / np.pi
        air_radiance = (1.0 - clear_trans_thermal) * SIGMA * 255.0**4 / np.pi
        clear_thermal = surface_radiance * clear_trans_thermal + air_radiance
        fallback_cloud_temperature = np.clip(
            surface_temperature_k - 6.5 * np.maximum(cloud_height_m, 0.0) / 1000.0,
            180.0,
            330.0,
        )
        model_cloud_temperature = np.where(
            np.isfinite(cloud_temperature_k), cloud_temperature_k, fallback_cloud_temperature
        )
        cloudy_thermal = SIGMA * model_cloud_temperature**4 / np.pi * (1.0 - cloud_trans_thermal) + surface_radiance * cloud_trans_thermal
        thermal = (1.0 - cloud_fraction) * clear_thermal + cloud_fraction * cloudy_thermal

        tau_solar = aerosol_tau + rayleigh_tau
        down = np.exp(-tau_solar / np.clip(solar_mu, 0.03, 1.0))
        up = np.exp(-tau_solar / mu)
        ground = SOLAR_CONSTANT * solar_mu * surface_albedo * down * up / np.pi
        if enable_scattering:
            type_scattering_factor = 0.08 * aerosol_ssa / 0.925 * (
                1.0 + 0.3 * (aerosol_g - 0.703)
            )
            single_scatter = SOLAR_CONSTANT * solar_mu * (1.0 - np.exp(-tau_solar * (1.0 / np.clip(solar_mu, 0.03, 1.0) + 1.0 / mu))) * type_scattering_factor / np.pi
            multiple_scatter = ground * np.clip(
                0.18 * aerosol_ssa / 0.925 * tau_solar / (1.0 + tau_solar), 0.0, 0.25
            )
        else:
            single_scatter = 0.0
            multiple_scatter = 0.0
        cloud_reflection = SOLAR_CONSTANT * solar_mu * 0.55 * (1.0 - np.exp(-cloud_tau / np.clip(solar_mu, 0.03, 1.0))) / np.pi
        reflected = (1.0 - cloud_fraction) * (ground + single_scatter + multiple_scatter) + cloud_fraction * cloud_reflection
        reflected = np.where(mu0 > 0.0, reflected, 0.0)
        return np.maximum(thermal, 0.0), np.maximum(reflected, 0.0)


class EarthIrradianceManager:
    def __init__(
        self,
        modis_manager: ModisDataManager | None = None,
        merra2_aerosol_manager: Merra2AerosolManager | None = None,
    ) -> None:
        self.modis_manager = modis_manager or ModisDataManager()
        self.merra2_aerosol_manager = merra2_aerosol_manager or Merra2AerosolManager()
        self.atmosphere = LayeredAtmosphereSolver()
        self._results: dict[str, np.ndarray] = {}
        self._parameters: dict[str, Any] = {}
        self._info: dict[str, Any] = {}
        self._summary: pd.DataFrame | None = None

    def get_results(self) -> dict[str, np.ndarray]:
        return dict(self._results)

    def get_parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    def get_info(self) -> dict[str, Any]:
        return dict(self._info)

    def get_summary(self) -> pd.DataFrame | None:
        return None if self._summary is None else self._summary.copy()

    def compute(
        self,
        trajectory: pd.DataFrame,
        parameters: dict[str, Any],
        progress_callback: Callable[[int, int], bool] | None = None,
    ) -> dict[str, np.ndarray]:
        grid = self.modis_manager.get_grid()
        if grid is None:
            raise RuntimeError("请先加载 MODIS 地球环境数据或环境缓存。")
        required = {"time", "lat", "lon", "alt", "right_ascension", "declination"}
        if not required.issubset(trajectory.columns):
            raise ValueError(f"轨迹缺少字段：{', '.join(sorted(required - set(trajectory.columns)))}")
        params = self._validate_parameters(parameters)
        self.atmosphere.set_upper_atmosphere_temperature_offset(
            params["upper_atmosphere_temperature_offset_k"]
        )
        absorption_file = params.get("absorption_file", "")
        if absorption_file and self.atmosphere._absorption_tau is None:
            self.atmosphere.load_absorption_optical_depth(absorption_file)
        self._configure_optical_depth_correction(params)
        thermal: list[float] = []
        reflected: list[float] = []
        visible_counts: list[int] = []
        cloud_means: list[float] = []
        aerosol_aod_550: list[float] = []
        directions: list[np.ndarray] = []
        total = len(trajectory)
        for step, row in enumerate(trajectory.itertuples(index=False), start=1):
            sample = row._asdict()
            result = self.compute_at_position(grid, sample, params)
            thermal.append(result["earth_thermal_irradiance"])
            reflected.append(result["earth_reflected_irradiance"])
            visible_counts.append(result["visible_cell_count"])
            cloud_means.append(result["visible_cloud_fraction"])
            aerosol_aod_550.append(result["aerosol_optical_depth_550"])
            directions.append(result["effective_earth_direction_ecef"])
            if progress_callback is not None and progress_callback(step, total) is False:
                raise RuntimeError("大气辐射计算已取消。")
        arrays = {
            "time": trajectory["time"].to_numpy(dtype=float),
            "earth_thermal_irradiance": np.asarray(thermal),
            "earth_reflected_irradiance": np.asarray(reflected),
            "earth_total_irradiance": np.asarray(thermal) + np.asarray(reflected),
            "visible_cell_count": np.asarray(visible_counts, dtype=int),
            "visible_cloud_fraction": np.asarray(cloud_means),
            "aerosol_optical_depth_550": np.asarray(aerosol_aod_550),
            "effective_earth_direction_ecef": np.vstack(directions),
        }
        self._results = arrays
        self._parameters = params
        self._summary = pd.DataFrame({key: value for key, value in arrays.items() if np.asarray(value).ndim == 1})
        self._info = {
            "status": "computed", "number_of_time_steps": total,
            "thermal_mean": float(np.mean(arrays["earth_thermal_irradiance"])),
            "reflected_mean": float(np.mean(arrays["earth_reflected_irradiance"])),
            "total_mean": float(np.mean(arrays["earth_total_irradiance"])),
            "grid_resolution_deg": float(grid.metadata.get("resolution_deg", 0.0)),
            "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": (
                "35层宽带大气 + MODIS地球盘积分 + 下垫面OPAC + MERRA-2 AOD"
                if params["use_merra2_aerosol"]
                else "35层宽带大气 + MODIS地球盘积分 + 下垫面OPAC"
            ),
        }
        return dict(arrays)

    def compute_at_position(self, grid: EarthEnvironmentGrid, sample: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        lat = np.deg2rad(float(sample["lat"]))
        lon = np.deg2rad(float(sample["lon"]))
        altitude_m = self._altitude_m(float(sample["alt"]), parameters)
        target = (EARTH_RADIUS_M + altitude_m) * np.asarray([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        grid_lat = np.deg2rad(grid.latitude)
        grid_lon = np.deg2rad(grid.longitude)
        normals = np.stack((np.cos(grid_lat) * np.cos(grid_lon), np.cos(grid_lat) * np.sin(grid_lon), np.sin(grid_lat)), axis=-1)
        cells = EARTH_RADIUS_M * normals
        cell_to_target = target - cells
        distance = np.linalg.norm(cell_to_target, axis=-1)
        view_direction = cell_to_target / np.maximum(distance[..., None], 1.0)
        mu_view = np.sum(normals * view_direction, axis=-1)
        visible = (mu_view > 0.0) & grid.valid_mask

        # 轨迹只提供赤经/赤纬而没有UTC恒星时。此处把赤经作为地固近似经度；
        # 若后续轨迹提供UTC，应在轨迹层先转换为太阳地固方向。
        sun_lon = np.deg2rad(float(sample.get("right_ascension", 0.0)))
        sun_lat = np.deg2rad(float(sample.get("declination", 0.0)))
        sun = np.asarray([np.cos(sun_lat) * np.cos(sun_lon), np.cos(sun_lat) * np.sin(sun_lon), np.sin(sun_lat)])
        mu0 = np.sum(normals * sun, axis=-1)

        lat_step = np.deg2rad(180.0 / grid.latitude.shape[0])
        lon_step = np.deg2rad(360.0 / grid.latitude.shape[1])
        area = EARTH_RADIUS_M**2 * lon_step * (np.sin(grid_lat + 0.5 * lat_step) - np.sin(grid_lat - 0.5 * lat_step))
        solid_angle = np.maximum(area, 0.0) * np.maximum(mu_view, 0.0) / np.maximum(distance**2, 1.0)
        cloud_fraction = grid.cloud_fraction if parameters["enable_cloud"] else np.zeros_like(grid.cloud_fraction)
        cloud_re, cloud_lwp = self.modis_manager._grid_cloud_microphysics(grid)
        opac_aerosol_type = self.modis_manager.opac_aerosol_type_grid(
            grid.surface_type, grid.latitude
        )
        aerosol_aod, aerosol_info = self._aerosol_for_grid(grid, sample, parameters)
        rad_thermal, rad_reflected = self.atmosphere.broadband_radiance(
            grid.surface_temperature_k, grid.surface_albedo, grid.surface_emissivity,
            cloud_fraction, grid.cloud_top_temperature_k, grid.cloud_top_height_m,
            mu0, mu_view, parameters["visibility_km"], parameters["enable_scattering"],
            cloud_re, cloud_lwp, aerosol_aod, opac_aerosol_type,
        )
        weights = np.where(visible, solid_angle, 0.0)
        thermal = float(np.sum(rad_thermal * weights))
        reflected = float(np.sum(rad_reflected * weights))
        weighted_direction = np.sum((-view_direction) * weights[..., None], axis=(0, 1))
        direction_norm = np.linalg.norm(weighted_direction)
        direction = weighted_direction / direction_norm if direction_norm > 0.0 else -target / np.linalg.norm(target)
        weight_sum = float(np.sum(weights))
        cloud_mean = float(np.sum(grid.cloud_fraction * weights) / weight_sum) if weight_sum > 0.0 else 0.0
        return {
            "earth_thermal_irradiance": thermal,
            "earth_reflected_irradiance": reflected,
            "earth_total_irradiance": thermal + reflected,
            "visible_cell_count": int(np.count_nonzero(visible)),
            "visible_cloud_fraction": cloud_mean,
            "effective_earth_direction_ecef": direction,
            **aerosol_info,
        }

    def compute_spectrum_at_position(
        self,
        grid: EarthEnvironmentGrid,
        sample: dict[str, Any],
        parameters: dict[str, Any],
        progress_callback: Callable[[int, int], bool] | None = None,
        spectral_mode: str = "fast",
        maximum_spectral_points: int = 600,
        maximum_spatial_samples: int = 768,
        batch_size: int = 64,
        high_resolution_step_cm: float = 0.1,
        high_resolution_chunk_size: int = 2000,
        wavelength_grid_um: np.ndarray | None = None,
        wavenumber_grid_cm: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """计算目标位置处经大气传输后的地球热辐射与反射太阳光谱。

        各可见地球网格的光谱辐亮度按其对目标张开的立体角积分，输出光谱
        辐照度。计算采用分批矢量化，进度以完成的可见网格批次数更新。
        """
        params = self._validate_parameters(parameters)
        self.atmosphere.set_upper_atmosphere_temperature_offset(
            params["upper_atmosphere_temperature_offset_k"]
        )
        mode_text = str(spectral_mode).strip().lower()
        high_resolution = mode_text in {
            "high_resolution", "high-resolution", "高分辨率目标单柱", "高分辨率",
        }
        absorption_file = params.get("absorption_file", "")
        if absorption_file and self.atmosphere._absorption_tau is None:
            self.atmosphere.load_absorption_optical_depth(absorption_file)
        self._configure_optical_depth_correction(params)

        lat = np.deg2rad(float(sample["lat"]))
        lon = np.deg2rad(float(sample["lon"]))
        altitude_m = self._altitude_m(float(sample["alt"]), params)
        target = (EARTH_RADIUS_M + altitude_m) * np.asarray([
            np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat),
        ])
        grid_lat = np.deg2rad(grid.latitude)
        grid_lon = np.deg2rad(grid.longitude)
        normals = np.stack((
            np.cos(grid_lat) * np.cos(grid_lon),
            np.cos(grid_lat) * np.sin(grid_lon),
            np.sin(grid_lat),
        ), axis=-1)
        cells = EARTH_RADIUS_M * normals
        cell_to_target = target - cells
        distance = np.linalg.norm(cell_to_target, axis=-1)
        view_direction = cell_to_target / np.maximum(distance[..., None], 1.0)
        mu_view = np.sum(normals * view_direction, axis=-1)
        visible = (mu_view > 0.0) & np.asarray(grid.valid_mask, dtype=bool)

        sun_lon = np.deg2rad(float(sample.get("right_ascension", 0.0)))
        sun_lat = np.deg2rad(float(sample.get("declination", 0.0)))
        sun = np.asarray([
            np.cos(sun_lat) * np.cos(sun_lon),
            np.cos(sun_lat) * np.sin(sun_lon),
            np.sin(sun_lat),
        ])
        mu0 = np.sum(normals * sun, axis=-1)
        lat_step = np.deg2rad(180.0 / grid.latitude.shape[0])
        lon_step = np.deg2rad(360.0 / grid.latitude.shape[1])
        area = EARTH_RADIUS_M**2 * lon_step * (
            np.sin(grid_lat + 0.5 * lat_step) - np.sin(grid_lat - 0.5 * lat_step)
        )
        solid_angle = (
            np.maximum(area, 0.0)
            * np.maximum(mu_view, 0.0)
            / np.maximum(distance**2, 1.0)
        )
        selected = np.flatnonzero(visible.reshape(-1) & (solid_angle.reshape(-1) > 0.0))
        if selected.size == 0:
            raise RuntimeError("目标当前位置没有可积分的有效地球网格。")
        visible_cell_count = int(selected.size)
        weights = solid_angle.reshape(-1)[selected]
        requested_wavelength = None
        requested_wavenumber_grid = None
        if wavelength_grid_um is not None and wavenumber_grid_cm is not None:
            raise ValueError("自定义波长网格和波数网格不能同时指定。")
        if high_resolution:
            if self.atmosphere._absorption_wavenumber is None or self.atmosphere._absorption_tau is None:
                raise RuntimeError("高分辨率目标单柱预览需要先导入总气体分子光学厚度文件。")
            representative_local = int(np.argmax(weights))
            selected = selected[representative_local:representative_local + 1]
            weights = np.ones(1, dtype=float)
        else:
            sample_limit = max(1, int(maximum_spatial_samples))
            if selected.size > sample_limit:
                total_weight = float(np.sum(weights))
                cumulative = np.cumsum(weights) / max(total_weight, 1e-30)
                quantiles = (np.arange(sample_limit, dtype=float) + 0.5) / sample_limit
                sampled_local = np.searchsorted(cumulative, quantiles, side="left")
                sampled_local = np.clip(sampled_local, 0, selected.size - 1)
                unique_local, counts = np.unique(sampled_local, return_counts=True)
                selected = selected[unique_local]
                weights = counts.astype(float) * total_weight / sample_limit

        reference_minimum = SPECTRAL_WAVENUMBER_MIN_CM
        reference_maximum = SPECTRAL_WAVENUMBER_MAX_CM
        if self.atmosphere._absorption_wavenumber is not None:
            source_wavenumber = np.unique(np.asarray(self.atmosphere._absorption_wavenumber, dtype=float))
            source_wavenumber = source_wavenumber[np.isfinite(source_wavenumber) & (source_wavenumber > 0.0)]
            if source_wavenumber.size < 2:
                raise RuntimeError("总光学厚度文件中的有效波数不足。")
            lower_wavenumber = max(reference_minimum, float(source_wavenumber[0]))
            upper_wavenumber = min(reference_maximum, float(source_wavenumber[-1]))
            if lower_wavenumber >= upper_wavenumber:
                raise RuntimeError("总光学厚度文件不包含500–33300 cm⁻¹参考波段。")
        else:
            lower_wavenumber = reference_minimum
            upper_wavenumber = reference_maximum
        if wavenumber_grid_cm is not None:
            requested_wavenumber_grid = np.unique(
                np.asarray(wavenumber_grid_cm, dtype=float).reshape(-1)
            )
            requested_wavenumber_grid = requested_wavenumber_grid[
                np.isfinite(requested_wavenumber_grid) & (requested_wavenumber_grid > 0.0)
            ]
            if requested_wavenumber_grid.size < 2:
                raise ValueError("自定义波数网格至少需要两个有限正值。")
            if (
                requested_wavenumber_grid[0] < lower_wavenumber - 1.0e-8
                or requested_wavenumber_grid[-1] > upper_wavenumber + 1.0e-8
            ):
                raise RuntimeError(
                    "自定义波数网格超出太阳参考谱或总光学厚度文件的有效覆盖范围。"
                )
            wavenumber = requested_wavenumber_grid
        elif wavelength_grid_um is not None:
            requested_wavelength = np.unique(np.asarray(wavelength_grid_um, dtype=float).reshape(-1))
            requested_wavelength = requested_wavelength[
                np.isfinite(requested_wavelength) & (requested_wavelength > 0.0)
            ]
            if requested_wavelength.size < 2:
                raise ValueError("自定义波长网格至少需要两个有限正值。")
            requested_wavenumber = np.sort(10_000.0 / requested_wavelength)
            if (
                requested_wavenumber[0] < lower_wavenumber - 1.0e-8
                or requested_wavenumber[-1] > upper_wavenumber + 1.0e-8
            ):
                raise RuntimeError(
                    "自定义波长网格超出太阳参考谱或总光学厚度文件的有效覆盖范围。"
                )
            wavenumber = requested_wavenumber
        elif high_resolution:
            if lower_wavenumber > reference_minimum or upper_wavenumber < reference_maximum:
                raise RuntimeError("高分辨率求解要求总光学厚度完整覆盖500–33300 cm⁻¹。")
            spectral_step = max(float(high_resolution_step_cm), 1e-6)
            wavenumber = np.arange(
                reference_minimum,
                reference_maximum + 0.5 * spectral_step,
                spectral_step,
            )
            wavenumber[-1] = reference_maximum
        else:
            maximum_points = max(100, int(maximum_spectral_points))
            wavenumber = np.linspace(lower_wavenumber, upper_wavenumber, maximum_points)

        cloud_fraction = (
            np.asarray(grid.cloud_fraction, dtype=float)
            if params["enable_cloud"]
            else np.zeros_like(grid.cloud_fraction, dtype=float)
        )
        cloud_re, cloud_lwp = self.modis_manager._grid_cloud_microphysics(grid)
        opac_aerosol_type = self.modis_manager.opac_aerosol_type_grid(
            grid.surface_type, grid.latitude
        )
        aerosol_aod, aerosol_info = self._aerosol_for_grid(grid, sample, params)
        flat_fields = {
            "surface_temperature_k": np.asarray(grid.surface_temperature_k, dtype=float).reshape(-1),
            "surface_albedo": np.asarray(grid.surface_albedo, dtype=float).reshape(-1),
            "surface_emissivity": np.asarray(grid.surface_emissivity, dtype=float).reshape(-1),
            "cloud_fraction": cloud_fraction.reshape(-1),
            "cloud_temperature_k": np.asarray(grid.cloud_top_temperature_k, dtype=float).reshape(-1),
            "cloud_height_m": np.asarray(grid.cloud_top_height_m, dtype=float).reshape(-1),
            "cloud_effective_radius_um": cloud_re.reshape(-1),
            "cloud_liquid_water_path_g_m2": cloud_lwp.reshape(-1),
            "solar_mu": mu0.reshape(-1),
            "view_mu": mu_view.reshape(-1),
            "scattering_cosine": np.sum(view_direction * sun, axis=-1).reshape(-1),
            "aerosol_type": opac_aerosol_type.reshape(-1),
        }
        if aerosol_aod is not None:
            flat_fields["aerosol_optical_depth_550"] = np.asarray(aerosol_aod, dtype=float).reshape(-1)
        high_resolution_fields = {
            name: values[selected] for name, values in flat_fields.items()
        }
        thermal_wavenumber = np.zeros(wavenumber.shape, dtype=float)
        reflected_wavenumber = np.zeros(wavenumber.shape, dtype=float)
        single_scattering_wavenumber = np.zeros(wavenumber.shape, dtype=float)
        multiple_scattering_wavenumber = np.zeros(wavenumber.shape, dtype=float)
        surface_reflection_wavenumber = np.zeros(wavenumber.shape, dtype=float)
        custom_spectral_grid = (
            requested_wavelength is not None or requested_wavenumber_grid is not None
        )
        if high_resolution:
            chunk_size = max(100, int(high_resolution_chunk_size))
            spectral_slices: list[slice] = []
            start = 0
            while start < wavenumber.size:
                stop = min(start + chunk_size, wavenumber.size)
                # 光谱核至少需要两个波数点。若整除后只剩一个点，
                # 将该点并入当前（倒数第一）分块。
                if wavenumber.size - stop == 1:
                    stop = wavenumber.size
                spectral_slices.append(slice(start, stop))
                start = stop
            total_batches = len(spectral_slices)
            if progress_callback is not None and progress_callback(0, total_batches) is False:
                raise RuntimeError("大气辐射光谱预览已取消。")
            for batch_index, spectral_slice in enumerate(spectral_slices, start=1):
                result = self.atmosphere.spectral_radiance_batch(
                    wavenumber[spectral_slice],
                    **high_resolution_fields,
                    visibility_km=params["visibility_km"],
                    enable_scattering=params["enable_scattering"],
                    return_solar_components=True,
                )
                for destination, values in zip(
                    (
                        thermal_wavenumber,
                        reflected_wavenumber,
                        single_scattering_wavenumber,
                        multiple_scattering_wavenumber,
                        surface_reflection_wavenumber,
                    ),
                    result,
                ):
                    destination[spectral_slice] = values[0]
                if progress_callback is not None and progress_callback(batch_index, total_batches) is False:
                    raise RuntimeError("大气辐射光谱预览已取消。")
        elif custom_spectral_grid:
            size = max(1, int(batch_size))
            spectral_chunk_size = max(2, int(high_resolution_chunk_size))
            spectral_slices: list[slice] = []
            spectral_start = 0
            while spectral_start < wavenumber.size:
                spectral_stop = min(spectral_start + spectral_chunk_size, wavenumber.size)
                if wavenumber.size - spectral_stop == 1:
                    spectral_stop = wavenumber.size
                spectral_slices.append(slice(spectral_start, spectral_stop))
                spectral_start = spectral_stop
            spatial_starts = list(range(0, selected.size, size))
            total_batches = len(spatial_starts) * len(spectral_slices)
            completed_batches = 0
            if progress_callback is not None and progress_callback(0, total_batches) is False:
                raise RuntimeError("大气辐射光谱预览已取消。")
            destinations = (
                thermal_wavenumber,
                reflected_wavenumber,
                single_scattering_wavenumber,
                multiple_scattering_wavenumber,
                surface_reflection_wavenumber,
            )
            for spectral_slice in spectral_slices:
                for start in spatial_starts:
                    batch = selected[start:start + size]
                    result = self.atmosphere.spectral_radiance_batch(
                        wavenumber[spectral_slice],
                        **{name: values[batch] for name, values in flat_fields.items()},
                        visibility_km=params["visibility_km"],
                        enable_scattering=params["enable_scattering"],
                        return_solar_components=True,
                    )
                    batch_weights = weights[start:start + size, None]
                    for destination, values in zip(destinations, result):
                        destination[spectral_slice] += np.sum(values * batch_weights, axis=0)
                    completed_batches += 1
                    if (
                        progress_callback is not None
                        and progress_callback(completed_batches, total_batches) is False
                    ):
                        raise RuntimeError("大气辐射光谱预览已取消。")
        else:
            size = max(1, int(batch_size))
            total_batches = int(np.ceil(selected.size / size))
            if progress_callback is not None and progress_callback(0, total_batches) is False:
                raise RuntimeError("大气辐射光谱预览已取消。")
            for batch_index, start in enumerate(range(0, selected.size, size), start=1):
                batch = selected[start:start + size]
                (
                    thermal_radiance,
                    reflected_radiance,
                    single_scattering_radiance,
                    multiple_scattering_radiance,
                    surface_reflection_radiance,
                ) = self.atmosphere.spectral_radiance_batch(
                    wavenumber,
                    **{name: values[batch] for name, values in flat_fields.items()},
                    visibility_km=params["visibility_km"],
                    enable_scattering=params["enable_scattering"],
                    return_solar_components=True,
                )
                batch_weights = weights[start:start + size, None]
                thermal_wavenumber += np.sum(thermal_radiance * batch_weights, axis=0)
                reflected_wavenumber += np.sum(reflected_radiance * batch_weights, axis=0)
                single_scattering_wavenumber += np.sum(single_scattering_radiance * batch_weights, axis=0)
                multiple_scattering_wavenumber += np.sum(multiple_scattering_radiance * batch_weights, axis=0)
                surface_reflection_wavenumber += np.sum(surface_reflection_radiance * batch_weights, axis=0)
                if progress_callback is not None and progress_callback(batch_index, total_batches) is False:
                    raise RuntimeError("大气辐射光谱预览已取消。")

        thermal_integral = float(np.trapezoid(thermal_wavenumber, wavenumber))
        reflected_integral = float(np.trapezoid(reflected_wavenumber, wavenumber))
        single_scattering_integral = float(np.trapezoid(single_scattering_wavenumber, wavenumber))
        multiple_scattering_integral = float(np.trapezoid(multiple_scattering_wavenumber, wavenumber))
        surface_reflection_integral = float(np.trapezoid(surface_reflection_wavenumber, wavenumber))
        wavelength_descending = 10_000.0 / wavenumber
        wavelength_um = wavelength_descending[::-1]
        jacobian = 10_000.0 / np.maximum(wavelength_um**2, 1e-30)
        thermal_wavelength = thermal_wavenumber[::-1] * jacobian
        reflected_wavelength = reflected_wavenumber[::-1] * jacobian
        single_scattering_wavelength = single_scattering_wavenumber[::-1] * jacobian
        multiple_scattering_wavelength = multiple_scattering_wavenumber[::-1] * jacobian
        surface_reflection_wavelength = surface_reflection_wavenumber[::-1] * jacobian
        summary_weight = weights / max(float(np.sum(weights)), 1e-30)
        representative_cloud_fraction = float(
            np.sum(np.asarray(high_resolution_fields["cloud_fraction"], dtype=float) * summary_weight)
        )
        representative_cloud_re = float(
            np.sum(np.asarray(high_resolution_fields["cloud_effective_radius_um"], dtype=float) * summary_weight)
        )
        representative_cloud_lwp = float(
            np.sum(np.asarray(high_resolution_fields["cloud_liquid_water_path_g_m2"], dtype=float) * summary_weight)
        )
        subpoint_environment = self.modis_manager.sample_environment_at_subpoint(
            grid,
            float(sample["lat"]),
            float(sample["lon"]),
        )
        return {
            "wavenumber_cm": wavenumber,
            "wavelength_um": wavelength_um,
            "earth_thermal_spectral_wavenumber": thermal_wavenumber,
            "earth_reflected_spectral_wavenumber": reflected_wavenumber,
            "earth_total_spectral_wavenumber": thermal_wavenumber + reflected_wavenumber,
            "solar_single_scattering_spectral_wavenumber": single_scattering_wavenumber,
            "solar_multiple_scattering_spectral_wavenumber": multiple_scattering_wavenumber,
            "solar_surface_reflection_spectral_wavenumber": surface_reflection_wavenumber,
            "earth_thermal_spectral_irradiance": thermal_wavelength,
            "earth_reflected_spectral_irradiance": reflected_wavelength,
            "solar_single_scattering_spectral_irradiance": single_scattering_wavelength,
            "solar_multiple_scattering_spectral_irradiance": multiple_scattering_wavelength,
            "solar_surface_reflection_spectral_irradiance": surface_reflection_wavelength,
            "earth_total_spectral_irradiance": thermal_wavelength + reflected_wavelength,
            "earth_thermal_irradiance": thermal_integral,
            "earth_reflected_irradiance": reflected_integral,
            "solar_single_scattering_irradiance": single_scattering_integral,
            "solar_multiple_scattering_irradiance": multiple_scattering_integral,
            "solar_surface_reflection_irradiance": surface_reflection_integral,
            "earth_total_irradiance": thermal_integral + reflected_integral,
            "visible_cell_count": visible_cell_count,
            "evaluated_cell_count": int(selected.size),
            "spectral_point_count": int(wavenumber.size),
            "target_time": float(sample.get("time", 0.0)),
            "target_latitude_deg": float(sample["lat"]),
            "target_longitude_deg": float(sample["lon"]),
            "target_altitude_m": altitude_m,
            "uses_total_optical_depth": self.atmosphere._absorption_tau is not None,
            "optical_depth_correction_enabled": bool(
                params.get("enable_optical_depth_correction", False)
            ),
            "optical_depth_corrections": self.atmosphere.get_optical_depth_corrections(),
            "upper_atmosphere_temperature_offset_k": float(
                self.atmosphere._upper_atmosphere_temperature_offset_k
            ),
            "preview_mode": (
                "high_resolution" if high_resolution
                else "detector_earth_disk" if custom_spectral_grid
                else "fast"
            ),
            "spectral_quantity": "radiance" if high_resolution else "irradiance",
            "representative_latitude_deg": float(np.asarray(grid.latitude).reshape(-1)[selected[0]]),
            "representative_longitude_deg": float(np.asarray(grid.longitude).reshape(-1)[selected[0]]),
            "representative_cloud_fraction": representative_cloud_fraction,
            "representative_cloud_effective_radius_um": representative_cloud_re,
            "representative_cloud_liquid_water_path_g_m2": representative_cloud_lwp,
            **subpoint_environment,
            "visibility_km": float(params["visibility_km"]),
            **aerosol_info,
            "wavenumber_min_cm": float(wavenumber[0]),
            "wavenumber_max_cm": float(wavenumber[-1]),
            "solar_spectrum_source": SOLAR_REFERENCE_SOURCE,
            "solar_spectrum_reference_distance_au": 1.0,
        }

    def export_results(self, output_dir: str | Path) -> dict[str, str]:
        if not self._results or self._summary is None:
            raise RuntimeError("没有可导出的大气辐射结果。")
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        summary_path = target / "summary.csv"
        field_path = target / "irradiance.npz"
        parameters_path = target / "parameters.json"
        manifest_path = target / "manifest.json"
        self._summary.to_csv(summary_path, index=False)
        np.savez_compressed(field_path, **self._results)
        parameters_path.write_text(json.dumps(self._parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps({"schema_version": 1, "model": self._info.get("model"), "arrays": {key: list(value.shape) for key, value in self._results.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"summary_file": str(summary_path), "field_file": str(field_path), "parameters_file": str(parameters_path), "manifest_file": str(manifest_path)}

    def load_results(self, file_path: str | Path) -> dict[str, np.ndarray]:
        source = Path(file_path).expanduser().resolve()
        field_path = source if source.suffix.lower() == ".npz" else source.parent / "irradiance.npz"
        with np.load(field_path, allow_pickle=False) as data:
            self._results = {key: np.asarray(data[key]) for key in data.files}
        parameter_path = field_path.parent / "parameters.json"
        self._parameters = json.loads(parameter_path.read_text(encoding="utf-8")) if parameter_path.exists() else {}
        self._summary = pd.DataFrame({key: value for key, value in self._results.items() if value.ndim == 1})
        self._info = {
            "status": "loaded", "number_of_time_steps": len(self._results.get("time", [])),
            "thermal_mean": float(np.mean(self._results["earth_thermal_irradiance"])),
            "reflected_mean": float(np.mean(self._results["earth_reflected_irradiance"])),
            "total_mean": float(np.mean(self._results["earth_total_irradiance"])),
            "loaded_from_file": str(field_path), "model": "35层宽带大气 + MODIS地球盘积分",
        }
        return dict(self._results)

    def augment_trajectory(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        if not self._results:
            return trajectory.copy()
        result = trajectory.copy()
        result_time = np.asarray(self._results["time"], dtype=float)
        source_time = result["time"].to_numpy(dtype=float)
        for column in ("earth_thermal_irradiance", "earth_reflected_irradiance"):
            result[column] = np.interp(source_time, result_time, np.asarray(self._results[column], dtype=float))
        return result

    def _validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "visibility_km": max(float(parameters.get("visibility_km", 23.0)), 1.0),
            "enable_cloud": bool(parameters.get("enable_cloud", True)),
            "enable_scattering": bool(parameters.get("enable_scattering", True)),
            "altitude_unit": str(parameters.get("altitude_unit", "km")).lower(),
            "mode": str(parameters.get("mode", "fast")),
            "absorption_file": str(parameters.get("absorption_file", "")),
            "enable_optical_depth_correction": bool(
                parameters.get("enable_optical_depth_correction", False)
            ),
            "optical_depth_corrections": parameters.get("optical_depth_corrections", []),
            "corrected_absorption_file": str(parameters.get("corrected_absorption_file", "")),
            "upper_atmosphere_temperature_offset_k": float(np.clip(
                float(parameters.get("upper_atmosphere_temperature_offset_k", 0.0)),
                -15.0,
                15.0,
            )),
            "use_merra2_aerosol": bool(parameters.get("use_merra2_aerosol", False)),
            "merra2_aerosol_file": str(parameters.get("merra2_aerosol_file", "")),
            "merra2_time_offset_hours": float(parameters.get("merra2_time_offset_hours", 0.0)),
        }

    def _configure_optical_depth_correction(self, parameters: dict[str, Any]) -> None:
        if not bool(parameters.get("enable_optical_depth_correction", False)):
            self.atmosphere.clear_optical_depth_corrections()
            return
        corrections = parameters.get("optical_depth_corrections", [])
        if not corrections:
            raise ValueError("已启用总光学厚度修正，但尚未配置修正波段。")
        self.atmosphere.apply_optical_depth_corrections(corrections)

    def _aerosol_for_grid(
        self,
        grid: EarthEnvironmentGrid,
        sample: dict[str, Any],
        parameters: dict[str, Any],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        estimated_aod = 3.912 / float(parameters["visibility_km"])
        fallback = {
            "aerosol_optical_depth_source": "能见度估算（3.912/V）",
            "aerosol_optical_depth_550": estimated_aod,
            "aerosol_angstrom_exponent": np.nan,
            "merra2_grid_latitude_deg": np.nan,
            "merra2_grid_longitude_deg": np.nan,
            "merra2_time": "-",
        }
        if not parameters.get("use_merra2_aerosol", False):
            return None, fallback
        source = str(parameters.get("merra2_aerosol_file", "")).strip()
        if not source:
            raise RuntimeError("已启用MERRA-2 AOD，但尚未选择M2T1NXAER产品文件。")
        self.merra2_aerosol_manager.ensure_loaded(source)
        elapsed_seconds = (
            float(sample.get("time", 0.0))
            + 3600.0 * float(parameters.get("merra2_time_offset_hours", 0.0))
        )
        aerosol_aod, info = self.merra2_aerosol_manager.resample_for_grid(
            elapsed_seconds,
            grid.latitude,
            grid.longitude,
            float(sample["lat"]),
            float(sample["lon"]),
        )
        aerosol_aod = np.where(
            np.isfinite(aerosol_aod) & (aerosol_aod >= 0.0), aerosol_aod, estimated_aod
        )
        if not np.isfinite(float(info["aerosol_optical_depth_550"])):
            info["aerosol_optical_depth_550"] = estimated_aod
            info["aerosol_optical_depth_source"] += "（星下点无效，采用能见度回退）"
        return aerosol_aod, info

    def _altitude_m(self, altitude: float, parameters: dict[str, Any]) -> float:
        unit = parameters.get("altitude_unit", "km")
        value = altitude * 1000.0 if unit in {"km", "千米"} else altitude
        if value < 0.0:
            raise ValueError("目标轨道高度不能为负值。")
        return value
