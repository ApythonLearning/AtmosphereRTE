from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


# NOAA CrIS full-spectral-resolution SDR channel grids.  These are the same
# calibrated channel centres exposed by the NASA SNDRJ1CrISL1B v3 product.
BAND_GRIDS = {
    "LW": np.arange(648.75, 1096.25 + 0.3125, 0.625),
    "MW": np.arange(1208.75, 1751.25 + 0.3125, 0.625),
    "SW": np.arange(2153.75, 2551.25 + 0.3125, 0.625),
}
IET_EPOCH = datetime(1958, 1, 1, tzinfo=timezone.utc)


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
    cosine = (
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    )
    return 6371.0088 * np.arccos(np.clip(cosine, -1.0, 1.0))


def _valid_footprints(radiance_group: Any, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    valid &= np.asarray(radiance_group["QF1_SCAN_CRISSDR"])[:, None, None] == 0
    valid &= np.all(np.asarray(radiance_group["QF2_CRISSDR"]) == 0, axis=-1)[:, None, :]
    valid &= np.all(np.asarray(radiance_group["QF3_CRISSDR"]) == 0, axis=-1)
    valid &= np.all(np.asarray(radiance_group["QF4_CRISSDR"]) == 0, axis=-1)
    return valid


def _iet_microseconds_to_utc(value: int, tai_minus_utc_seconds: int) -> datetime:
    # JPSS IET is TAI microseconds since 1958-01-01.  TAI-UTC was 37 s in 2025.
    return IET_EPOCH + timedelta(microseconds=int(value), seconds=-tai_minus_utc_seconds)


def _solar_equatorial_position(observation_time: datetime) -> tuple[float, float, float]:
    julian_day = observation_time.timestamp() / 86400.0 + 2440587.5
    centuries = (julian_day - 2451545.0) / 36525.0
    mean_longitude = (
        280.46646 + 36000.76983 * centuries + 0.0003032 * centuries**2
    ) % 360.0
    mean_anomaly = (
        357.52911
        + 35999.05029 * centuries
        - 0.0001537 * centuries**2
        + centuries**3 / 24490000.0
    ) % 360.0
    anomaly_rad = math.radians(mean_anomaly)
    centre = (
        (1.914602 - 0.004817 * centuries - 0.000014 * centuries**2)
        * math.sin(anomaly_rad)
        + (0.019993 - 0.000101 * centuries) * math.sin(2.0 * anomaly_rad)
        + 0.000289 * math.sin(3.0 * anomaly_rad)
    )
    omega = 125.04 - 1934.136 * centuries
    apparent_longitude = (
        mean_longitude
        + centre
        - 0.00569
        - 0.00478 * math.sin(math.radians(omega))
    )
    obliquity_seconds = 21.448 - centuries * (
        46.815 + centuries * (0.00059 - centuries * 0.001813)
    )
    mean_obliquity = 23.0 + (26.0 + obliquity_seconds / 60.0) / 60.0
    obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(omega))
    longitude_rad = math.radians(apparent_longitude)
    obliquity_rad = math.radians(obliquity)
    right_ascension = math.degrees(
        math.atan2(
            math.cos(obliquity_rad) * math.sin(longitude_rad),
            math.cos(longitude_rad),
        )
    ) % 360.0
    declination = math.degrees(
        math.asin(math.sin(obliquity_rad) * math.sin(longitude_rad))
    )
    return julian_day, right_ascension, declination


def _wgs84_height_km(position_ecef_m: np.ndarray) -> float:
    x, y, z = (float(value) for value in position_ecef_m)
    semi_major_axis = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = flattening * (2.0 - flattening)
    horizontal = math.hypot(x, y)
    latitude = math.atan2(z, horizontal * (1.0 - eccentricity_squared))
    height = 0.0
    for _ in range(10):
        prime_vertical = semi_major_axis / math.sqrt(
            1.0 - eccentricity_squared * math.sin(latitude) ** 2
        )
        height = horizontal / math.cos(latitude) - prime_vertical
        latitude = math.atan2(
            z,
            horizontal
            * (1.0 - eccentricity_squared * prime_vertical / (prime_vertical + height)),
        )
    return height / 1000.0


def extract_spectrum(
    radiance_path: Path,
    geolocation_path: Path,
    destination: Path,
    target_latitude: float,
    target_longitude: float,
    tai_minus_utc_seconds: int = 37,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("提取 NOAA CrIS SDR 光谱需要安装 h5py。") from exc

    with h5py.File(radiance_path, "r") as radiance_file, h5py.File(
        geolocation_path, "r"
    ) as geolocation_file:
        radiance_group = radiance_file["All_Data/CrIS-FS-SDR_All"]
        geolocation_group = geolocation_file["All_Data/CrIS-SDR-GEO_All"]
        latitude = np.asarray(geolocation_group["Latitude"], dtype=np.float64)
        longitude = np.asarray(geolocation_group["Longitude"], dtype=np.float64)
        valid = _valid_footprints(radiance_group, latitude, longitude)
        if not np.any(valid):
            raise ValueError("CrIS SDR 颗粒中没有质量合格的地理定位视场。")
        distance = _great_circle_distance_km(
            target_latitude, target_longitude, latitude, longitude
        )
        distance[~valid] = np.inf
        selected = tuple(int(value) for value in np.unravel_index(np.argmin(distance), distance.shape))
        scan_index, xtrack_index, fov_index = selected

        wavenumber_parts: list[np.ndarray] = []
        radiance_parts: list[np.ndarray] = []
        channel_counts: dict[str, int] = {}
        for band, wavenumber in BAND_GRIDS.items():
            spectral_radiance = np.asarray(
                radiance_group[f"ES_Real{band}"][selected], dtype=np.float64
            )
            if spectral_radiance.shape != wavenumber.shape:
                raise ValueError(
                    f"{band}通道数异常：{spectral_radiance.size}，预期{wavenumber.size}。"
                )
            finite = np.isfinite(spectral_radiance)
            # NOAA SDR uses mW/(m2 sr cm-1); validation uses W/(m2 sr um).
            radiance_per_um = spectral_radiance[finite] * np.square(wavenumber[finite]) / 1.0e7
            wavenumber_parts.append(wavenumber[finite])
            radiance_parts.append(radiance_per_um)
            channel_counts[band] = int(np.count_nonzero(finite))

        wavenumber_cm = np.concatenate(wavenumber_parts)
        radiance_w_m2_sr_um = np.concatenate(radiance_parts)
        order = np.argsort(wavenumber_cm)
        wavenumber_cm = wavenumber_cm[order]
        radiance_w_m2_sr_um = radiance_w_m2_sr_um[order]
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            wavenumber_cm=wavenumber_cm,
            radiance_w_m2_sr_um=radiance_w_m2_sr_um,
        )

        for_time = int(geolocation_group["FORTime"][scan_index, xtrack_index])
        observation_time = _iet_microseconds_to_utc(for_time, tai_minus_utc_seconds)
        julian_day, solar_right_ascension, solar_declination = _solar_equatorial_position(
            observation_time
        )
        spacecraft_position = np.asarray(
            geolocation_group["SCPosition"][scan_index], dtype=np.float64
        )
        solar_azimuth_raw = float(geolocation_group["SolarAzimuthAngle"][selected])
        metadata = {
            "source_radiance_file": radiance_path.name,
            "source_geolocation_file": geolocation_path.name,
            "product": "NOAA-20 CrIS-FS-SDR",
            "platform": "JPSS-1 / NOAA-20",
            "instrument": "CrIS",
            "indices": {
                "scan": scan_index,
                "xtrack": xtrack_index,
                "fov": fov_index,
            },
            "for_number": xtrack_index + 1,
            "fov_number": fov_index + 1,
            "target_latitude_deg": float(target_latitude),
            "target_longitude_deg": float(target_longitude),
            "target_distance_km": float(distance[selected]),
            "observation_time_utc": observation_time.isoformat().replace("+00:00", "Z"),
            "latitude_deg": float(latitude[selected]),
            "longitude_deg": float(longitude[selected]),
            "surface_height_m": float(geolocation_group["Height"][selected]),
            "satellite_height_km": _wgs84_height_km(spacecraft_position),
            "satellite_range_km": float(
                geolocation_group["SatelliteRange"][selected]
            )
            / 1000.0,
            "satellite_zenith_deg": float(
                geolocation_group["SatelliteZenithAngle"][selected]
            ),
            "satellite_azimuth_deg": float(
                geolocation_group["SatelliteAzimuthAngle"][selected]
            )
            % 360.0,
            "solar_zenith_deg": float(
                geolocation_group["SolarZenithAngle"][selected]
            ),
            "solar_azimuth_deg": solar_azimuth_raw % 360.0,
            "solar_right_ascension_deg": solar_right_ascension,
            "solar_declination_deg": solar_declination,
            "julian_day": julian_day,
            "spacecraft_position_ecef_m": spacecraft_position.tolist(),
            "quality_flags": {
                "QF1_SCAN_CRISSDR": int(radiance_group["QF1_SCAN_CRISSDR"][scan_index]),
                "QF2_CRISSDR": np.asarray(
                    radiance_group["QF2_CRISSDR"][scan_index, fov_index]
                ).astype(int).tolist(),
                "QF3_CRISSDR": np.asarray(radiance_group["QF3_CRISSDR"][selected])
                .astype(int)
                .tolist(),
                "QF4_CRISSDR": np.asarray(radiance_group["QF4_CRISSDR"][selected])
                .astype(int)
                .tolist(),
                "QF1_CRISSDRGEO": int(
                    geolocation_group["QF1_CRISSDRGEO"][scan_index]
                ),
            },
            "channel_counts": channel_counts,
            "spectral_point_count": int(wavenumber_cm.size),
            "wavenumber_min_cm-1": float(wavenumber_cm.min()),
            "wavenumber_max_cm-1": float(wavenumber_cm.max()),
            "radiance_unit": "W m-2 sr-1 um-1",
            "source_radiance_unit": "mW m-2 sr-1 (cm-1)-1",
        }

    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata["output_file"] = str(destination)
    metadata["metadata_file"] = str(metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 NOAA CrIS-FS-SDR 与配套 GEO HDF5 提取最近的合格单FOV光谱。"
    )
    parser.add_argument("radiance", type=Path, help="SCRIF CrIS-FS-SDR HDF5 文件")
    parser.add_argument("geolocation", type=Path, help="GCRSO CrIS-SDR-GEO HDF5 文件")
    parser.add_argument("destination", type=Path, help="输出验证 NPZ 文件")
    parser.add_argument("--latitude", type=float, required=True, help="目标纬度（deg）")
    parser.add_argument("--longitude", type=float, required=True, help="目标经度（deg）")
    parser.add_argument(
        "--tai-minus-utc",
        type=int,
        default=37,
        help="观测日期的TAI-UTC秒数，2025年为37",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = extract_spectrum(
        args.radiance,
        args.geolocation,
        args.destination,
        args.latitude,
        args.longitude,
        args.tai_minus_utc,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
