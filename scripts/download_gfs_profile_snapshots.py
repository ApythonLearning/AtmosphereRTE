from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

from eccodes import codes_get, codes_get_values, codes_new_from_message, codes_release
from netCDF4 import Dataset
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PRESSURE_PATTERN = re.compile(
    r":(?P<variable>TMP|SPFH|O3MR):(?P<level>[0-9.]+) mb:anl:"
)
SURFACE_PATTERNS = {
    ":PRES:surface:anl:": "surface_pressure_pa",
    ":HGT:surface:anl:": "surface_geopotential_height_m",
}
OUTPUT_NAMES = {
    "TMP": "temperature_k",
    "SPFH": "specific_humidity_kg_kg",
    "O3MR": "ozone_mass_mixing_ratio_kg_kg",
}


def month_sequence(first: str, last: str) -> list[str]:
    start = datetime.strptime(first, "%Y%m")
    stop = datetime.strptime(last, "%Y%m")
    if stop < start:
        raise ValueError("结束月份不能早于开始月份。")
    months: list[str] = []
    current = start
    while current <= stop:
        months.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def snapshot_dates(first: str, last: str, day: int) -> list[str]:
    return [f"{month}{day:02d}" for month in month_sequence(first, last)]


def base_url(date: str) -> str:
    return (
        f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/00/atmos/"
        "gfs.t00z.pgrb2.0p25.anl"
    )


def http_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    session.headers.update({"User-Agent": "ARTE-Atmosphere/1.0 global-profile-builder"})
    return session


def parse_index(text: str) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) < 3:
            continue
        rows.append({"message": int(fields[0]), "start": int(fields[1]), "line": line})
    for index, row in enumerate(rows):
        row["end"] = rows[index + 1]["start"] - 1 if index + 1 < len(rows) else None

    selected: list[dict[str, Any]] = []
    levels: set[float] = set()
    for row in rows:
        match = PRESSURE_PATTERN.search(row["line"])
        if match:
            variable = match.group("variable")
            level = float(match.group("level"))
            levels.add(level)
            selected.append(row | {"kind": "pressure", "variable": variable, "level": level})
            continue
        for marker, output_name in SURFACE_PATTERNS.items():
            if marker in row["line"]:
                selected.append(row | {"kind": "surface", "output_name": output_name})
                break
    ordered_levels = sorted(levels)
    for variable in OUTPUT_NAMES:
        found = {item["level"] for item in selected if item.get("variable") == variable}
        if found != set(ordered_levels):
            raise ValueError(f"GFS索引中{variable}压力层不完整。")
    if not all(any(item.get("output_name") == name for item in selected) for name in SURFACE_PATTERNS.values()):
        raise ValueError("GFS索引中缺少地表气压或地形高度。")
    return selected, ordered_levels


def range_message(
    session: requests.Session, url: str, start: int, end: int | None
) -> bytes:
    if end is None:
        response = session.get(url, headers={"Range": f"bytes={start}-"}, timeout=(30, 180))
    else:
        response = session.get(
            url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 180)
        )
    response.raise_for_status()
    message = response.content
    if end is not None and len(message) != end - start + 1:
        raise IOError(f"GRIB范围响应长度错误：期望{end - start + 1}，实际{len(message)}")
    return message


def grid_from_message(message: bytes, stride: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    handle = codes_new_from_message(message)
    try:
        ni = int(codes_get(handle, "Ni"))
        nj = int(codes_get(handle, "Nj"))
        first_lon = float(codes_get(handle, "longitudeOfFirstGridPointInDegrees"))
        first_lat = float(codes_get(handle, "latitudeOfFirstGridPointInDegrees"))
        lon_step = float(codes_get(handle, "iDirectionIncrementInDegrees"))
        lat_step = float(codes_get(handle, "jDirectionIncrementInDegrees"))
        i_negative = bool(codes_get(handle, "iScansNegatively"))
        j_positive = bool(codes_get(handle, "jScansPositively"))
        if bool(codes_get(handle, "jPointsAreConsecutive")):
            raise ValueError("暂不支持纬度点连续的GRIB扫描顺序。")
        longitude = first_lon + np.arange(ni) * lon_step * (-1.0 if i_negative else 1.0)
        latitude = first_lat + np.arange(nj) * lat_step * (1.0 if j_positive else -1.0)
        return latitude[::stride], longitude[::stride], nj, ni
    finally:
        codes_release(handle)


def decode_field(message: bytes, nj: int, ni: int, stride: int) -> np.ndarray:
    handle = codes_new_from_message(message)
    try:
        values = np.asarray(codes_get_values(handle), dtype=np.float32)
        if values.size != nj * ni:
            raise ValueError(f"GRIB网格大小错误：{values.size} != {nj}x{ni}")
        return values.reshape(nj, ni)[::stride, ::stride]
    finally:
        codes_release(handle)


def create_output(
    path: Path,
    dates: list[str],
    levels: list[float],
    latitude: np.ndarray,
    longitude: np.ndarray,
    stride: int,
) -> Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = Dataset(path, "w", format="NETCDF4")
    output.createDimension("time", len(dates))
    output.createDimension("level", len(levels))
    output.createDimension("latitude", latitude.size)
    output.createDimension("longitude", longitude.size)
    time_variable = output.createVariable("time", "i8", ("time",))
    time_variable.units = "seconds since 1970-01-01 00:00:00 UTC"
    time_variable.long_name = "GFS analysis time"
    time_variable[:] = [
        int(datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())
        for date in dates
    ]
    level_variable = output.createVariable("level", "f4", ("level",))
    level_variable.units = "hPa"
    level_variable.positive = "down"
    level_variable[:] = levels
    latitude_variable = output.createVariable("latitude", "f4", ("latitude",))
    latitude_variable.units = "degrees_north"
    latitude_variable[:] = latitude
    longitude_variable = output.createVariable("longitude", "f4", ("longitude",))
    longitude_variable.units = "degrees_east"
    longitude_variable[:] = longitude

    chunks = (1, 1, min(45, latitude.size), min(90, longitude.size))
    for output_name in OUTPUT_NAMES.values():
        variable = output.createVariable(
            output_name,
            "f4",
            ("time", "level", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=chunks,
            fill_value=np.float32(np.nan),
        )
        variable.coordinates = "time level latitude longitude"
    output.variables["temperature_k"].units = "K"
    output.variables["specific_humidity_kg_kg"].units = "kg kg-1"
    output.variables["ozone_mass_mixing_ratio_kg_kg"].units = "kg kg-1"
    surface_chunks = (1, min(45, latitude.size), min(90, longitude.size))
    for output_name, units in (
        ("surface_pressure_pa", "Pa"),
        ("surface_geopotential_height_m", "m"),
    ):
        variable = output.createVariable(
            output_name,
            "f4",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=surface_chunks,
            fill_value=np.float32(np.nan),
        )
        variable.units = units
        variable.coordinates = "time latitude longitude"
    output.title = "Compact monthly global GFS atmospheric profile snapshots"
    output.source = "NOAA Global Forecast System 0.25 degree analysis"
    output.processing_note = (
        f"One 00 UTC analysis on day 15 of each month; every {stride}th native grid point; "
        "selected GRIB messages decoded in memory without persistent GRIB cache."
    )
    output.requested_dates = ",".join(dates)
    output.completed_dates = ""
    output.history = f"created {datetime.now(timezone.utc).isoformat()}"
    output.sync()
    return output


def build_dataset(dates: list[str], output_path: Path, stride: int) -> None:
    session = http_session()
    output: Dataset | None = None
    completed: list[str] = []
    source_records: list[dict[str, Any]] = []
    if output_path.exists():
        output = Dataset(output_path, "a")
        completed = [item for item in str(getattr(output, "completed_dates", "")).split(",") if item]
        if completed != dates[: len(completed)]:
            output.close()
            raise RuntimeError("现有文件的已完成日期与本次任务不一致。")
    try:
        for time_index, date in enumerate(dates):
            if date in completed:
                print(f"[{time_index + 1}/{len(dates)}] {date} 已完成，跳过", flush=True)
                continue
            url = base_url(date)
            print(f"[{time_index + 1}/{len(dates)}] 读取GFS索引 {date} 00 UTC", flush=True)
            index_response = session.get(url + ".idx", timeout=(30, 60))
            index_response.raise_for_status()
            selected, levels = parse_index(index_response.text)
            level_indices = {level: index for index, level in enumerate(levels)}
            first_message = range_message(
                session, url, int(selected[0]["start"]), selected[0]["end"]
            )
            if output is None:
                latitude, longitude, nj, ni = grid_from_message(first_message, stride)
                output = create_output(
                    output_path, dates, levels, latitude, longitude, stride
                )
            else:
                nj = int((len(output.dimensions["latitude"]) - 1) * stride + 1)
                ni = int(len(output.dimensions["longitude"]) * stride)

            variable_counts: dict[str, int] = {name: 0 for name in OUTPUT_NAMES}
            for item_index, item in enumerate(selected):
                message = (
                    first_message
                    if item_index == 0
                    else range_message(session, url, int(item["start"]), item["end"])
                )
                field = decode_field(message, nj, ni, stride)
                if item["kind"] == "pressure":
                    variable = str(item["variable"])
                    output.variables[OUTPUT_NAMES[variable]][
                        time_index, level_indices[float(item["level"])]
                    ] = field
                    variable_counts[variable] += 1
                else:
                    output.variables[str(item["output_name"])][time_index] = field
            completed.append(date)
            output.completed_dates = ",".join(completed)
            output.sync()
            source_records.append(
                {
                    "date": date,
                    "source": url,
                    "pressure_levels": len(levels),
                    "messages": len(selected),
                }
            )
            counts = ", ".join(f"{name}={count}" for name, count in variable_counts.items())
            print(
                f"[{time_index + 1}/{len(dates)}] 完成 {counts}；文件 "
                f"{output_path.stat().st_size / 1024**2:.1f} MiB",
                flush=True,
            )
    finally:
        if output is not None:
            output.close()
        session.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": "NOAA GFS 0.25 degree global analysis",
        "sampling": "00 UTC on day 15 of each month",
        "dates": dates,
        "native_resolution_deg": 0.25,
        "spatial_stride": stride,
        "output_resolution_deg": 0.25 * stride,
        "variables": list(OUTPUT_NAMES.values()) + list(SURFACE_PATTERNS.values()),
        "output_file": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "memory_strategy": "HTTP byte ranges; one GRIB message decoded at a time",
        "persistent_download_cache": False,
        "sources": source_records,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a compact global pressure-profile library from NOAA GFS analyses."
    )
    parser.add_argument("--first", default="202508", help="First YYYYMM month")
    parser.add_argument("--last", default="202607", help="Last YYYYMM month")
    parser.add_argument("--day", type=int, default=15, help="Snapshot day in each month")
    parser.add_argument("--stride", type=int, default=4, help="Spatial stride on 0.25 degree grid")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.day < 1 or arguments.day > 28:
        raise ValueError("采样日必须在1到28之间。")
    if arguments.stride < 1:
        raise ValueError("空间步长必须大于0。")
    dates = snapshot_dates(arguments.first, arguments.last, arguments.day)
    output_path = arguments.output.expanduser().resolve()
    build_dataset(dates, output_path, arguments.stride)
    print(f"完成：{output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
