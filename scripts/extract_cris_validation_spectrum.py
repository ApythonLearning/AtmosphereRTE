from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BANDS = ("lw", "mw", "sw")


def _as_float(variable: Any) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _valid_footprints(dataset: Any) -> np.ndarray:
    shape = tuple(int(value) for value in dataset.variables["lat"].shape)
    valid = np.ones(shape, dtype=bool)
    for name in (
        "instrument_state",
        "cal_qualflag",
        "cal_lw_qualflag",
        "cal_mw_qualflag",
        "cal_sw_qualflag",
        "rad_lw_qc",
        "rad_mw_qc",
        "rad_sw_qc",
    ):
        if name in dataset.variables:
            valid &= np.asarray(dataset.variables[name][:]) == 0
    valid &= np.isfinite(_as_float(dataset.variables["lat"]))
    valid &= np.isfinite(_as_float(dataset.variables["lon"]))
    return valid


def _choose_footprint(dataset: Any) -> tuple[int, int, int]:
    valid = _valid_footprints(dataset)
    candidates = np.argwhere(valid)
    if candidates.size == 0:
        candidates = np.argwhere(
            np.isfinite(_as_float(dataset.variables["lat"]))
            & np.isfinite(_as_float(dataset.variables["lon"]))
        )
    if candidates.size == 0:
        raise ValueError("CrIS 文件中没有可用视场。")
    center = (np.asarray(valid.shape, dtype=np.float64) - 1.0) / 2.0
    normalized = (candidates - center) / np.maximum(center, 1.0)
    best = int(np.argmin(np.sum(normalized * normalized, axis=1)))
    return tuple(int(value) for value in candidates[best])


def extract_spectrum(
    source: Path,
    destination: Path,
    indices: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("提取 CrIS L1B 光谱需要安装 netCDF4。") from exc

    with Dataset(source, "r") as dataset:
        selected = indices or _choose_footprint(dataset)
        atrack, xtrack, fov = selected
        footprint_shape = tuple(int(value) for value in dataset.variables["lat"].shape)
        if any(index < 0 or index >= size for index, size in zip(selected, footprint_shape)):
            raise IndexError(
                f"视场索引 {selected} 超出 CrIS 数据范围 {footprint_shape}。"
            )

        wavenumber_parts: list[np.ndarray] = []
        radiance_parts: list[np.ndarray] = []
        channel_counts: dict[str, int] = {}
        quality_flags: dict[str, int] = {}
        for band in BANDS:
            wavenumber = _as_float(dataset.variables[f"wnum_{band}"])
            radiance_per_cm = _as_float(dataset.variables[f"rad_{band}"])[
                atrack, xtrack, fov, :
            ]
            valid = (
                np.isfinite(wavenumber)
                & (wavenumber > 0.0)
                & np.isfinite(radiance_per_cm)
            )
            wavenumber = wavenumber[valid]
            radiance_per_cm = radiance_per_cm[valid]
            # Source unit is mW/(m2 sr cm-1).  The validation workbench uses
            # W/(m2 sr um), and |d(wavenumber)/d(wavelength_um)| = wn**2/1e4.
            radiance_per_um = radiance_per_cm * np.square(wavenumber) / 1.0e7
            wavenumber_parts.append(wavenumber)
            radiance_parts.append(radiance_per_um)
            channel_counts[band.upper()] = int(wavenumber.size)
            qc_name = f"rad_{band}_qc"
            if qc_name in dataset.variables:
                quality_flags[qc_name] = int(
                    np.asarray(dataset.variables[qc_name][atrack, xtrack, fov]).item()
                )

        wavenumber = np.concatenate(wavenumber_parts)
        radiance = np.concatenate(radiance_parts)
        order = np.argsort(wavenumber)
        wavenumber = wavenumber[order]
        radiance = radiance[order]

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            wavenumber_cm=wavenumber,
            radiance_w_m2_sr_um=radiance,
        )
        metadata = {
            "source_file": source.name,
            "product": getattr(dataset, "shortname", "SNDRJ1CrISL1B"),
            "platform": getattr(dataset, "platform", "JPSS-1 / NOAA-20"),
            "instrument": getattr(dataset, "instrument", "CrIS"),
            "title": getattr(dataset, "title", ""),
            "time_coverage_start": getattr(dataset, "time_coverage_start", ""),
            "time_coverage_end": getattr(dataset, "time_coverage_end", ""),
            "indices": {"atrack": atrack, "xtrack": xtrack, "fov": fov},
            "latitude_deg": float(dataset.variables["lat"][atrack, xtrack, fov]),
            "longitude_deg": float(dataset.variables["lon"][atrack, xtrack, fov]),
            "satellite_zenith_deg": float(
                dataset.variables["sat_zen"][atrack, xtrack, fov]
            ),
            "quality_flags": quality_flags,
            "channel_counts": channel_counts,
            "spectral_point_count": int(wavenumber.size),
            "wavenumber_min_cm-1": float(wavenumber.min()),
            "wavenumber_max_cm-1": float(wavenumber.max()),
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
        description="从 NOAA-20/JPSS-1 CrIS L1B 文件提取单视场光谱。"
    )
    parser.add_argument("source", type=Path, help="SNDRJ1CrISL1B NetCDF 文件")
    parser.add_argument("destination", type=Path, help="输出 NPZ 文件")
    parser.add_argument(
        "--indices",
        type=int,
        nargs=3,
        metavar=("ATRACK", "XTRACK", "FOV"),
        help="指定视场索引；省略时自动选择中心附近的合格视场。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = tuple(args.indices) if args.indices is not None else None
    metadata = extract_spectrum(args.source, args.destination, indices)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
