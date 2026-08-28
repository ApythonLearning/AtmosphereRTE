from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "resources" / "data" / "earth_environment" / "20250203"
RAW = OUTPUT / "raw"
PATMOS = RAW / "patmosx_noaa20_20250203.nc"
OISST = RAW / "oisst_20250203.nc"
AOT = RAW / "avhrr_aot_20250203.nc"
MERRA2 = OUTPUT / "MERRA2_400.tavg1_2d_aer_Nx.20250203.nc4"
MOD11C1 = OUTPUT / "MOD11C1.A2025034.061.2025036053155.hdf"
MODIS_SST = OUTPUT / "TERRA_MODIS.20250203.L3m.DAY.SST.sst.4km.nc"
MOD11C1_URL = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
    "MOD11C1.061/MOD11C1.A2025034.061.2025036053155/"
    "MOD11C1.A2025034.061.2025036053155.hdf"
)
MODIS_SST_URL = (
    "https://oceandata.sci.gsfc.nasa.gov/getfile/"
    "TERRA_MODIS.20250203.L3m.DAY.SST.sst.4km.nc"
)
MERRA2_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/"
    "M2T1NXAER.5.12.4/2025/02/MERRA2_400.tavg1_2d_aer_Nx.20250203.nc4"
)
SOURCE_URLS = {
    PATMOS: "https://www.ncei.noaa.gov/data/avhrr-hirs-reflectance-and-cloud-properties-patmosx/access/2025/patmosx_v06r00-preliminary_NOAA-20_asc_d20250203_c20250918.nc",
    OISST: "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/202502/oisst-avhrr-v02r01.20250203.nc",
    AOT: "https://www.ncei.noaa.gov/data/avhrr-aerosol-optical-thickness/access/daily/2025/AOT_AVHRR_v04r00-preliminary_daily-avg_20250203_c20250421.nc",
}


def numeric(variable: object) -> np.ndarray:
    values = np.ma.asarray(variable[:], dtype=np.float64)
    return np.asarray(values.filled(np.nan), dtype=np.float64).squeeze()


def block_mean(values: np.ndarray, row_factor: int, column_factor: int) -> np.ndarray:
    rows = values.shape[0] // row_factor
    columns = values.shape[1] // column_factor
    array = values[: rows * row_factor, : columns * column_factor]
    blocks = array.reshape(rows, row_factor, columns, column_factor)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(1, 3))
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    return np.divide(total, count, out=np.full((rows, columns), np.nan), where=count > 0)


def block_mode(values: np.ndarray, row_factor: int, column_factor: int) -> np.ndarray:
    rows = values.shape[0] // row_factor
    columns = values.shape[1] // column_factor
    blocks = values[: rows * row_factor, : columns * column_factor].reshape(
        rows, row_factor, columns, column_factor
    )
    classes = np.arange(14, dtype=np.int16)
    counts = np.stack(
        [(blocks == value).sum(axis=(1, 3)) for value in classes], axis=0
    )
    return classes[np.argmax(counts, axis=0)]


def nearest_fill(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    filled = values.copy()
    valid = np.isfinite(filled) & target
    if not np.any(valid):
        return filled
    # Pure NumPy nearest-neighbour-like propagation constrained to land cells.
    # Longitude is periodic; latitude does not wrap across the poles.
    for _ in range(values.shape[0] + values.shape[1]):
        missing = target & ~valid
        if not np.any(missing):
            break
        total = np.zeros(values.shape, dtype=float)
        count = np.zeros(values.shape, dtype=np.int8)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            neighbor_values = np.roll(filled, shift, axis=axis)
            neighbor_valid = np.roll(valid, shift, axis=axis)
            if axis == 0:
                edge = 0 if shift > 0 else -1
                neighbor_valid[edge, :] = False
            total += np.where(neighbor_valid, neighbor_values, 0.0)
            count += neighbor_valid
        newly_filled = missing & (count > 0)
        if not np.any(newly_filled):
            break
        filled[newly_filled] = total[newly_filled] / count[newly_filled]
        valid[newly_filled] = True
    return filled


def create_2d_product(
    path: Path,
    variables: dict[str, tuple[np.ndarray, str, str]],
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    title: str,
    source: str,
) -> None:
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("latitude", latitude.size)
        dataset.createDimension("longitude", longitude.size)
        lat = dataset.createVariable("latitude", "f4", ("latitude",))
        lon = dataset.createVariable("longitude", "f4", ("longitude",))
        lat[:] = latitude
        lon[:] = longitude
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        dataset.title = title
        dataset.source = source
        dataset.observation_date = "2025-02-03"
        dataset.processing_note = (
            "Converted to a compact regular grid for ARTE Atmosphere; no data from other years used."
        )
        for name, (values, units, long_name) in variables.items():
            variable = dataset.createVariable(
                name,
                "f4" if np.issubdtype(values.dtype, np.floating) else "i2",
                ("latitude", "longitude"),
                zlib=True,
                complevel=6,
                fill_value=np.float32(np.nan)
                if np.issubdtype(values.dtype, np.floating)
                else np.int16(-32768),
            )
            variable[:] = values
            variable.units = units
            variable.long_name = long_name


def prepare_patmos() -> list[Path]:
    with Dataset(PATMOS) as source:
        latitude = numeric(source.variables["latitude"])
        longitude = numeric(source.variables["longitude"])
        surface_temperature = numeric(source.variables["surface_temperature_retrieved"])
        surface_type_values = numeric(source.variables["surface_type"])
        surface_type = np.where(
            np.isfinite(surface_type_values), surface_type_values, 0
        ).astype(np.int16)
        cloud_fraction = numeric(source.variables["cloud_fraction"])
        cloud_temperature = numeric(source.variables["cld_temp_acha"])
        cloud_height = numeric(source.variables["cld_height_acha"])
        cloud_radius = numeric(source.variables["cld_reff_dcomp"])
        cloud_optical_depth = numeric(source.variables["cld_opd_dcomp"])
        water_probability = numeric(source.variables["water_cloud_probability"])

    latitude = block_mean(latitude[:, None], 10, 1).reshape(-1)
    longitude = block_mean(longitude[None, :], 1, 10).reshape(-1)
    surface_type = block_mode(surface_type, 10, 10)
    surface_temperature = block_mean(surface_temperature, 10, 10)
    land = surface_type > 0
    surface_temperature[~land] = np.nan
    surface_temperature = nearest_fill(surface_temperature, land)

    cloud_fraction_high = cloud_fraction.copy()
    cloudy = np.isfinite(cloud_fraction_high) & (cloud_fraction_high > 0.0)
    liquid = cloudy & np.isfinite(water_probability) & (water_probability >= 0.5)
    cloud_fraction = block_mean(cloud_fraction_high, 10, 10)
    cloud_temperature = block_mean(np.where(cloudy, cloud_temperature, np.nan), 10, 10)
    cloud_height = block_mean(np.where(cloudy, cloud_height, np.nan), 10, 10)
    cloud_radius = block_mean(np.where(liquid, cloud_radius, np.nan), 10, 10)
    cloud_optical_depth = block_mean(
        np.where(liquid, cloud_optical_depth, np.nan), 10, 10
    )

    # PATMOS-x is south-to-north. The MODIS workbench expects north in row zero.
    latitude = latitude[::-1]
    surface_temperature = surface_temperature[::-1]
    surface_type = surface_type[::-1]
    cloud_fraction = cloud_fraction[::-1]
    cloud_temperature = cloud_temperature[::-1]
    cloud_height = cloud_height[::-1]
    cloud_radius = cloud_radius[::-1]
    cloud_optical_depth = cloud_optical_depth[::-1]

    # Convert the UMD classes carried by the 2025 PATMOS-x file to IGBP-like codes.
    umd_to_igbp = np.asarray([0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 12, 16, 13])
    land_type = umd_to_igbp[np.clip(surface_type, 0, 13)].astype(np.int16)
    source_name = PATMOS.name

    land_temperature_path = OUTPUT / "NOAA20_PATMOSX_LST_20250203.nc"
    create_2d_product(
        land_temperature_path,
        {"LST_Day_CMG": (surface_temperature.astype(np.float32), "K", "land surface temperature")},
        latitude=latitude,
        longitude=longitude,
        title="NOAA-20 PATMOS-x land surface temperature substitute for MOD11",
        source=source_name,
    )

    cloud_path = OUTPUT / "NOAA20_PATMOSX_CLOUD_20250203.nc"
    create_2d_product(
        cloud_path,
        {
            "Cloud_Fraction_Mean_Mean": (cloud_fraction.astype(np.float32), "1", "cloud fraction"),
            "Cloud_Top_Temperature_Mean_Mean": (cloud_temperature.astype(np.float32), "K", "cloud top temperature"),
            "Cloud_Top_Height_Mean_Mean": (cloud_height.astype(np.float32), "m", "cloud top height"),
            "Cloud_Effective_Radius_Liquid_Mean_Mean": (cloud_radius.astype(np.float32), "micron", "liquid cloud effective radius"),
            "Cloud_Optical_Thickness_Liquid_Mean_Mean": (cloud_optical_depth.astype(np.float32), "1", "liquid cloud optical thickness"),
        },
        latitude=latitude,
        longitude=longitude,
        title="NOAA-20 PATMOS-x cloud substitute for MOD08",
        source=source_name,
    )

    land_type_path = OUTPUT / "NOAA20_PATMOSX_LAND_TYPE_20250203.nc"
    create_2d_product(
        land_type_path,
        {"LC_Type1": (land_type, "class", "IGBP-like land cover converted from PATMOS-x UMD surface type")},
        latitude=latitude,
        longitude=longitude,
        title="NOAA-20 PATMOS-x 2025 surface type substitute for MCD12",
        source=source_name,
    )
    return [land_temperature_path, cloud_path, land_type_path]


def prepare_oisst() -> Path:
    with Dataset(OISST) as source:
        latitude = numeric(source.variables["lat"])
        longitude = numeric(source.variables["lon"])
        sst = numeric(source.variables["sst"])
    canonical_longitude = (longitude + 180.0) % 360.0 - 180.0
    order = np.argsort(canonical_longitude)
    path = OUTPUT / "NOAA_OISST_SST_20250203.nc"
    create_2d_product(
        path,
        {"SST": (sst[:, order].astype(np.float32), "degree_Celsius", "daily sea surface temperature")},
        latitude=latitude,
        longitude=canonical_longitude[order],
        title="NOAA OISST AVHRR daily SST substitute for MOD28",
        source=OISST.name,
    )
    return path


def prepare_aot() -> Path:
    with Dataset(AOT) as source:
        latitude = numeric(source.variables["latitude"])
        longitude = numeric(source.variables["longitude"])
        aot_650 = numeric(source.variables["aot1"])
    latitude = block_mean(latitude[:, None], 10, 1).reshape(-1)
    longitude = block_mean(longitude[None, :], 1, 10).reshape(-1)
    aot_650 = block_mean(aot_650, 10, 10)
    angstrom = np.full(aot_650.shape, 1.3, dtype=np.float32)
    angstrom[~np.isfinite(aot_650)] = np.nan
    aot_550 = aot_650 * (0.65 / 0.55) ** angstrom
    path = OUTPUT / "NOAA_AVHRR_AOT_20250203_MERRA_COMPAT.nc"
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("lat", latitude.size)
        dataset.createDimension("lon", longitude.size)
        time = dataset.createVariable("time", "f8", ("time",))
        lat = dataset.createVariable("lat", "f4", ("lat",))
        lon = dataset.createVariable("lon", "f4", ("lon",))
        tau = dataset.createVariable("TOTEXTTAU", "f4", ("time", "lat", "lon"), zlib=True, complevel=6, fill_value=np.float32(np.nan))
        exponent = dataset.createVariable("TOTANGSTR", "f4", ("time", "lat", "lon"), zlib=True, complevel=6, fill_value=np.float32(np.nan))
        time[:] = [12.0]
        lat[:] = latitude
        lon[:] = longitude
        tau[0] = aot_550.astype(np.float32)
        exponent[0] = angstrom
        time.units = "hours since 2025-02-03 00:00:00"
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        tau.units = "1"
        exponent.units = "1"
        dataset.title = "NOAA AVHRR daily ocean AOT in MERRA-compatible layout"
        dataset.source = AOT.name
        dataset.observation_date = "2025-02-03"
        dataset.processing_note = (
            "AOT at 0.65 um converted to 0.55 um using Angstrom exponent 1.3; "
            "missing land cells are intentionally left missing for the solver visibility fallback."
        )
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_sources() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url in SOURCE_URLS.items():
        if path == AOT and MERRA2.is_file():
            continue
        if path == OISST and MODIS_SST.is_file():
            continue
        if path.is_file():
            continue
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            os.replace(partial, path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare strict-2025 Earth-environment inputs."
    )
    parser.add_argument(
        "--keep-raw", action="store_true", help="Keep the large downloaded source files."
    )
    arguments = parser.parse_args()
    ensure_sources()
    aerosol_product = MERRA2 if MERRA2.is_file() else prepare_aot()
    patmos_products = prepare_patmos()
    land_temperature_product = MOD11C1 if MOD11C1.is_file() else patmos_products[0]
    if MOD11C1.is_file():
        patmos_products[0].unlink(missing_ok=True)
    sea_temperature_product = MODIS_SST if MODIS_SST.is_file() else prepare_oisst()
    products = [
        land_temperature_product,
        patmos_products[1],
        patmos_products[2],
        sea_temperature_product,
        aerosol_product,
    ]
    sources = {
        PATMOS.name: {
            "url": SOURCE_URLS[PATMOS],
            "sha256": sha256(PATMOS),
        },
    }
    if not MODIS_SST.is_file():
        sources[OISST.name] = {
            "url": SOURCE_URLS[OISST],
            "sha256": sha256(OISST),
        }
    if MOD11C1.is_file():
        sources[MOD11C1.name] = {
            "url": MOD11C1_URL,
            "sha256": sha256(MOD11C1),
        }
    if MODIS_SST.is_file():
        sources[MODIS_SST.name] = {
            "url": MODIS_SST_URL,
            "sha256": sha256(MODIS_SST),
        }
    if MERRA2.is_file():
        sources[MERRA2.name] = {
            "url": MERRA2_URL,
            "sha256": sha256(MERRA2),
        }
        notes = [
            "The aerosol product is the official hourly MERRA-2 M2T1NXAER field downloaded with Earthdata authentication.",
            "Every retained scientific input is dated 2025-02-03; no 2024 land-cover file is used.",
        ]
    else:
        sources[AOT.name] = {
            "url": SOURCE_URLS[AOT],
            "sha256": sha256(AOT),
        }
        notes = [
            "The aerosol fallback is satellite-derived over oceans and uses the solver visibility fallback over missing land cells.",
            "Every retained scientific input is dated 2025-02-03; no 2024 land-cover file is used.",
        ]
    manifest = {
        "observation_date": "2025-02-03",
        "strict_year": 2025,
        "products": {
            "land_temperature": products[0].name,
            "cloud": products[1].name,
            "land_type": products[2].name,
            "sea_temperature": products[3].name,
            "aerosol_merra_compatible": aerosol_product.name,
        },
        "sources": sources,
        "output_sha256": {path.name: sha256(path) for path in products},
        "notes": notes,
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path in products:
        print(f"{path.name}: {path.stat().st_size / 1024**2:.2f} MiB")
    if not arguments.keep_raw:
        for path in SOURCE_URLS:
            path.unlink(missing_ok=True)
        RAW.rmdir()


if __name__ == "__main__":
    main()
