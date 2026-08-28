from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "resources" / "data" / "earth_environment" / "20250203"
MANIFEST = OUTPUT / "source_manifest.json"


@dataclass(frozen=True)
class Product:
    role: str
    name: str
    url: str
    signatures: tuple[bytes, ...]


PRODUCTS = (
    Product(
        role="land_temperature",
        name="MOD11C1.A2025034.061.2025036053155.hdf",
        url=(
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
            "MOD11C1.061/MOD11C1.A2025034.061.2025036053155/"
            "MOD11C1.A2025034.061.2025036053155.hdf"
        ),
        signatures=(b"\x0e\x03\x13\x01",),
    ),
    Product(
        role="sea_temperature",
        name="TERRA_MODIS.20250203.L3m.DAY.SST.sst.4km.nc",
        url=(
            "https://oceandata.sci.gsfc.nasa.gov/getfile/"
            "TERRA_MODIS.20250203.L3m.DAY.SST.sst.4km.nc"
        ),
        signatures=(b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n"),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def total_size(response: requests.Response, offset: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1))
    length = response.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + (offset if response.status_code == 206 else 0)
    return None


def download(product: Product, token: str, retries: int = 10) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / product.name
    partial = target.with_name(target.name + ".part")
    if target.exists():
        validate_signature(target, product.signatures)
        print(f"已存在并通过签名检查：{product.name} ({target.stat().st_size} bytes)")
        return target

    headers_base = {"Authorization": f"Bearer {token}"}
    expected: int | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = dict(headers_base)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                product.url,
                headers=headers,
                stream=True,
                timeout=(30, 120),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                    partial.unlink(missing_ok=True)
                expected = total_size(response, offset)
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"下载 {product.name} 失败：{exc}") from exc
            print(f"连接中断，保留断点并重试 {attempt}/{retries}：{product.name}")
            time.sleep(min(2 * attempt, 15))
            continue

        actual = partial.stat().st_size
        if expected is None or actual == expected:
            validate_signature(partial, product.signatures)
            partial.replace(target)
            print(f"下载完成：{product.name} ({actual} bytes)")
            return target
        if actual > expected:
            partial.unlink()
            raise RuntimeError(
                f"下载大小超过服务器声明值：{product.name} ({actual} > {expected})"
            )
        print(f"文件未完整，继续断点下载 {actual}/{expected}：{product.name}")

    raise RuntimeError(f"下载未完成：{product.name}")


def validate_signature(path: Path, signatures: tuple[bytes, ...]) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(8)
    if not any(prefix.startswith(signature) for signature in signatures):
        raise RuntimeError(f"{path.name} 文件签名异常，可能下载到了登录页或错误页面。")


def update_manifest(paths: dict[str, Path]) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest["observation_date"] = "2025-02-03"
    manifest["strict_year"] = 2025
    products = manifest.setdefault("products", {})
    sources = manifest.setdefault("sources", {})
    checksums = manifest.setdefault("output_sha256", {})
    sources.pop("oisst_20250203.nc", None)
    for superseded in (
        "NOAA20_PATMOSX_LST_20250203.nc",
        "NOAA_OISST_SST_20250203.nc",
    ):
        checksums.pop(superseded, None)
    for product in PRODUCTS:
        path = paths[product.role]
        checksum = sha256(path)
        products[product.role] = product.name
        sources[product.name] = {
            "url": product.url,
            "sha256": checksum,
        }
        checksums[product.name] = checksum
    notes = manifest.setdefault("notes", [])
    note = (
        "Land temperature now uses MOD11C1.061; sea temperature uses the official "
        "Terra MODIS OB.DAAC Level-3 mapped daily SST successor to the legacy MOD28 naming."
    )
    if note not in notes:
        notes.append(note)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="断点下载并校验 2025-02-03 MOD11C1 与 Terra MODIS 日海温产品。"
    )
    parser.add_argument(
        "--keep-superseded",
        action="store_true",
        help="保留旧的 PATMOS-X 陆温与 NOAA OISST 替代文件。",
    )
    args = parser.parse_args()
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token and any(not (OUTPUT / product.name).exists() for product in PRODUCTS):
        raise RuntimeError("缺少 EARTHDATA_TOKEN 环境变量。")

    paths = {product.role: download(product, token) for product in PRODUCTS}
    update_manifest(paths)
    if not args.keep_superseded:
        for name in ("NOAA20_PATMOSX_LST_20250203.nc", "NOAA_OISST_SST_20250203.nc"):
            old_path = OUTPUT / name
            if old_path.exists():
                old_path.unlink()
                print(f"已清理被替代产品：{name}")
    print("SHA256：")
    for path in paths.values():
        print(f"  {path.name}  {sha256(path)}")


if __name__ == "__main__":
    main()
