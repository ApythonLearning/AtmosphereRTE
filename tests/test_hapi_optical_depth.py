from __future__ import annotations

import sys
import tempfile
import unittest
import json
import importlib.util
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hapi_optical_depth_manager import (  # noqa: E402
    ABSORBING_GASES,
    HapiOpticalDepthManager,
    LayeredAtmosphericProfile,
    NucapsAtmosphericProfileReader,
)
from core.atmospheric_radiation_manager import LayeredAtmosphereSolver  # noqa: E402


class HapiOpticalDepthTests(unittest.TestCase):
    def test_import_local_hitran_table_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            database = root / "database"
            source.mkdir()
            (source / "H2O.data").write_text("synthetic-line\n", encoding="utf-8")
            (source / "H2O.header").write_text(
                json.dumps({"table_name": "H2O", "number_of_rows": 1}),
                encoding="utf-8",
            )
            manager = HapiOpticalDepthManager(database_dir=database)
            result = manager.import_line_tables([source])
            self.assertEqual(result["imported_gases"], ["H2O"])
            self.assertEqual(result["referenced_gases"], ["H2O"])
            self.assertEqual(result["copied_gases"], [])
            self.assertTrue(manager.line_database_status(["H2O"])["H2O"])
            self.assertFalse((database / "H2O.data").exists())
            header = json.loads((source / "H2O.header").read_text(encoding="utf-8"))
            self.assertEqual(header["table_name"], "H2O")
            repeated = manager.import_line_tables([source])
            self.assertEqual(repeated["overwritten_gases"], [])
            restored_manager = HapiOpticalDepthManager(
                database_dir=database,
                table_sources=manager.export_table_sources(),
            )
            self.assertTrue(restored_manager.line_database_status(["H2O"])["H2O"])
            self.assertEqual(
                restored_manager.line_database_source("H2O"), (source, "H2O")
            )

            raw_database = root / "raw_database"
            raw_par = root / "downloaded_01.par"
            raw_par.write_text("01" + " " * 158 + "\n", encoding="ascii")
            raw_manager = HapiOpticalDepthManager(database_dir=raw_database)
            raw_result = raw_manager.import_line_tables([raw_par])
            self.assertEqual(raw_result["imported_gases"], ["H2O"])
            self.assertEqual(raw_result["copied_gases"], ["H2O"])
            raw_header = json.loads(
                (raw_database / "H2O.header").read_text(encoding="utf-8")
            )
            self.assertEqual(raw_header["table_name"], "H2O")
            self.assertEqual(raw_header["number_of_rows"], 1)

    def test_wavenumber_step_cannot_exceed_one(self) -> None:
        self.assertEqual(
            HapiOpticalDepthManager._validate_grid(500.0, 33_300.0, 0.5),
            (500.0, 33_300.0, 0.5),
        )
        with self.assertRaisesRegex(ValueError, "不超过1"):
            HapiOpticalDepthManager._validate_grid(500.0, 33_300.0, 1.01)

    def test_hitran_query_range_is_clipped_per_gas(self) -> None:
        gases = {gas.name: gas for gas in ABSORBING_GASES}
        self.assertEqual(
            HapiOpticalDepthManager._hitran_query_range(gases["CO"], 500.0, 33_300.0),
            (500.0, 14_477.377),
        )
        self.assertEqual(
            HapiOpticalDepthManager._hitran_query_range(gases["SO2"], 500.0, 33_300.0),
            (500.0, 4_159.945),
        )
        self.assertEqual(
            HapiOpticalDepthManager._hitran_query_range(gases["HNO3"], 500.0, 33_300.0),
            (500.0, 4_167.053),
        )
        self.assertIsNone(
            HapiOpticalDepthManager._hitran_query_range(gases["SO2"], 5_000.0, 6_000.0)
        )

    def test_conservative_layer_remapping_preserves_column_amount(self) -> None:
        source_boundaries = np.asarray([0.0, 2.0, 5.0, 10.0])
        source_columns = np.asarray([2.0, 6.0, 12.0])
        target_boundaries = np.asarray([0.0, 1.0, 4.0, 7.0, 10.0])
        remapped = NucapsAtmosphericProfileReader._conservative_remap(
            source_boundaries, source_columns, target_boundaries
        )
        self.assertTrue(np.isfinite(remapped).all())
        self.assertTrue((remapped >= 0.0).all())
        self.assertAlmostEqual(float(remapped.sum()), float(source_columns.sum()), places=12)

    @unittest.skipUnless(importlib.util.find_spec("netCDF4"), "需要 netCDF4")
    def test_bundled_nucaps_profile_detects_all_absorbing_gases(self) -> None:
        profile_root = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "data"
            / "atmospheric_profiles"
            / "nucaps_noaa20_20250203"
        )
        candidates = sorted(profile_root.glob("*.nc"))
        if not candidates:
            self.skipTest("未提供NUCAPS测试廓线。")
        inspection = NucapsAtmosphericProfileReader.inspect(candidates[0])
        self.assertEqual(
            set(inspection["gas_names"]),
            {"H2O", "CO2", "O3", "N2O", "CO", "CH4", "SO2", "HNO3"},
        )
        nearest = NucapsAtmosphericProfileReader.nearest_valid_for(
            inspection, -45.4019356, -168.3135986
        )
        self.assertEqual(nearest, 4)
        profile = NucapsAtmosphericProfileReader.read(candidates[0], nearest)
        self.assertEqual(profile.altitude_mid_km.size, 35)
        self.assertEqual(profile.altitude_boundaries_km[-1], 97.0)
        self.assertEqual(set(profile.gas_names), set(inspection["gas_names"]))
        for values in profile.gas_columns_molec_cm2.values():
            self.assertEqual(values.shape, (35,))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values >= 0.0).all())

    @unittest.skipUnless(importlib.util.find_spec("netCDF4"), "需要 netCDF4")
    def test_saved_total_csv_is_loadable_by_current_solver(self) -> None:
        manager = HapiOpticalDepthManager()
        boundaries = np.r_[0.0, np.cumsum(LayeredAtmosphereSolver.LAYER_HEIGHT_KM)]
        profile = LayeredAtmosphericProfile(
            source_path=Path("synthetic.nc"),
            for_index=0,
            observation_time_utc="2025-02-03T00:00:00Z",
            latitude_deg=0.0,
            longitude_deg=0.0,
            quality_flag=0,
            altitude_boundaries_km=boundaries,
            altitude_mid_km=0.5 * (boundaries[:-1] + boundaries[1:]),
            pressure_hpa=np.linspace(1000.0, 0.01, 35),
            temperature_k=np.linspace(288.0, 190.0, 35),
            gas_columns_molec_cm2={"H2O": np.ones(35)},
            gas_sources={"H2O": "synthetic"},
            source_level_count=35,
            source_top_altitude_km=97.0,
        )
        wavenumber = np.asarray([500.0, 500.5, 501.0])
        total_tau = np.arange(105, dtype=float).reshape(35, 3) / 1000.0
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = manager._save_result(
                temporary_dir,
                profile,
                wavenumber,
                total_tau,
                {"H2O": total_tau.astype(np.float32)},
                500.0,
                501.0,
                0.5,
                [],
            )
            solver = LayeredAtmosphereSolver()
            solver.load_absorption_optical_depth(result["total_optical_depth_file"])
            visualization = solver.get_absorption_visualization_data()
            np.testing.assert_allclose(visualization["wavenumber_cm"], wavenumber)
            np.testing.assert_allclose(visualization["total_tau_layers"], total_tau)

    def test_hitran_download_retries_after_interrupted_tls_read(self) -> None:
        class FakeHapi:
            def __init__(self) -> None:
                self.database_dir = Path()
                self.calls = 0
                self.LOCAL_TABLE_CACHE: dict[str, object] = {}

            def db_begin(self, database_dir: str) -> None:
                self.database_dir = Path(database_dir)

            def fetch(
                self, table_name: str, _molecule: int, _isotope: int, lower: float, upper: float
            ) -> None:
                self.calls += 1
                data_path = self.database_dir / f"{table_name}.data"
                if self.calls == 1:
                    data_path.write_text("partial", encoding="utf-8")
                    raise OSError("[SSL: NOT_ENOUGH_DATA] not enough data")
                data_path.write_text(f"line {lower:g}\nline {upper:g}\n", encoding="utf-8")
                header = {"table_name": table_name, "number_of_rows": -1}
                (self.database_dir / f"{table_name}.header").write_text(
                    json.dumps(header), encoding="utf-8"
                )
                self.LOCAL_TABLE_CACHE[table_name] = object()

        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = HapiOpticalDepthManager(database_dir=temporary_dir)
            fake_hapi = FakeHapi()
            manager._hapi = fake_hapi
            manager.DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 0.0
            downloaded = manager._ensure_line_tables(
                [ABSORBING_GASES[4]],
                500.0,
                501.0,
                lambda _value, _maximum, _message: None,
                lambda: False,
            )
            self.assertEqual(downloaded, ["CO"])
            self.assertEqual(fake_hapi.calls, 2)
            self.assertTrue(manager._valid_local_table("CO"))
            data = (Path(temporary_dir) / "CO.data").read_text(encoding="utf-8")
            self.assertNotIn("partial", data)
            self.assertEqual(data.splitlines(), ["line 500", "line 501"])

    def test_hitran_download_splits_repeatedly_interrupted_large_response(self) -> None:
        class FakeHapi:
            def __init__(self) -> None:
                self.database_dir = Path()
                self.ranges: list[tuple[float, float]] = []
                self.LOCAL_TABLE_CACHE: dict[str, object] = {}
                self.VARIABLES = {"GLOBAL_HOST": "http://hitran.org"}

            def db_begin(self, database_dir: str) -> None:
                self.database_dir = Path(database_dir)

            def fetch(
                self, table_name: str, _molecule: int, _isotope: int, lower: float, upper: float
            ) -> None:
                self.ranges.append((lower, upper))
                if upper - lower > 1_000.0:
                    raise OSError("[ASN1: NOT_ENOUGH_DATA] not enough data")
                (self.database_dir / f"{table_name}.data").write_text(
                    f"line {lower:g}\nline {upper:g}\n", encoding="utf-8"
                )
                (self.database_dir / f"{table_name}.header").write_text(
                    json.dumps({"table_name": table_name}), encoding="utf-8"
                )

        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = HapiOpticalDepthManager(database_dir=temporary_dir)
            fake_hapi = FakeHapi()
            manager._hapi = fake_hapi
            manager.DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 0.0
            downloaded = manager._ensure_line_tables(
                [ABSORBING_GASES[4]],
                500.0,
                2_500.0,
                lambda _value, _maximum, _message: None,
                lambda: False,
            )
            self.assertEqual(downloaded, ["CO"])
            self.assertEqual(fake_hapi.VARIABLES["GLOBAL_HOST"], "https://hitran.org")
            self.assertEqual(fake_hapi.ranges[:2], [(500.0, 2_500.0)] * 2)
            self.assertIn((500.0, 1_500.0), fake_hapi.ranges)
            self.assertIn((1_500.0, 2_500.0), fake_hapi.ranges)
            data = (Path(temporary_dir) / "CO.data").read_text(encoding="utf-8")
            self.assertEqual(data.splitlines(), ["line 500", "line 1500", "line 2500"])

    def test_download_uses_gas_specific_hitran_upper_bound(self) -> None:
        class FakeHapi:
            def __init__(self) -> None:
                self.database_dir = Path()
                self.ranges: list[tuple[float, float]] = []
                self.LOCAL_TABLE_CACHE: dict[str, object] = {}

            def db_begin(self, database_dir: str) -> None:
                self.database_dir = Path(database_dir)

            def fetch(
                self, table_name: str, _molecule: int, _isotope: int, lower: float, upper: float
            ) -> None:
                self.ranges.append((lower, upper))
                (self.database_dir / f"{table_name}.data").write_text(
                    f"line {lower:g} {upper:g}\n", encoding="utf-8"
                )
                (self.database_dir / f"{table_name}.header").write_text(
                    json.dumps({"table_name": table_name}), encoding="utf-8"
                )

        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = HapiOpticalDepthManager(database_dir=temporary_dir)
            fake_hapi = FakeHapi()
            manager._hapi = fake_hapi
            downloaded = manager._ensure_line_tables(
                [next(gas for gas in ABSORBING_GASES if gas.name == "SO2")],
                500.0,
                33_300.0,
                lambda _value, _maximum, _message: None,
                lambda: False,
            )
            self.assertEqual(downloaded, ["SO2"])
            self.assertEqual(fake_hapi.ranges, [(500.0, 4_159.945)])

    def test_parallel_worker_prefilters_and_reuses_exact_coefficient_cache(
        self,
    ) -> None:
        fake_hapi_source = '''
import numpy as np

VARIABLES = {"BACKEND_DATABASE_NAME": "."}
LOCAL_TABLE_CACHE = {}

def storage2cache(table_name):
    centers = np.asarray([499.0, 500.0, 500.5, 501.0, 502.0])
    LOCAL_TABLE_CACHE[table_name] = {
        "header": {"table_name": table_name, "number_of_rows": centers.size},
        "data": {"nu": centers, "dummy": np.arange(centers.size)},
    }

def absorptionCoefficient_Voigt(Components, SourceTables, Environment,
                                WavenumberGrid, **_kwargs):
    grid = np.asarray(WavenumberGrid, dtype=float)
    coefficient = np.full(grid.shape, Environment["T"] * 1.0e-25)
    return grid, coefficient
'''

        boundaries = np.asarray([0.0, 1.0, 2.0])
        profile = LayeredAtmosphericProfile(
            source_path=Path("synthetic.nc"),
            for_index=0,
            observation_time_utc="2025-02-03T00:00:00Z",
            latitude_deg=0.0,
            longitude_deg=0.0,
            quality_flag=0,
            altitude_boundaries_km=boundaries,
            altitude_mid_km=np.asarray([0.5, 1.5]),
            pressure_hpa=np.asarray([1000.0, 800.0]),
            temperature_k=np.asarray([280.0, 260.0]),
            gas_columns_molec_cm2={"H2O": np.asarray([2.0, 3.0])},
            gas_sources={"H2O": "synthetic"},
            source_level_count=2,
            source_top_altitude_km=2.0,
        )

        class FakeReader:
            @staticmethod
            def read(_path: str | Path, _for_index: int) -> LayeredAtmosphericProfile:
                return profile

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            hapi_path = root / "fake_hapi.py"
            hapi_path.write_text(fake_hapi_source, encoding="utf-8")
            database_dir = root / "database"
            database_dir.mkdir()
            (database_dir / "H2O.data").write_text("synthetic\n", encoding="utf-8")
            (database_dir / "H2O.header").write_text(
                json.dumps({"table_name": "H2O"}), encoding="utf-8"
            )
            manager = HapiOpticalDepthManager(
                hapi_path=hapi_path, database_dir=database_dir
            )
            manager.profile_reader = FakeReader()
            manager.DEFAULT_PREFILTER_MARGIN_CM = 0.1
            arguments = {
                "profile_path": root / "synthetic.nc",
                "for_index": 0,
                "wavenumber_min_cm": 500.0,
                "wavenumber_max_cm": 501.0,
                "wavenumber_step_cm": 0.5,
                "output_root": root / "output",
                "max_workers": 1,
                "use_cache": True,
                "prefilter_lines": True,
                "save_components": False,
            }
            first = manager.calculate(**arguments)
            profile.gas_columns_molec_cm2["H2O"] = np.asarray([4.0, 5.0])
            second = manager.calculate(**arguments)

            self.assertEqual(first["component_optical_depth_file"], "")
            self.assertEqual(first["optimization"]["coefficient_cache_hits"], 0)
            self.assertEqual(second["optimization"]["coefficient_cache_hits"], 1)
            gas_task = first["performance"]["gas_tasks"][0]
            self.assertEqual(gas_task["original_line_count"], 5)
            self.assertEqual(gas_task["filtered_line_count"], 3)
            second_values = np.loadtxt(
                second["total_optical_depth_file"], delimiter=",", skiprows=1
            )
            np.testing.assert_allclose(
                second_values[:, 1], np.full(3, 280.0e-25 * 4.0), rtol=1.0e-7
            )
            np.testing.assert_allclose(
                second_values[:, 2], np.full(3, 260.0e-25 * 5.0), rtol=1.0e-7
            )
            manifest = json.loads(
                Path(second["manifest_file"]).read_text(encoding="utf-8")
            )
            self.assertIsNone(manifest["files"]["component_optical_depth"])


if __name__ == "__main__":
    unittest.main()
