from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.atmospheric_radiation_manager import (
    LayeredAtmosphereSolver,
    ModisDataManager,
    build_specified_location_sample,
    local_direction_from_zenith_azimuth,
    solar_direction_from_sample,
)
from core.spectrum_validation_manager import SpectrumValidationManager


class MigratedCoreTests(unittest.TestCase):
    def test_modis_packed_values_and_latitude_orientation(self) -> None:
        packed = np.array([[-32767, -360], [4000, 8000]], dtype=np.int16)
        unpacked = ModisDataManager._unpack_numeric(
            packed,
            {
                "_FillValue": -32767,
                "valid_min": -1000,
                "valid_max": 10000,
                "scale_factor": 0.005,
                "add_offset": 0.0,
            },
        )
        self.assertTrue(np.isnan(unpacked[0, 0]))
        self.assertAlmostEqual(float(unpacked[0, 1]), -1.8)
        self.assertAlmostEqual(float(unpacked[1, 1]), 40.0)

        with tempfile.TemporaryDirectory() as directory:
            north_first = Path(directory) / "north_first.npz"
            south_first = Path(directory) / "south_first.npz"
            np.savez(north_first, latitude=np.array([89.0, 0.0, -89.0]))
            np.savez(south_first, latitude=np.array([-89.0, 0.0, 89.0]))
            self.assertTrue(ModisDataManager._is_north_first(north_first, 3))
            self.assertFalse(ModisDataManager._is_north_first(south_first, 3))

    def test_surface_temperature_gap_fill_stays_inside_surface_domain(self) -> None:
        manager = ModisDataManager()
        values = np.array([
            [280.0, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 310.0, np.nan],
        ])
        ocean = np.array([
            [True, True, True, True],
            [False, False, False, False],
        ])
        filled = manager._fill_missing_spatial(values, ocean, 288.15)
        filled = manager._fill_missing_spatial(filled, ~ocean, 288.15)
        np.testing.assert_allclose(filled[0], 280.0)
        np.testing.assert_allclose(filled[1], 310.0)

    def test_specified_location_validation(self) -> None:
        sample = build_specified_location_sample({
            "specified_latitude_deg": 35.0,
            "specified_longitude_deg": 139.0,
            "specified_altitude": 700.0,
            "specified_solar_zenith_deg": 155.6405,
            "specified_solar_azimuth_deg": 72.3456,
            "specified_satellite_zenith_deg": 40.390335,
            "specified_satellite_azimuth_deg": 83.462189,
        })
        self.assertEqual(sample["lat"], 35.0)
        self.assertEqual(sample["lon"], 139.0)
        self.assertAlmostEqual(sample["solar_zenith"], 155.6405)
        self.assertAlmostEqual(sample["solar_azimuth"], 72.3456)
        self.assertAlmostEqual(sample["view_zenith"], 40.390335)
        self.assertAlmostEqual(sample["view_azimuth"], 83.462189)

        latitude = np.deg2rad(sample["lat"])
        longitude = np.deg2rad(sample["lon"])
        local_up = np.asarray([
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ])
        sun = solar_direction_from_sample(sample)
        measured_zenith = np.rad2deg(
            np.arccos(np.clip(np.dot(local_up, sun), -1.0, 1.0))
        )
        self.assertAlmostEqual(float(measured_zenith), 155.6405, places=6)
        local_north = np.asarray([
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ])
        local_east = np.asarray([
            -np.sin(longitude),
            np.cos(longitude),
            0.0,
        ])
        measured_azimuth = (
            np.rad2deg(np.arctan2(np.dot(sun, local_east), np.dot(sun, local_north)))
            % 360.0
        )
        self.assertAlmostEqual(float(measured_azimuth), 72.3456, places=6)
        view = local_direction_from_zenith_azimuth(
            sample["lat"], sample["lon"],
            sample["view_zenith"], sample["view_azimuth"],
        )
        expected_cosine = (
            np.cos(np.deg2rad(sample["solar_zenith"]))
            * np.cos(np.deg2rad(sample["view_zenith"]))
            + np.sin(np.deg2rad(sample["solar_zenith"]))
            * np.sin(np.deg2rad(sample["view_zenith"]))
            * np.cos(np.deg2rad(
                sample["solar_azimuth"] - sample["view_azimuth"]
            ))
        )
        self.assertAlmostEqual(float(np.linalg.norm(view)), 1.0, places=12)
        self.assertAlmostEqual(float(np.dot(view, sun)), expected_cosine, places=6)

        with self.assertRaisesRegex(ValueError, "0～180"):
            build_specified_location_sample({
                "specified_altitude": 700.0,
                "specified_solar_zenith_deg": 181.0,
            })
        with self.assertRaisesRegex(ValueError, "0～360"):
            build_specified_location_sample({
                "specified_altitude": 700.0,
                "specified_solar_zenith_deg": 45.0,
                "specified_solar_azimuth_deg": 361.0,
            })
        with self.assertRaisesRegex(ValueError, "0～90"):
            build_specified_location_sample({
                "specified_altitude": 700.0,
                "specified_solar_zenith_deg": 45.0,
                "specified_satellite_zenith_deg": 90.0,
            })

    def test_thermal_transfer_uses_slant_layer_optical_depth(self) -> None:
        solver = LayeredAtmosphereSolver()
        wavenumber = np.asarray([900.0, 1000.0])
        tau = np.full((1, 35, 2), 0.01)
        zeros = np.zeros_like(tau)
        vertical = solver._arte_thermal_multiple_scattering(
            wavenumber, tau, zeros, zeros,
            np.asarray([300.0]), np.asarray([1.0]), np.asarray([1.0]),
        )
        view_mu = float(np.cos(np.deg2rad(40.390335)))
        slanted = solver._arte_thermal_multiple_scattering(
            wavenumber, tau, zeros, zeros,
            np.asarray([300.0]), np.asarray([1.0]), np.asarray([view_mu]),
        )
        self.assertTrue(np.all(np.isfinite(slanted)))
        self.assertFalse(np.allclose(slanted, vertical * view_mu))
        self.assertTrue(np.all(slanted < vertical))

    def test_legacy_solar_coordinates_are_migrated_to_zenith(self) -> None:
        sample = build_specified_location_sample({
            "specified_latitude_deg": 35.0,
            "specified_longitude_deg": 139.0,
            "specified_altitude": 700.0,
            "specified_solar_right_ascension_deg": 10.0,
            "specified_solar_declination_deg": -5.0,
        })
        self.assertIn("solar_zenith", sample)
        self.assertIn("solar_azimuth", sample)
        self.assertNotIn("right_ascension", sample)
        self.assertNotIn("declination", sample)
        self.assertGreaterEqual(sample["solar_zenith"], 0.0)
        self.assertLessEqual(sample["solar_zenith"], 180.0)

    def test_layered_optical_depth_import_and_correction(self) -> None:
        wavenumber = np.array([500.0, 1000.0, 1500.0])
        tau = np.full((3, 35), 0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tau.csv"
            np.savetxt(
                path,
                np.column_stack((wavenumber, tau)),
                delimiter=",",
                header="wavenumber_cm," + ",".join(f"layer_{i:02d}" for i in range(35)),
                comments="",
            )
            solver = LayeredAtmosphereSolver()
            solver.load_absorption_optical_depth(path)
            solver.apply_optical_depth_corrections([{
                "wavelength_min_um": 6.0,
                "wavelength_max_um": 12.0,
                "factor": 2.0,
            }])
            self.assertEqual(solver._absorption_tau.shape, (35, 3))
            self.assertGreater(float(solver._absorption_tau[0, 1]), 0.1)

    def test_optical_depth_import_loads_sibling_temperature_profile(self) -> None:
        wavenumber = np.array([640.0, 700.0, 750.0])
        tau = np.full((3, 35), 0.01)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optical_depth_path = root / "optical_depth_total.csv"
            np.savetxt(
                optical_depth_path,
                np.column_stack((wavenumber, tau)),
                delimiter=",",
                header="wavenumber_cm," + ",".join(
                    f"layer_{index + 1:02d}" for index in range(35)
                ),
                comments="",
            )
            solver = LayeredAtmosphereSolver()
            expected_temperature = np.linspace(286.0, 205.0, 35)
            np.savetxt(
                root / "atmospheric_profile_35_layers.csv",
                np.column_stack((solver.altitude_mid_km, expected_temperature)),
                delimiter=",",
                header="altitude_mid(km),temperature(K)",
                comments="",
            )
            solver.load_absorption_optical_depth(optical_depth_path)
            np.testing.assert_allclose(solver.temperature_k, expected_temperature)
            self.assertTrue(
                solver._temperature_profile_source.endswith(
                    "atmospheric_profile_35_layers.csv"
                )
            )
            solver.set_upper_atmosphere_temperature_offset(3.0)
            low = solver.altitude_mid_km < 10.0
            high = ~low
            np.testing.assert_allclose(solver.temperature_k[low], expected_temperature[low])
            np.testing.assert_allclose(
                solver.temperature_k[high], expected_temperature[high] + 3.0
            )

    def test_spectrum_validation_metrics(self) -> None:
        manager = SpectrumValidationManager()
        simulated = manager.build_spectrum(
            {"wavelength_um": np.array([8.0, 10.0, 12.0]), "value": np.array([1.0, 2.0, 1.0])},
            "wavelength_um",
            "value",
            "wavelength_um",
        )
        observed = manager.build_spectrum(
            {"wavelength_um": np.array([8.0, 10.0, 12.0]), "value": np.array([1.0, 2.0, 1.0])},
            "wavelength_um",
            "value",
            "wavelength_um",
        )
        metrics = manager.compare(simulated, observed)["metrics"]
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
