from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.atmospheric_pattern_manager import AtmosphericPatternManager
from core.atmospheric_radiation_manager import LayeredAtmosphereSolver
from core.hapi_optical_depth_manager import LayeredAtmosphericProfileCsvReader


class AtmosphericPatternTests(unittest.TestCase):
    @staticmethod
    def _write_sample(root: Path, index: int, temperature_offset: float) -> Path:
        sample = root / f"sample_{index:02d}"
        sample.mkdir()
        altitude = np.linspace(0.5, 96.0, 35)
        pressure = 1013.25 * np.exp(-altitude / 7.2)
        temperature = 288.0 - 5.5 * np.minimum(altitude, 11.0) + temperature_offset
        water = 2.0e23 * np.exp(-altitude / 2.0) * (1.0 + index * 0.03)
        ozone = 2.0e18 * np.exp(-((altitude - 25.0) / 10.0) ** 2)
        profile_path = sample / "atmospheric_profile_35_layers.csv"
        np.savetxt(
            profile_path,
            np.column_stack([altitude, pressure, temperature, water, ozone]),
            delimiter=",",
            header=(
                "altitude_mid(km),pressure(hPa),temperature(K),"
                "column_H2O(molec_cm-2),column_O3(molec_cm-2)"
            ),
            comments="",
        )
        wavenumber = np.linspace(700.0, 704.0, 5)
        tau = (
            (0.002 + index * 0.0002)
            * np.exp(-altitude[:, None] / 8.0)
            * (1.0 + np.arange(wavenumber.size)[None, :] * 0.1)
        )
        np.savetxt(
            sample / "optical_depth_total.csv",
            np.column_stack([wavenumber, tau.T]),
            delimiter=",",
            header="wavenumber(cm-1)," + ",".join(f"layer_{i + 1:02d}" for i in range(35)),
            comments="",
        )
        (sample / "calculation_manifest.json").write_text(
            json.dumps(
                {
                    "profile": {
                        "latitude_deg": -60.0 + index * 20.0,
                        "longitude_deg": index * 30.0,
                        "observation_time_utc": f"2026-01-{index + 1:02d}T00:00:00Z",
                    }
                }
            ),
            encoding="utf-8",
        )
        return profile_path

    def test_autoencoder_patterns_interpolate_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = [
                self._write_sample(root, index, float(index % 3) * 2.0)
                for index in range(8)
            ]
            manager = AtmosphericPatternManager()
            summary = manager.fit_directory(
                root,
                cluster_count=4,
                latent_dimension=3,
                method="autoencoder",
                epochs=80,
            )
            self.assertEqual(summary["training_sample_count"], 8)
            self.assertEqual(summary["representative_count"], 4)
            self.assertTrue(summary["has_optical_library"])
            self.assertTrue(summary["has_visualization_data"])
            self.assertEqual(manager.training_labels.shape, (8,))
            self.assertEqual(int(np.sum(manager.cluster_counts)), 8)
            self.assertEqual(manager.cluster_profile_p10.shape[0], 4)
            self.assertGreater(manager.training_loss_history.size, 0)

            prediction = manager.predict(profiles[0], neighbor_count=3)
            self.assertFalse(prediction["out_of_distribution"])
            self.assertEqual(prediction["optical_depth_layers"].shape, (35, 5))
            self.assertAlmostEqual(float(np.sum(prediction["weights"])), 1.0)
            self.assertTrue(np.all(prediction["optical_depth_layers"] >= 0.0))

            predicted_path = manager.save_prediction(
                prediction, root / "prediction" / "optical_depth.csv"
            )
            solver = LayeredAtmosphereSolver()
            solver.load_absorption_optical_depth(predicted_path)
            self.assertIsNotNone(solver._absorption_tau)

            model_path = manager.save(root / "pattern_model.npz")
            restored = AtmosphericPatternManager()
            restored_summary = restored.load(model_path)
            restored_prediction = restored.predict(profiles[0], neighbor_count=3)
            self.assertEqual(restored_summary["training_sample_count"], 8)
            self.assertTrue(restored_summary["has_visualization_data"])
            np.testing.assert_array_equal(
                restored.training_labels, manager.training_labels
            )
            np.testing.assert_allclose(
                restored.cluster_profile_p90, manager.cluster_profile_p90
            )
            np.testing.assert_allclose(
                restored_prediction["optical_depth_layers"],
                prediction["optical_depth_layers"],
            )

    def test_out_of_distribution_profile_requires_hapi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                self._write_sample(root, index, index * 0.2)
            manager = AtmosphericPatternManager()
            manager.fit_directory(
                root, cluster_count=2, latent_dimension=2, method="pca"
            )
            extreme = self._write_sample(root, 99, 120.0)
            result = manager.predict(extreme)
            self.assertTrue(result["out_of_distribution"])
            self.assertTrue(result["requires_hapi"])

    def test_gfs_library_learns_profiles_before_exporting_final_modes(self) -> None:
        try:
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest("netCDF4 is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gfs_compact.nc4"
            with Dataset(source, "w") as dataset:
                dataset.createDimension("time", 2)
                dataset.createDimension("level", 5)
                dataset.createDimension("latitude", 3)
                dataset.createDimension("longitude", 4)
                dataset.createVariable("time", "i8", ("time",))[:] = [
                    1_750_000_000,
                    1_752_678_400,
                ]
                dataset.createVariable("level", "f4", ("level",))[:] = [
                    100.0,
                    300.0,
                    500.0,
                    700.0,
                    1000.0,
                ]
                dataset.createVariable("latitude", "f4", ("latitude",))[:] = [
                    60.0,
                    0.0,
                    -60.0,
                ]
                dataset.createVariable("longitude", "f4", ("longitude",))[:] = [
                    0.0,
                    90.0,
                    180.0,
                    270.0,
                ]
                shape = (2, 5, 3, 4)
                level_temperature = np.asarray([210, 230, 250, 270, 288], dtype=np.float32)
                temperature = np.broadcast_to(
                    level_temperature[None, :, None, None], shape
                ).copy()
                for time_index in range(2):
                    temperature[time_index] += time_index * 3.0
                    temperature[time_index] += np.linspace(-8, 8, 3)[None, :, None]
                humidity = np.broadcast_to(
                    np.asarray([1e-6, 1e-5, 1e-4, 2e-3, 8e-3], dtype=np.float32)[
                        None, :, None, None
                    ],
                    shape,
                ).copy()
                ozone = np.broadcast_to(
                    np.asarray([8e-6, 4e-6, 1e-6, 2e-7, 8e-8], dtype=np.float32)[
                        None, :, None, None
                    ],
                    shape,
                ).copy()
                dataset.createVariable("temperature_k", "f4", ("time", "level", "latitude", "longitude"))[:] = temperature
                dataset.createVariable("specific_humidity_kg_kg", "f4", ("time", "level", "latitude", "longitude"))[:] = humidity
                dataset.createVariable("ozone_mass_mixing_ratio_kg_kg", "f4", ("time", "level", "latitude", "longitude"))[:] = ozone
                dataset.createVariable("surface_pressure_pa", "f4", ("time", "latitude", "longitude"))[:] = 101325.0
                dataset.createVariable("surface_geopotential_height_m", "f4", ("time", "latitude", "longitude"))[:] = 0.0

            manager = AtmosphericPatternManager()
            summary = manager.fit_directory(
                source,
                cluster_count=3,
                latent_dimension=2,
                method="pca",
                maximum_samples=12,
            )
            self.assertEqual(summary["training_sample_count"], 12)
            self.assertEqual(summary["representative_count"], 3)
            self.assertFalse(summary["has_optical_library"])
            self.assertTrue(summary["has_representative_profiles"])
            self.assertTrue(summary["has_visualization_data"])
            self.assertEqual(summary["geolocated_sample_count"], 12)
            self.assertEqual(manager.training_scores.shape, (12, 2))
            self.assertEqual(int(np.sum(manager.cluster_counts)), 12)
            self.assertEqual(manager.cluster_profile_mean.shape[:2], (3, 35))
            self.assertGreater(manager.pca_explained_variance_ratio.size, 0)

            paths = manager.export_representative_profiles(root / "patterns")
            self.assertEqual(len(paths), 3)
            profile = LayeredAtmosphericProfileCsvReader.read(paths[0], 0)
            self.assertEqual(profile.temperature_k.size, 35)
            self.assertEqual(
                set(profile.gas_names), {"H2O", "CO2", "O3", "N2O", "CO", "CH4"}
            )
            self.assertFalse((root / "patterns" / "optical_depth_total.csv").exists())

            optical_paths: list[Path] = []
            wavenumber = np.asarray([700.0, 700.5, 701.0])
            for index in range(3):
                optical_path = root / f"representative_tau_{index + 1:03d}.csv"
                tau = np.full((35, wavenumber.size), 0.001 * (index + 1))
                np.savetxt(
                    optical_path,
                    np.column_stack([wavenumber, tau.T]),
                    delimiter=",",
                    header="wavenumber(cm-1),"
                    + ",".join(f"layer_{layer + 1:02d}" for layer in range(35)),
                    comments="",
                )
                optical_paths.append(optical_path)
            attached = manager.set_representative_optical_depth_files(optical_paths)
            self.assertTrue(attached["has_optical_library"])
            self.assertEqual(manager.representative_tau.dtype, np.float32)
            self.assertEqual(manager.representative_tau.shape, (3, 35, 3))
            self.assertEqual(
                [Path(item["optical_depth_file"]) for item in manager.representative_metadata],
                optical_paths,
            )


if __name__ == "__main__":
    unittest.main()
