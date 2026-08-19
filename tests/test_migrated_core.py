from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.atmospheric_radiation_manager import (
    LayeredAtmosphereSolver,
    build_specified_location_sample,
)
from core.spectrum_validation_manager import SpectrumValidationManager


class MigratedCoreTests(unittest.TestCase):
    def test_specified_location_validation(self) -> None:
        sample = build_specified_location_sample({
            "specified_latitude_deg": 35.0,
            "specified_longitude_deg": 139.0,
            "specified_altitude": 700.0,
            "specified_solar_right_ascension_deg": 10.0,
            "specified_solar_declination_deg": -5.0,
        })
        self.assertEqual(sample["lat"], 35.0)
        self.assertEqual(sample["lon"], 139.0)

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

