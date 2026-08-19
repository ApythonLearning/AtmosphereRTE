from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.spectrum_validation_manager import SpectrumValidationManager  # noqa: E402


class SpectrumValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SpectrumValidationManager()

    def test_compare_recovers_least_squares_scale(self) -> None:
        wavelength = np.linspace(8.0, 12.0, 21)
        shape = np.exp(-np.square(wavelength - 10.0))
        simulated = {"wavelength_um": wavelength, "values": shape}
        observed = {"wavelength_um": wavelength, "values": 2.5 * shape}
        result = self.manager.compare(simulated, observed, "least_squares")
        self.assertAlmostEqual(result["metrics"]["scale_factor"], 2.5)
        self.assertAlmostEqual(result["metrics"]["rmse"], 0.0, places=12)
        self.assertAlmostEqual(result["metrics"]["correlation"], 1.0)

    def test_three_point_hamming_is_applied_to_both_spectra(self) -> None:
        values = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
        expected = np.convolve(
            np.pad(values, (1, 1), mode="edge"),
            np.hamming(3) / np.sum(np.hamming(3)),
            mode="valid",
        )
        np.testing.assert_allclose(self.manager.three_point_hamming(values), expected)
        result = self.manager.compare(
            {"wavelength_um": np.arange(5.0), "values": values},
            {"wavelength_um": np.arange(5.0), "values": values},
        )
        np.testing.assert_allclose(result["simulated"], expected)
        np.testing.assert_allclose(result["observed"], expected)

    def test_per_wavenumber_corrections_protect_co2_center_and_longwave_tail(self) -> None:
        wavelength = np.asarray([14.7, 14.8, 15.0, 15.1, 15.2, 15.35, 15.4])
        factors = np.asarray([2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5])
        selected = (wavelength < 14.8) | ((wavelength > 15.1) & (wavelength <= 15.35))
        corrections = self.manager.build_per_wavenumber_corrections(
            wavelength, factors, selected
        )
        self.assertTrue(corrections)
        self.assertLessEqual(corrections[0]["wavelength_max_um"], 14.8)
        self.assertGreaterEqual(corrections[-1]["wavelength_min_um"], 15.1)
        self.assertLessEqual(corrections[-1]["wavelength_max_um"], 15.35)
        self.assertFalse(any(
            item["wavelength_min_um"] < 15.1
            and item["wavelength_max_um"] > 14.8
            for item in corrections
        ))

    def test_wavenumber_is_converted_and_sorted(self) -> None:
        spectrum = self.manager.build_spectrum(
            {"wn": np.array([1250.0, 1000.0, 833.333333]), "radiance": np.array([1.0, 2.0, 3.0])},
            "wn",
            "radiance",
            "wavenumber_cm",
        )
        np.testing.assert_allclose(spectrum["wavelength_um"], [8.0, 10.0, 12.0], rtol=1e-7)
        np.testing.assert_allclose(spectrum["values"], [1.0, 2.0, 3.0])

    def test_validation_uses_detector_received_atmospheric_spectrum(self) -> None:
        wavelength = np.array([8.0, 9.0, 10.0])
        atmospheric = np.array([1.0, 2.0, 3.0])
        spectrum = self.manager.from_atmospheric_detector_result({
            "wavelength_um": wavelength,
            "earth_total_spectral_irradiance": atmospheric,
            "detector_received_spectral_irradiance": np.array([90.0, 90.0, 90.0]),
            "spectral_quantity": "radiance",
        })
        np.testing.assert_allclose(spectrum["values"], atmospheric)
        self.assertEqual(spectrum["spectral_quantity"], "radiance")
        self.assertIn("sr⁻¹", spectrum["value_unit"])

    def test_headerless_table_preserves_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "satellite.dat"
            path.write_text("8.0 1.0\n9.0 2.0\n10.0 3.0\n", encoding="utf-8")
            series = self.manager.read_numeric_series(path)
        self.assertEqual(list(series), ["0", "1"])
        np.testing.assert_allclose(series["0"], [8.0, 9.0, 10.0])


if __name__ == "__main__":
    unittest.main()
