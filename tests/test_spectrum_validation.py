from __future__ import annotations

import json
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
            np.asarray([0.23, 0.54, 0.23]),
            mode="valid",
        )
        np.testing.assert_allclose(self.manager.three_point_hamming(values), expected)
        result = self.manager.compare(
            {"wavelength_um": np.arange(5.0), "values": values},
            {"wavelength_um": np.arange(5.0), "values": values},
        )
        np.testing.assert_allclose(result["simulated"], expected)
        np.testing.assert_allclose(result["observed"], expected)

    def test_hamming_does_not_mix_across_cris_band_gaps(self) -> None:
        values = np.asarray([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
        wavenumber = np.asarray([650.0, 650.625, 651.25, 1210.0, 1210.625, 1211.25])
        filtered = self.manager.three_point_hamming(values, wavenumber)
        np.testing.assert_allclose(filtered, values)

    def test_cris_sinc_resample_preserves_constant_radiance(self) -> None:
        source_wavenumber = np.arange(640.0, 760.0, 0.05)
        target_wavenumber = np.arange(648.75, 752.5, 0.625)
        result = self.manager.cris_sinc_resample(
            source_wavenumber,
            np.full(source_wavenumber.shape, 4.25),
            target_wavenumber,
        )
        np.testing.assert_allclose(result, 4.25, atol=1.0e-12)

    def test_compare_uses_cris_sinc_for_oversampled_simulation(self) -> None:
        simulated_wavenumber = np.arange(640.0, 760.0, 0.05)
        observed_wavenumber = np.arange(648.75, 752.5, 0.625)
        constant = 3.5
        simulated = {
            "wavelength_um": np.sort(1.0e4 / simulated_wavenumber),
            "values": np.full(simulated_wavenumber.shape, constant),
        }
        observed = {
            "wavelength_um": np.sort(1.0e4 / observed_wavenumber),
            "values": np.full(observed_wavenumber.shape, constant),
            "instrument": "CrIS",
        }
        result = self.manager.compare(simulated, observed)
        self.assertIn("CrIS sinc ILS", result["spectral_processing"])
        self.assertAlmostEqual(result["metrics"]["rmse"], 0.0, places=12)

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

    def test_reads_same_stem_observation_geometry_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spectrum_path = Path(directory) / "cris_validation.npz"
            np.savez(
                spectrum_path,
                wavenumber_cm=np.asarray([900.0, 1000.0]),
                radiance_w_m2_sr_um=np.asarray([1.0, 2.0]),
            )
            metadata = {
                "satellite_zenith_deg": 40.390335,
                "satellite_azimuth_deg": 83.462189,
            }
            spectrum_path.with_suffix(".json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            self.assertEqual(
                self.manager.read_sidecar_metadata(spectrum_path), metadata
            )

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

    def test_effective_cloud_fraction_is_fitted_from_clear_and_overcast_endmembers(self) -> None:
        wavelength = np.linspace(8.0, 12.0, 41)
        clear = 5.0 + 0.2 * np.sin(wavelength)
        overcast = 2.0 + 0.1 * np.cos(2.0 * wavelength)
        expected_fraction = 0.37
        observed_values = clear + expected_fraction * (overcast - clear)
        atmospheric = {
            "wavelength_um": wavelength,
            "earth_total_spectral_irradiance": clear,
            "cloud_clear_total_spectral_irradiance": clear,
            "cloud_overcast_total_spectral_irradiance": overcast,
            "representative_cloud_fraction": 0.9933,
            "spectral_quantity": "radiance",
        }
        observed = {
            "wavelength_um": wavelength,
            "values": observed_values,
        }
        fitted = self.manager.fit_effective_cloud_fraction(atmospheric, observed)
        self.assertAlmostEqual(
            fitted["effective_cloud_fraction"], expected_fraction, places=12
        )
        self.assertAlmostEqual(fitted["environment_grid_fraction"], 0.9933)
        self.assertLess(
            fitted["rmse_fitted"], fitted["rmse_environment_grid_fraction"]
        )
        np.testing.assert_allclose(
            fitted["simulated_spectrum"]["values"], observed_values
        )

    def test_cloud_fit_uses_windows_and_nucaps_as_weak_prior(self) -> None:
        wavelength = np.linspace(4.0, 15.4, 229)
        clear = 8.0 + 0.05 * np.sin(wavelength)
        overcast = clear - 4.0
        true_fraction = 0.20
        observed_values = clear + true_fraction * (overcast - clear)
        # 在云量拟合窗口外模拟严重的气体吸收带模型误差；这些误差不应
        # 被有效云量吸收。
        outside_windows = ~(
            ((wavelength >= 8.0) & (wavelength <= 9.2))
            | ((wavelength >= 10.2) & (wavelength <= 12.5))
        )
        observed_values[outside_windows] += 3.0
        prior_fraction = 0.60
        atmospheric = {
            "wavelength_um": wavelength,
            "earth_total_spectral_irradiance": clear,
            "cloud_clear_total_spectral_irradiance": clear,
            "cloud_overcast_total_spectral_irradiance": overcast,
            "representative_cloud_fraction": prior_fraction,
            "nucaps_same_footprint_cloud": {"fraction": prior_fraction},
            "spectral_quantity": "radiance",
        }
        observed = {"wavelength_um": wavelength, "values": observed_values}
        fitted = self.manager.fit_effective_cloud_fraction(atmospheric, observed)
        expected_regularized = (
            true_fraction + 0.05 * prior_fraction
        ) / 1.05
        self.assertAlmostEqual(
            fitted["unconstrained_cloud_fraction"], true_fraction, places=12
        )
        self.assertAlmostEqual(
            fitted["effective_cloud_fraction"], expected_regularized, places=12
        )
        self.assertEqual(fitted["prior_cloud_fraction"], prior_fraction)
        self.assertEqual(fitted["fit_windows_um"], [[8.0, 9.2], [10.2, 12.5]])

    def test_headerless_table_preserves_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "satellite.dat"
            path.write_text("8.0 1.0\n9.0 2.0\n10.0 3.0\n", encoding="utf-8")
            series = self.manager.read_numeric_series(path)
        self.assertEqual(list(series), ["0", "1"])
        np.testing.assert_allclose(series["0"], [8.0, 9.0, 10.0])


if __name__ == "__main__":
    unittest.main()
