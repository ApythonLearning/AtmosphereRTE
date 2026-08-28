from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QSpinBox

from ui.main_window import AtmosphereMainWindow


class ProjectPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_project_round_trip_and_wheel_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            previous_recent = os.environ.get("ARTE_ATMOSPHERE_RECENT_FILE")
            os.environ["ARTE_ATMOSPHERE_RECENT_FILE"] = str(root / "recent.json")
            first = AtmosphereMainWindow()
            first._set_workspace(workspace, create=True, load_project=False)
            first.latitude.setValue(35.12345)
            first.longitude.setValue(139.54321)
            first.solar_zenith.setValue(155.6405)
            first.solar_azimuth.setValue(72.3456)
            first.satellite_zenith.setValue(40.3903)
            first.satellite_azimuth.setValue(83.4622)
            first.visibility.setValue(42.0)
            first.enable_cloud.setChecked(False)
            first.spectral_mode.setCurrentIndex(1)
            first.max_spectral_points.setValue(1234)
            first.latest_sample = {"lat": 35.12345, "lon": 139.54321}
            first.latest_parameters = {"visibility_km": 42.0}
            first.latest_spectrum = {
                "wavelength_um": np.asarray([8.0, 10.0, 12.0]),
                "wavenumber_cm": np.asarray([1250.0, 1000.0, 833.333]),
                "earth_total_spectral_irradiance": np.asarray([1.0, 2.0, 1.0]),
                "spectral_quantity": "radiance",
                "spectral_point_count": 3,
                "earth_thermal_irradiance": 1.5,
                "earth_reflected_irradiance": 0.5,
                "earth_total_irradiance": 2.0,
            }
            line_source = root / "external_hapi"
            line_source.mkdir()
            (line_source / "H2O.data").write_text("synthetic-line\n", encoding="utf-8")
            (line_source / "H2O.header").write_text(
                '{"table_name": "H2O", "number_of_rows": 1}', encoding="utf-8"
            )
            first.hapi_workbench.manager.import_line_tables([line_source])
            self.assertTrue(first._save_project(silent=True))

            second = AtmosphereMainWindow()
            self.assertEqual(second.workspace, workspace.resolve())
            self.assertTrue(second.hapi_workbench.embedded)
            self.assertTrue(second.validation_workbench.embedded)
            self.assertEqual(second.navigation.count(), 4)
            self.assertEqual(
                second.navigation.item(3).text(), "大气廓线模式学习"
            )
            self.assertEqual(second.pattern_workbench.project_dir, workspace.resolve())
            self.assertTrue(second.hapi_workbench.close_button.isHidden())
            self.assertTrue(second.validation_workbench.close_button.isHidden())
            self.assertAlmostEqual(second.latitude.value(), 35.12345, places=5)
            self.assertAlmostEqual(second.longitude.value(), 139.54321, places=5)
            self.assertAlmostEqual(second.solar_zenith.value(), 155.6405, places=4)
            self.assertAlmostEqual(second.solar_azimuth.value(), 72.3456, places=4)
            self.assertAlmostEqual(second.satellite_zenith.value(), 40.3903, places=4)
            self.assertAlmostEqual(second.satellite_azimuth.value(), 83.4622, places=4)
            self.assertEqual(second.visibility.value(), 42.0)
            self.assertFalse(second.enable_cloud.isChecked())
            self.assertEqual(second.spectral_mode.currentData(), "high_resolution")
            self.assertEqual(second.max_spectral_points.value(), 1234)
            self.assertIsNotNone(second.latest_spectrum)
            self.assertEqual(
                second.hapi_workbench.line_table_sources()["H2O"],
                str(line_source / "H2O.data"),
            )
            np.testing.assert_allclose(
                second.latest_spectrum["earth_total_spectral_irradiance"],
                [1.0, 2.0, 1.0],
            )

            spin = QSpinBox()
            wheel_event = QEvent(QEvent.Type.Wheel)
            self.assertTrue(second.eventFilter(spin, wheel_event))
            self.assertTrue(wheel_event.isAccepted())
            first.close()
            second.close()
            if previous_recent is None:
                os.environ.pop("ARTE_ATMOSPHERE_RECENT_FILE", None)
            else:
                os.environ["ARTE_ATMOSPHERE_RECENT_FILE"] = previous_recent


if __name__ == "__main__":
    unittest.main()
