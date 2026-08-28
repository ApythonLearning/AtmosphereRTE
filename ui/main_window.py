from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core.atmospheric_radiation_manager import (
    EarthIrradianceManager,
    Merra2AerosolManager,
    ModisDataManager,
    build_specified_location_sample,
)
from core.spectrum_validation_manager import SpectrumValidationManager
from dialogs.absorption_preview_dialog import AbsorptionPreviewDialog
from dialogs.atmospheric_pattern_dialog import AtmosphericPatternDialog
from dialogs.atmospheric_spectrum_preview_dialog import AtmosphericSpectrumPreviewDialog
from dialogs.hapi_optical_depth_dialog import HapiOpticalDepthDialog
from dialogs.modis_preview_dialog import ModisPreviewDialog
from dialogs.optical_depth_correction_dialog import OpticalDepthCorrectionDialog
from dialogs.spectrum_validation_dialog import SpectrumValidationDialog


class FileField(QWidget):
    def __init__(self, title: str, file_filter: str, directory: bool = False) -> None:
        super().__init__()
        self.title = title
        self.file_filter = file_filter
        self.directory = directory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        button = QPushButton("浏览…")
        button.clicked.connect(self.browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:  # noqa: N802 - Qt naming convention
        self.edit.setText(value)

    def browse(self) -> None:
        current = self.text()
        if self.directory:
            value = QFileDialog.getExistingDirectory(self, self.title, current)
        else:
            value, _ = QFileDialog.getOpenFileName(
                self, self.title, current, self.file_filter
            )
        if value:
            self.setText(value)


def double_spin(
    value: float, minimum: float, maximum: float, decimals: int = 3, suffix: str = ""
) -> QDoubleSpinBox:
    control = QDoubleSpinBox()
    control.setRange(minimum, maximum)
    control.setDecimals(decimals)
    control.setValue(value)
    control.setSuffix(suffix)
    control.setSingleStep(0.1 if decimals else 1.0)
    return control


def scroll_page(content: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(content)
    return area


class AtmosphereMainWindow(QMainWindow):
    """Focused workbench for Earth environment and atmospheric RTE."""

    PROJECT_FILENAME = "arte_atmosphere_project.json"
    RECENT_FILENAME = ".arte_atmosphere_recent.json"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ARTE Atmosphere — 地气系统与大气辐射传输")
        self.resize(1320, 860)
        self.setMinimumSize(1040, 700)

        self.modis = ModisDataManager()
        self.merra2 = Merra2AerosolManager()
        self.radiation = EarthIrradianceManager(self.modis, self.merra2)
        self.latest_spectrum: dict[str, Any] | None = None
        self.latest_sample: dict[str, Any] | None = None
        self.latest_parameters: dict[str, Any] | None = None
        self.effective_cloud_fraction: float | None = None
        self.cloud_fraction_source = ""
        self.effective_cloud_location: tuple[float, float] | None = None
        self.nucaps_cloud_file = ""
        self.nucaps_cloud_for_index = 0
        self.corrections: list[dict[str, float]] = []
        saved_workspace = self._read_recent_workspace()
        self.workspace = (
            Path(saved_workspace)
            if saved_workspace
            else Path.cwd() / "arte_atmosphere_workspace"
        )
        self.project_file = self.workspace / self.PROJECT_FILENAME

        self._build_ui()
        self._connect_navigation()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        restored = self._set_workspace(self.workspace, create=False, load_project=True)
        if not restored:
            self._log("应用已就绪。请先读取 MODIS 产品或加载环境缓存。")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("ARTE Atmosphere")
        title.setObjectName("appTitle")
        subtitle = QLabel("地气系统建模 · 35层大气辐射传输 · 光谱验证")
        subtitle.setObjectName("secondaryText")
        self.workspace_label = QLabel()
        self.workspace_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        workspace_button = QPushButton("设置工作目录…")
        workspace_button.clicked.connect(self._choose_workspace)
        open_project_button = QPushButton("打开项目…")
        open_project_button.clicked.connect(self._open_project)
        save_project_button = QPushButton("保存项目")
        save_project_button.setObjectName("primaryButton")
        save_project_button.clicked.connect(self._save_project)
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch(1)
        header.addWidget(self.workspace_label)
        header.addWidget(workspace_button)
        header.addWidget(open_project_button)
        header.addWidget(save_project_button)
        root_layout.addLayout(header)

        body = QHBoxLayout()
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.addItems([
            "地球环境与大气辐射",
            "分层吸收光学厚度",
            "大气光谱验证",
            "大气廓线模式学习",
        ])
        self.navigation.setFixedWidth(220)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_atmosphere_page())
        self.pages.addWidget(self._build_hapi_page())
        self.pages.addWidget(self._build_validation_page())
        self.pages.addWidget(self._build_pattern_page())
        body.addWidget(self.navigation)
        body.addWidget(self.pages, 1)
        root_layout.addLayout(body, 1)

        self.log = QPlainTextEdit()
        self.log.setObjectName("applicationLog")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setFixedHeight(120)
        root_layout.addWidget(self.log)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _build_atmosphere_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        intro = QLabel(
            "从 MODIS 地表/云产品与可选的 MERRA-2 气溶胶场建立地球环境，"
            "在指定经纬度和高度计算宽带或光谱辐射。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("secondaryText")
        layout.addWidget(intro)

        environment = QGroupBox("1. 地球环境数据")
        environment_form = QFormLayout(environment)
        self.land_temperature = FileField(
            "选择 MOD11 陆地温度产品", "MODIS/HDF (*.hdf *.h5 *.hdf5);;所有文件 (*)"
        )
        self.sea_temperature = FileField(
            "选择海表温度产品", "NetCDF/HDF (*.nc *.nc4 *.hdf *.h5);;所有文件 (*)"
        )
        self.cloud_product = FileField(
            "选择 MOD08 云产品", "MODIS/HDF (*.hdf *.h5 *.hdf5);;所有文件 (*)"
        )
        self.land_type = FileField(
            "选择 MCD12 地表类型产品", "MODIS/HDF (*.hdf *.h5 *.hdf5);;所有文件 (*)"
        )
        self.merra_product = FileField(
            "选择 MERRA-2 M2T1NXAER 产品", "NetCDF (*.nc4 *.nc *.h5);;所有文件 (*)"
        )
        self.environment_cache = FileField(
            "选择地球环境缓存", "环境缓存 (*.npz);;所有文件 (*)"
        )
        self.resolution = double_spin(2.0, 0.5, 10.0, 1, "°")
        environment_form.addRow("MOD11 陆地温度", self.land_temperature)
        environment_form.addRow("海表温度", self.sea_temperature)
        environment_form.addRow("MOD08 云产品", self.cloud_product)
        environment_form.addRow("MCD12 地表类型", self.land_type)
        environment_form.addRow("MERRA-2 气溶胶（可选）", self.merra_product)
        environment_form.addRow("环境缓存", self.environment_cache)
        environment_form.addRow("网格分辨率", self.resolution)
        environment_buttons = QHBoxLayout()
        for text, slot in (
            ("读取产品并生成缓存", self._load_environment_products),
            ("加载缓存", self._load_environment_cache),
            ("预览环境场", self._preview_environment),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            environment_buttons.addWidget(button)
        environment_buttons.addStretch(1)
        environment_form.addRow("", environment_buttons)
        layout.addWidget(environment)

        absorption = QGroupBox("2. 分层气体吸收")
        absorption_form = QFormLayout(absorption)
        self.absorption_file = FileField(
            "选择35层总光学厚度", "光学厚度 CSV (*.csv);;所有文件 (*)"
        )
        absorption_form.addRow("总气体分子光学厚度", self.absorption_file)
        absorption_buttons = QHBoxLayout()
        load_absorption = QPushButton("导入")
        load_absorption.clicked.connect(self._load_absorption)
        preview_absorption = QPushButton("查看分层光学厚度")
        preview_absorption.clicked.connect(self._preview_absorption)
        correction_button = QPushButton("配置波段修正")
        correction_button.clicked.connect(self._configure_corrections)
        absorption_buttons.addWidget(load_absorption)
        absorption_buttons.addWidget(preview_absorption)
        absorption_buttons.addWidget(correction_button)
        absorption_buttons.addStretch(1)
        absorption_form.addRow("", absorption_buttons)
        layout.addWidget(absorption)

        scenario = QGroupBox("3. 指定位置辐射传输")
        form = QFormLayout(scenario)
        self.time_s = double_spin(0.0, 0.0, 1.0e9, 3, " s")
        self.latitude = double_spin(0.0, -90.0, 90.0, 5, "°")
        self.longitude = double_spin(0.0, -180.0, 180.0, 5, "°")
        self.altitude = double_spin(700.0, 0.001, 1.0e6, 3, " km")
        self.solar_zenith = double_spin(0.0, 0.0, 180.0, 4, "°")
        self.solar_zenith.setToolTip(
            "目标位置的局地太阳天顶角：0°为太阳位于天顶，90°为地平线，"
            "大于90°为夜间。"
        )
        self.solar_azimuth = double_spin(180.0, 0.0, 360.0, 4, "°")
        self.solar_azimuth.setToolTip(
            "目标位置的局地太阳方位角：从正北起顺时针计量，"
            "北=0°、东=90°、南=180°、西=270°。"
        )
        self.satellite_zenith = double_spin(0.0, 0.0, 89.999, 4, "°")
        self.satellite_zenith.setToolTip(
            "足迹处卫星观测方向与局地天顶的夹角。0°为垂直观测；"
            "高分辨率单柱模式按 tau/cos(卫星天顶角) 计算斜程。"
        )
        self.satellite_azimuth = double_spin(0.0, 0.0, 360.0, 4, "°")
        self.satellite_azimuth.setToolTip(
            "足迹指向卫星的局地方位角，从正北起顺时针计量；"
            "用于计算太阳—观测散射夹角。"
        )
        self.visibility = double_spin(23.0, 1.0, 200.0, 1, " km")
        self.temperature_offset = double_spin(0.0, -15.0, 15.0, 1, " K")
        self.enable_cloud = QCheckBox("启用液态水云模型")
        self.enable_cloud.setChecked(True)
        self.enable_cloud.setToolTip(
            "高分辨率验证读取同足迹NUCAPS云量作为弱先验，并由"
            "CrIS/卫星观测谱的大气窗口晴空—全云端元拟合最终有效云量；"
            "不直接固定采用NUCAPS或全球云图网格值。"
        )
        self.enable_cloud.toggled.connect(self._on_cloud_model_toggled)
        self.enable_scattering = QCheckBox("启用大气散射")
        self.enable_scattering.setChecked(True)
        toggles = QHBoxLayout()
        toggles.addWidget(self.enable_cloud)
        toggles.addWidget(self.enable_scattering)
        toggles.addStretch(1)
        self.spectral_mode = QComboBox()
        self.spectral_mode.addItem("快速地球盘光谱（交互预览）", "fast")
        self.spectral_mode.addItem("高分辨率目标单柱（需完整光学厚度）", "high_resolution")
        self.max_spectral_points = QSpinBox()
        self.max_spectral_points.setRange(100, 5000)
        self.max_spectral_points.setValue(600)
        form.addRow("计算时刻", self.time_s)
        form.addRow("纬度", self.latitude)
        form.addRow("经度", self.longitude)
        form.addRow("观测高度", self.altitude)
        form.addRow("太阳天顶角", self.solar_zenith)
        form.addRow("太阳方位角", self.solar_azimuth)
        form.addRow("卫星天顶角", self.satellite_zenith)
        form.addRow("卫星方位角", self.satellite_azimuth)
        form.addRow("能见度", self.visibility)
        form.addRow("10 km以上温度偏移", self.temperature_offset)
        form.addRow("物理过程", toggles)
        form.addRow("光谱模式", self.spectral_mode)
        form.addRow("快速模式最大光谱点", self.max_spectral_points)
        solve_buttons = QHBoxLayout()
        solve = QPushButton("执行大气辐射传输求解")
        solve.setObjectName("primaryButton")
        solve.clicked.connect(self._compute_spectrum)
        export = QPushButton("导出最近结果")
        export.clicked.connect(self._export_spectrum)
        solve_buttons.addWidget(solve)
        solve_buttons.addWidget(export)
        solve_buttons.addStretch(1)
        form.addRow("", solve_buttons)
        self.result_summary = QLabel("尚无计算结果")
        self.result_summary.setWordWrap(True)
        self.result_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("最近结果", self.result_summary)
        layout.addWidget(scenario)
        layout.addStretch(1)
        return scroll_page(page)

    def _build_hapi_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.hapi_workbench = HapiOpticalDepthDialog(
            self.workspace, page, embedded=True
        )
        self.hapi_workbench.resultReady.connect(self._handle_hapi_result)
        self.hapi_workbench.batchResultReady.connect(self._handle_hapi_batch_result)
        layout.addWidget(self.hapi_workbench)
        return scroll_page(page)

    def _build_validation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.validation_workbench = SpectrumValidationDialog(
            {},
            page,
            correction_handler=self._configure_corrections,
            automatic_correction_handler=self._automatically_correct_optical_depth,
            cloud_fit_handler=self._apply_effective_cloud_fit,
            observation_geometry_handler=self._apply_validation_geometry,
            embedded=True,
        )
        layout.addWidget(self.validation_workbench)
        return page

    def _build_pattern_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.pattern_workbench = AtmosphericPatternDialog(self.workspace, page)
        self.pattern_workbench.opticalDepthReady.connect(
            self._handle_pattern_optical_depth
        )
        self.pattern_workbench.statusMessage.connect(self._log)
        self.pattern_workbench.exactCalculationRequested.connect(self._open_hapi)
        self.pattern_workbench.exactProfileRequested.connect(self._open_hapi_profile)
        self.pattern_workbench.batchProfilesRequested.connect(self._open_hapi_batch)
        layout.addWidget(self.pattern_workbench)
        return scroll_page(page)

    def _connect_navigation(self) -> None:
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)

    def _choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择工作目录", str(self.workspace)
        )
        if path:
            if self.workspace.exists():
                self._save_project(silent=True)
            self._set_workspace(Path(path), create=True, load_project=True)

    def _set_workspace(
        self, path: Path, create: bool, load_project: bool = False
    ) -> bool:
        self.workspace = path.expanduser().resolve()
        self.project_file = self.workspace / self.PROJECT_FILENAME
        if create:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self._write_recent_workspace()
        self.workspace_label.setText(f"工作目录：{self.workspace}")
        self.workspace_label.setToolTip(str(self.workspace))
        if hasattr(self, "hapi_workbench"):
            self.hapi_workbench.set_project_dir(self.workspace)
            if load_project:
                self.hapi_workbench.set_line_table_sources({})
        if hasattr(self, "pattern_workbench"):
            self.pattern_workbench.set_project_dir(self.workspace)
        if load_project:
            return self._load_project(
                self.workspace / self.PROJECT_FILENAME, silent=True
            )
        return False

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 ARTE Atmosphere 项目",
            str(self.workspace),
            "ARTE Atmosphere Project (*.json);;所有文件 (*)",
        )
        if not path:
            return
        if self.workspace.exists():
            self._save_project(silent=True)
        project_path = Path(path).expanduser().resolve()
        self._set_workspace(project_path.parent, create=True, load_project=False)
        self.project_file = project_path
        self._load_project(project_path, silent=False)

    def _save_project(self, checked: bool = False, silent: bool = False) -> bool:
        del checked
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            latest_path = self._save_latest_spectrum()
            data = {
                "schema_version": 1,
                "application": "ARTE Atmosphere",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "workspace": str(self.workspace),
                "navigation_index": self.navigation.currentRow(),
                "environment": {
                    "land_temperature_file": self.land_temperature.text(),
                    "sea_temperature_file": self.sea_temperature.text(),
                    "cloud_file": self.cloud_product.text(),
                    "land_type_file": self.land_type.text(),
                    "merra2_aerosol_file": self.merra_product.text(),
                    "environment_cache": self.environment_cache.text(),
                    "resolution_deg": self.resolution.value(),
                },
                "atmosphere": self._scenario_parameters(),
                "spectral_mode": str(self.spectral_mode.currentData()),
                "maximum_spectral_points": self.max_spectral_points.value(),
                "hapi": {
                    "external_line_table_sources": self.hapi_workbench.line_table_sources(),
                },
                "atmospheric_patterns": self.pattern_workbench.project_state(),
                "optical_depth_corrections": self.corrections,
                "latest_sample": self.latest_sample,
                "latest_parameters": self.latest_parameters,
                "latest_spectrum_file": latest_path,
            }
            project_path = self.project_file
            temporary_path = project_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                    default=self._json_default,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(project_path)
            self._write_recent_workspace()
            if not silent:
                self._log(f"项目已保存：{project_path}")
                self.statusBar().showMessage("项目已保存", 4000)
            return True
        except Exception as exc:  # noqa: BLE001
            if not silent:
                self._show_error("保存项目失败", exc)
            else:
                self._log(f"自动保存项目失败：{exc}")
            return False

    def _load_project(self, project_path: Path, silent: bool = False) -> bool:
        if not project_path.exists():
            return False
        try:
            self.project_file = project_path.resolve()
            data = json.loads(project_path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) != 1:
                raise ValueError("不支持的项目文件版本。")
            environment = dict(data.get("environment", {}))
            atmosphere = dict(data.get("atmosphere", {}))
            hapi_settings = dict(data.get("hapi", {}))
            self.hapi_workbench.set_line_table_sources(
                dict(hapi_settings.get("external_line_table_sources", {}))
            )
            pattern_settings = dict(data.get("atmospheric_patterns", {}))

            self.land_temperature.setText(str(environment.get("land_temperature_file", "")))
            self.sea_temperature.setText(str(environment.get("sea_temperature_file", "")))
            self.cloud_product.setText(str(environment.get("cloud_file", "")))
            self.land_type.setText(str(environment.get("land_type_file", "")))
            self.merra_product.setText(str(environment.get("merra2_aerosol_file", "")))
            self.environment_cache.setText(str(environment.get("environment_cache", "")))
            self.resolution.setValue(float(environment.get("resolution_deg", 2.0)))
            self.absorption_file.setText(str(atmosphere.get("absorption_file", "")))
            self.time_s.setValue(float(atmosphere.get("specified_time_s", 0.0)))
            self.latitude.setValue(float(atmosphere.get("specified_latitude_deg", 0.0)))
            self.longitude.setValue(float(atmosphere.get("specified_longitude_deg", 0.0)))
            self.altitude.setValue(float(atmosphere.get("specified_altitude", 700.0)))
            if "specified_solar_zenith_deg" in atmosphere:
                solar_zenith = float(atmosphere["specified_solar_zenith_deg"])
                solar_azimuth = float(
                    atmosphere.get("specified_solar_azimuth_deg", 180.0)
                )
            else:
                migrated_solar = build_specified_location_sample(atmosphere)
                solar_zenith = migrated_solar["solar_zenith"]
                solar_azimuth = migrated_solar["solar_azimuth"]
            self.solar_zenith.setValue(solar_zenith)
            self.solar_azimuth.setValue(solar_azimuth)
            self.satellite_zenith.setValue(float(
                atmosphere.get("specified_satellite_zenith_deg", 0.0)
            ))
            self.satellite_azimuth.setValue(float(
                atmosphere.get("specified_satellite_azimuth_deg", 0.0)
            ))
            self.visibility.setValue(float(atmosphere.get("visibility_km", 23.0)))
            self.temperature_offset.setValue(
                float(atmosphere.get("upper_atmosphere_temperature_offset_k", 0.0))
            )
            self.enable_cloud.setChecked(bool(atmosphere.get("enable_cloud", True)))
            fitted_cloud = atmosphere.get("effective_cloud_fraction")
            self.effective_cloud_fraction = (
                None if fitted_cloud in (None, "") else float(fitted_cloud)
            )
            self.cloud_fraction_source = str(
                atmosphere.get("cloud_fraction_source", "")
            )
            fitted_location = atmosphere.get("effective_cloud_location")
            self.effective_cloud_location = (
                (float(fitted_location[0]), float(fitted_location[1]))
                if isinstance(fitted_location, (list, tuple))
                and len(fitted_location) == 2
                else None
            )
            self.nucaps_cloud_file = str(
                atmosphere.get("nucaps_cloud_file", "")
            )
            self.nucaps_cloud_for_index = int(
                atmosphere.get("nucaps_cloud_for_index", 0)
            )
            self.enable_scattering.setChecked(
                bool(atmosphere.get("enable_scattering", True))
            )
            spectral_mode = str(data.get("spectral_mode", "fast"))
            spectral_index = self.spectral_mode.findData(spectral_mode)
            self.spectral_mode.setCurrentIndex(max(spectral_index, 0))
            self.max_spectral_points.setValue(
                int(data.get("maximum_spectral_points", 600))
            )
            self.corrections = self.radiation.atmosphere.normalize_optical_depth_corrections(
                data.get("optical_depth_corrections", [])
            )
            self.navigation.setCurrentRow(
                max(0, min(int(data.get("navigation_index", 0)), self.pages.count() - 1))
            )
            self.latest_sample = self._dictionary_or_none(data.get("latest_sample"))
            self.latest_parameters = self._dictionary_or_none(
                data.get("latest_parameters")
            )

            self._reset_calculation_models()
            restored: list[str] = []
            external_table_count = len(self.hapi_workbench.line_table_sources())
            if external_table_count:
                restored.append(f"HAPI外部谱线 {external_table_count} 组")
            if pattern_settings and self.pattern_workbench.restore_project_state(
                pattern_settings
            ):
                restored.append("大气状态模式模型")
            cache_path = self._resolve_project_path(self.environment_cache.text())
            if cache_path is not None and cache_path.exists():
                try:
                    grid = self.modis.load_cache(cache_path)
                except ValueError as exc:
                    if "旧版MODIS温度处理" not in str(exc):
                        raise
                    source_paths = [
                        self._resolve_project_path(self.land_temperature.text()),
                        self._resolve_project_path(self.sea_temperature.text()),
                        self._resolve_project_path(self.cloud_product.text()),
                        self._resolve_project_path(self.land_type.text()),
                    ]
                    if not all(path is not None and path.exists() for path in source_paths):
                        raise ValueError(
                            "环境缓存已过期，且原始陆温、海温、云或地表类型产品不完整。"
                        ) from exc
                    grid = self.modis.load_products(
                        *(str(path) for path in source_paths if path is not None),
                        self.resolution.value(),
                    )
                    self.modis.save_cache(cache_path)
                    self._log(f"旧环境缓存已按新温度处理流程自动重建：{cache_path}")
                self.environment_cache.setText(str(cache_path))
                restored.append(
                    f"环境缓存 {grid.latitude.shape[0]}×{grid.latitude.shape[1]}"
                )
            merra_path = self._resolve_project_path(self.merra_product.text())
            if merra_path is not None and merra_path.exists():
                self.merra2.load_product(merra_path)
                self.merra_product.setText(str(merra_path))
                restored.append("MERRA-2")
            absorption_path = self._resolve_project_path(self.absorption_file.text())
            if absorption_path is not None and absorption_path.exists():
                self.radiation.atmosphere.load_absorption_optical_depth(absorption_path)
                self.absorption_file.setText(str(absorption_path))
                if self.corrections:
                    self.radiation.atmosphere.apply_optical_depth_corrections(
                        self.corrections
                    )
                self.radiation.atmosphere.set_upper_atmosphere_temperature_offset(
                    self.temperature_offset.value()
                )
                restored.append("35层光学厚度")
            latest_path = self._resolve_project_path(
                str(data.get("latest_spectrum_file", ""))
            )
            self.latest_spectrum = self._load_latest_spectrum(latest_path)
            if self.latest_spectrum:
                self._update_spectrum_summary(self.latest_spectrum)
                restored.append("最近光谱")
            else:
                self.result_summary.setText("尚无计算结果")
                self.validation_workbench.set_atmospheric_result(None)

            detail = "、".join(restored) if restored else "参数"
            self._log(f"项目已自动读取：{project_path}；已恢复{detail}。")
            if not silent:
                self.statusBar().showMessage("项目读取完成", 5000)
            return True
        except Exception as exc:  # noqa: BLE001
            if not silent:
                self._show_error("读取项目失败", exc)
            else:
                self._log(f"自动读取项目失败：{exc}")
            return False

    def _reset_calculation_models(self) -> None:
        self.modis = ModisDataManager()
        self.merra2 = Merra2AerosolManager()
        self.radiation = EarthIrradianceManager(self.modis, self.merra2)

    def _save_latest_spectrum(self) -> str:
        if not self.latest_spectrum:
            return ""
        result_dir = self.workspace / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        array_path = result_dir / "latest_spectrum.npz"
        metadata_path = result_dir / "latest_spectrum.json"
        arrays = {
            key: np.asarray(value)
            for key, value in self.latest_spectrum.items()
            if isinstance(value, np.ndarray)
        }
        metadata = {
            key: value
            for key, value in self.latest_spectrum.items()
            if not isinstance(value, np.ndarray)
        }
        np.savez_compressed(array_path, **arrays)
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            ),
            encoding="utf-8",
        )
        return array_path.relative_to(self.workspace).as_posix()

    def _load_latest_spectrum(self, path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            result: dict[str, Any] = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            result.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        return result or None

    def _update_spectrum_summary(self, result: dict[str, Any]) -> None:
        quantity = "辐亮度" if result.get("spectral_quantity") == "radiance" else "辐照度"
        summary = (
            f"{quantity}光谱 {int(result.get('spectral_point_count', len(result.get('wavelength_um', [])))):,} 点；"
            f"热辐射={float(result.get('earth_thermal_irradiance', 0.0)):.6g}，"
            f"反射太阳={float(result.get('earth_reflected_irradiance', 0.0)):.6g}，"
            f"总量={float(result.get('earth_total_irradiance', 0.0)):.6g}"
        )
        self.result_summary.setText(summary)
        self.validation_workbench.set_atmospheric_result(result)

    def _resolve_project_path(self, value: str) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()

    @classmethod
    def _recent_file(cls) -> Path:
        override = os.environ.get("ARTE_ATMOSPHERE_RECENT_FILE", "").strip()
        if override:
            return Path(override).expanduser().resolve()
        return Path(__file__).resolve().parents[1] / cls.RECENT_FILENAME

    @classmethod
    def _read_recent_workspace(cls) -> str:
        path = cls._recent_file()
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("last_workspace", "")).strip()
        except (OSError, ValueError, TypeError):
            return ""

    def _write_recent_workspace(self) -> None:
        path = self._recent_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"last_workspace": str(self.workspace)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _dictionary_or_none(value: Any) -> dict[str, Any] | None:
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"无法序列化项目参数：{type(value).__name__}")

    def _load_environment_products(self) -> None:
        paths = [
            self.land_temperature.text(),
            self.sea_temperature.text(),
            self.cloud_product.text(),
            self.land_type.text(),
        ]
        if not all(paths):
            QMessageBox.information(self, "环境数据", "请选择四类 MODIS/海温输入产品。")
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            grid = self.modis.load_products(*paths, self.resolution.value())
            if self.merra_product.text():
                self.merra2.load_product(self.merra_product.text())
            self.workspace.mkdir(parents=True, exist_ok=True)
            cache = self.workspace / "config" / "earth_environment" / "environment_cache.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            self.modis.save_cache(cache)
            self.environment_cache.setText(str(cache))
            self._log(f"地球环境读取完成：{grid.latitude.shape[0]}×{grid.latitude.shape[1]}，缓存={cache}")
            self.statusBar().showMessage("地球环境已加载", 5000)
            self._save_project(silent=True)
        except Exception as exc:  # noqa: BLE001
            self._show_error("读取环境数据失败", exc)
        finally:
            QApplication.restoreOverrideCursor()

    def _load_environment_cache(self) -> None:
        path = self.environment_cache.text()
        if not path:
            self.environment_cache.browse()
            path = self.environment_cache.text()
        if not path:
            return
        try:
            grid = self.modis.load_cache(path)
            if self.merra_product.text():
                self.merra2.load_product(self.merra_product.text())
            self.resolution.setValue(float(grid.metadata.get("resolution_deg", 2.0)))
            self._log(f"环境缓存已加载：{path}（{grid.latitude.shape[0]}×{grid.latitude.shape[1]}）")
            self._save_project(silent=True)
        except Exception as exc:  # noqa: BLE001
            self._show_error("加载环境缓存失败", exc)

    def _preview_environment(self) -> None:
        grid = self.modis.get_grid()
        if grid is None:
            QMessageBox.information(self, "环境预览", "请先读取环境产品或加载缓存。")
            return
        ModisPreviewDialog(grid, self.merra2.get_field(), 0, self).exec()

    def _load_absorption(self) -> None:
        path = self.absorption_file.text()
        if not path:
            self.absorption_file.browse()
            path = self.absorption_file.text()
        if not path:
            return
        try:
            self.radiation.atmosphere.load_absorption_optical_depth(path)
            self.corrections = []
            self._clear_effective_cloud_fit()
            self._log(f"35层总光学厚度已加载：{path}")
            self._save_project(silent=True)
        except Exception as exc:  # noqa: BLE001
            self._show_error("导入光学厚度失败", exc)

    def _preview_absorption(self) -> None:
        try:
            data = self.radiation.atmosphere.get_absorption_visualization_data()
            AbsorptionPreviewDialog(data, self).exec()
        except Exception as exc:  # noqa: BLE001
            self._show_error("查看光学厚度失败", exc)

    def _configure_corrections(self) -> bool:
        if self.radiation.atmosphere._absorption_tau_original is None:
            QMessageBox.information(self, "光学厚度修正", "请先导入总光学厚度 CSV。")
            return False
        dialog = OpticalDepthCorrectionDialog(self.corrections, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        try:
            self.corrections = dialog.corrections()
            self.radiation.atmosphere.apply_optical_depth_corrections(self.corrections)
            output = self.workspace / "config" / "atmosphere" / "optical_depth_total_corrected.csv"
            self.radiation.atmosphere.save_corrected_absorption_optical_depth(output)
            self._log(f"已应用 {len(self.corrections)} 个修正波段并保存：{output}")
            self._save_project(silent=True)
            return True
        except Exception as exc:  # noqa: BLE001
            self._show_error("应用光学厚度修正失败", exc)
            return False

    def _scenario_parameters(self) -> dict[str, Any]:
        parameters = {
            "specified_time_s": self.time_s.value(),
            "specified_latitude_deg": self.latitude.value(),
            "specified_longitude_deg": self.longitude.value(),
            "specified_altitude": self.altitude.value(),
            "specified_solar_zenith_deg": self.solar_zenith.value(),
            "specified_solar_azimuth_deg": self.solar_azimuth.value(),
            "specified_satellite_zenith_deg": self.satellite_zenith.value(),
            "specified_satellite_azimuth_deg": self.satellite_azimuth.value(),
            "visibility_km": self.visibility.value(),
            "enable_cloud": self.enable_cloud.isChecked(),
            "cloud_fraction_policy": "same_footprint_or_spectral_fit_v1",
            "enable_scattering": self.enable_scattering.isChecked(),
            "altitude_unit": "km",
            "mode": "fast",
            "absorption_file": self.absorption_file.text(),
            "enable_optical_depth_correction": bool(self.corrections),
            "optical_depth_corrections": self.corrections,
            "upper_atmosphere_temperature_offset_k": self.temperature_offset.value(),
            "use_merra2_aerosol": bool(self.merra_product.text()),
            "merra2_aerosol_file": self.merra_product.text(),
            "merra2_time_offset_hours": 0.0,
        }
        if self.nucaps_cloud_file:
            parameters["nucaps_cloud_file"] = self.nucaps_cloud_file
            parameters["nucaps_cloud_for_index"] = self.nucaps_cloud_for_index
        current_location = (self.latitude.value(), self.longitude.value())
        if (
            self.effective_cloud_fraction is not None
            and self.effective_cloud_location is not None
            and np.allclose(
                current_location, self.effective_cloud_location, rtol=0.0, atol=1.0e-8
            )
        ):
            parameters["effective_cloud_fraction"] = self.effective_cloud_fraction
            parameters["cloud_fraction_source"] = self.cloud_fraction_source
            parameters["effective_cloud_location"] = list(
                self.effective_cloud_location
            )
        return parameters

    def _clear_effective_cloud_fit(self) -> None:
        self.effective_cloud_fraction = None
        self.cloud_fraction_source = ""
        self.effective_cloud_location = None

    def _on_cloud_model_toggled(self, enabled: bool) -> None:
        """物理过程开关变化后废弃按旧云设置生成的光谱。"""
        self._clear_effective_cloud_fit()
        if not self.latest_spectrum:
            return
        previous = bool(self.latest_spectrum.get("cloud_model_enabled", False))
        if previous == bool(enabled):
            return
        self.latest_spectrum = None
        self.latest_sample = None
        self.latest_parameters = None
        if hasattr(self, "validation_workbench"):
            self.validation_workbench.set_atmospheric_result(None)
        if hasattr(self, "result_summary"):
            self.result_summary.setText("云模型开关已变化，请重新执行大气辐射传输求解。")

    def _apply_effective_cloud_fit(
        self, diagnostics: dict[str, Any]
    ) -> dict[str, Any] | None:
        """保存 CrIS/卫星光谱拟合的视场有效云量并更新当前总谱。"""
        if not self.latest_spectrum or self.latest_sample is None:
            return None
        fraction = float(diagnostics["effective_cloud_fraction"])
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("光谱拟合返回了无效云量。")
        self.effective_cloud_fraction = fraction
        self.cloud_fraction_source = str(
            diagnostics.get("source", "卫星同足迹光谱晴空/全云端元拟合")
        )
        self.effective_cloud_location = (
            float(self.latest_sample["lat"]),
            float(self.latest_sample["lon"]),
        )
        updated = dict(self.latest_spectrum)
        clear_wavelength = np.asarray(
            updated["cloud_clear_total_spectral_irradiance"], dtype=float
        )
        cloudy_wavelength = np.asarray(
            updated["cloud_overcast_total_spectral_irradiance"], dtype=float
        )
        clear_wavenumber = np.asarray(
            updated["cloud_clear_total_spectral_wavenumber"], dtype=float
        )
        cloudy_wavenumber = np.asarray(
            updated["cloud_overcast_total_spectral_wavenumber"], dtype=float
        )
        total_wavelength = clear_wavelength + fraction * (
            cloudy_wavelength - clear_wavelength
        )
        total_wavenumber = clear_wavenumber + fraction * (
            cloudy_wavenumber - clear_wavenumber
        )
        updated["earth_total_spectral_irradiance"] = total_wavelength
        updated["earth_total_spectral_wavenumber"] = total_wavenumber
        updated["earth_total_irradiance"] = float(np.trapezoid(
            total_wavenumber, np.asarray(updated["wavenumber_cm"], dtype=float)
        ))
        updated["representative_cloud_fraction"] = fraction
        updated["effective_cloud_fraction"] = fraction
        updated["cloud_fraction_source"] = self.cloud_fraction_source
        updated["cloud_fraction_fit_pending"] = False
        updated["effective_cloud_fit"] = {
            key: value
            for key, value in diagnostics.items()
            if key != "simulated_spectrum"
        }
        updated["cloud_component_breakdown_is_provisional"] = True
        self.latest_spectrum = updated
        parameters = dict(self.latest_parameters or self._scenario_parameters())
        parameters.update({
            "enable_cloud": True,
            "effective_cloud_fraction": fraction,
            "cloud_fraction_source": self.cloud_fraction_source,
            "effective_cloud_location": list(self.effective_cloud_location),
        })
        self.latest_parameters = parameters
        self.validation_workbench.set_atmospheric_result(updated)
        self._update_spectrum_summary(updated)
        prior_fraction = diagnostics.get("prior_cloud_fraction")
        prior_text = (
            f"{float(prior_fraction):.2%}"
            if prior_fraction is not None else "无"
        )
        self._log(
            "已使用卫星同足迹大气窗口拟合有效云量："
            f"{fraction:.2%}（NUCAPS先验={prior_text}，"
            "先验仅作弱约束）。"
        )
        self._save_project(silent=True)
        return dict(updated)

    def _apply_validation_geometry(self, metadata: dict[str, Any]) -> bool:
        """将卫星光谱侧车中的同足迹观测几何写入单柱求解参数。

        返回值表示当前仿真结果是否需要按新几何重新计算。验证工作台据此
        暂停绝对值比较，避免把旧的垂直路径结果与斜视卫星谱直接比较。
        """
        mapping = (
            ("latitude_deg", self.latitude),
            ("longitude_deg", self.longitude),
            ("satellite_height_km", self.altitude),
            ("solar_zenith_deg", self.solar_zenith),
            ("solar_azimuth_deg", self.solar_azimuth),
            ("satellite_zenith_deg", self.satellite_zenith),
            ("satellite_azimuth_deg", self.satellite_azimuth),
        )
        applied: dict[str, float] = {}
        for name, widget in mapping:
            if name not in metadata:
                continue
            value = float(metadata[name])
            if not np.isfinite(value):
                continue
            widget.setValue(value)
            applied[name] = value
        if not applied:
            return False

        high_resolution_index = self.spectral_mode.findData("high_resolution")
        if high_resolution_index >= 0:
            self.spectral_mode.setCurrentIndex(high_resolution_index)

        result_names = {
            "latitude_deg": "target_latitude_deg",
            "longitude_deg": "target_longitude_deg",
            "satellite_height_km": "target_altitude_m",
            "solar_zenith_deg": "solar_zenith_deg",
            "solar_azimuth_deg": "solar_azimuth_deg",
            "satellite_zenith_deg": "satellite_zenith_deg",
            "satellite_azimuth_deg": "satellite_azimuth_deg",
        }
        recalculation_required = not bool(self.latest_spectrum)
        if self.latest_spectrum:
            for metadata_name, requested in applied.items():
                current = self.latest_spectrum.get(result_names[metadata_name])
                if current is None:
                    recalculation_required = True
                    break
                requested_value = (
                    requested * 1000.0
                    if metadata_name == "satellite_height_km"
                    else requested
                )
                if not np.isclose(
                    float(current), requested_value, rtol=0.0, atol=1.0e-3
                ):
                    recalculation_required = True
                    break
        if recalculation_required:
            self._clear_effective_cloud_fit()
            self.latest_spectrum = None
            self.latest_sample = None
            self.latest_parameters = None
            self.validation_workbench.set_atmospheric_result(None)
            self.result_summary.setText(
                "卫星观测几何已更新，请重新执行高分辨率单柱求解。"
            )
        self._log(
            "已从卫星光谱元数据读取同足迹观测几何："
            f"卫星天顶角 {self.satellite_zenith.value():.3f}°，"
            f"卫星方位角 {self.satellite_azimuth.value():.3f}°。"
        )
        self._save_project(silent=True)
        return recalculation_required

    def _compute_spectrum(self) -> None:
        grid = self.modis.get_grid()
        if grid is None:
            QMessageBox.information(self, "大气辐射传输", "请先加载地球环境数据。")
            return
        parameters = self._scenario_parameters()
        try:
            sample = build_specified_location_sample(parameters)
        except Exception as exc:  # noqa: BLE001
            self._show_error("场景参数无效", exc)
            return
        mode = str(self.spectral_mode.currentData())
        progress = QProgressDialog("正在计算大气辐射传输光谱…", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        def update(step: int, total: int) -> bool:
            progress.setMaximum(max(total, 1))
            progress.setValue(step)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            self.latest_spectrum = self.radiation.compute_spectrum_at_position(
                grid,
                sample,
                parameters,
                progress_callback=update,
                spectral_mode=mode,
                maximum_spectral_points=self.max_spectral_points.value(),
            )
            self.latest_spectrum["requested_time"] = self.time_s.value()
            self.latest_spectrum["position_source"] = "specified_location"
            self.latest_sample = dict(sample)
            self.latest_parameters = dict(parameters)
            result = self.latest_spectrum
            self._update_spectrum_summary(result)
            summary = self.result_summary.text()
            self._log("辐射传输计算完成：" + summary)
            self._save_project(silent=True)
            AtmosphericSpectrumPreviewDialog(result, self).exec()
        except Exception as exc:  # noqa: BLE001
            self._show_error("大气辐射传输计算失败", exc)
        finally:
            progress.close()

    def _export_spectrum(self) -> None:
        if not self.latest_spectrum:
            QMessageBox.information(self, "导出光谱", "尚无可导出的计算结果。")
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        default = self.workspace / "results" / "atmospheric_spectrum.csv"
        default.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "导出大气光谱", str(default), "CSV (*.csv)"
        )
        if not path:
            return
        result = self.latest_spectrum
        columns = [
            np.asarray(result["wavelength_um"], dtype=float),
            np.asarray(result["wavenumber_cm"], dtype=float),
            np.asarray(result["earth_thermal_spectral_irradiance"], dtype=float),
            np.asarray(result["earth_reflected_spectral_irradiance"], dtype=float),
            np.asarray(result["earth_total_spectral_irradiance"], dtype=float),
        ]
        np.savetxt(
            path,
            np.column_stack(columns),
            delimiter=",",
            header=(
                "wavelength_um,wavenumber_cm,earth_thermal_spectrum,"
                "earth_reflected_spectrum,earth_total_spectrum"
            ),
            comments="",
        )
        metadata = Path(path).with_suffix(".json")
        serializable = {
            key: value
            for key, value in result.items()
            if not isinstance(value, np.ndarray)
        }
        metadata.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._log(f"光谱已导出：{path}")

    def _handle_hapi_result(self, result: dict[str, Any]) -> None:
        path = str(result["total_optical_depth_file"])
        try:
            self.radiation.atmosphere.load_absorption_optical_depth(path)
            self.absorption_file.setText(path)
            profile_path = str(result.get("profile_file", ""))
            if profile_path:
                self.pattern_workbench.set_query_profile(profile_path)
            self.corrections = []
            self._clear_effective_cloud_fit()
            manifest_path = Path(str(result.get("manifest_file", "")))
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                profile = manifest.get("profile", {})
                source = Path(str(profile.get("source_file", ""))).expanduser()
                if source.suffix.lower() in {".nc", ".nc4"} and source.is_file():
                    self.nucaps_cloud_file = str(source.resolve())
                    self.nucaps_cloud_for_index = int(profile.get("for_index", 0))
            self._log(f"HAPI 35层总光学厚度生成并加载：{path}")
            self._save_project(silent=True)
        except Exception as exc:  # noqa: BLE001
            self._show_error("加载 HAPI 计算结果失败", exc)

    def _handle_pattern_optical_depth(
        self, path: str, prediction: dict[str, Any]
    ) -> None:
        try:
            self.radiation.atmosphere.load_absorption_optical_depth(path)
            self.absorption_file.setText(path)
            self.corrections = []
            self._clear_effective_cloud_fit()
            self.nucaps_cloud_file = ""
            self.nucaps_cloud_for_index = 0
            confidence = float(prediction.get("confidence", 0.0))
            self._log(
                f"大气状态模式插值光学厚度已加载：{path}；置信度={confidence:.1%}"
            )
            self._save_project(silent=True)
        except Exception as exc:  # noqa: BLE001
            self._show_error("加载模式插值光学厚度失败", exc)

    def _open_hapi(self) -> None:
        self.navigation.setCurrentRow(1)

    def _open_hapi_profile(self, source: str, for_index: int) -> None:
        if self.hapi_workbench.set_profile_selection(source, for_index):
            self.navigation.setCurrentRow(1)
            self._log(f"已在HAPI工作台加载代表廓线：{source}；FOR={for_index}")
        else:
            self._log(f"代表廓线加载失败：{source}；FOR={for_index}")

    def _open_hapi_batch(self, profiles: object) -> None:
        batch_profiles = list(profiles) if isinstance(profiles, list) else []
        if self.hapi_workbench.set_batch_profiles(batch_profiles):
            self.navigation.setCurrentRow(1)
            self._log(
                f"已在HAPI工作台装载{len(batch_profiles)}个最终代表廓线；"
                "请统一设置光谱网格后开始批量计算。"
            )
        else:
            self._log("代表廓线批量任务装载失败；请确认廓线文件存在且当前没有计算任务。")

    def _handle_hapi_batch_result(self, result: dict[str, Any]) -> None:
        try:
            complete = self.pattern_workbench.apply_batch_optical_depth_result(result)
            if complete:
                self._save_project(silent=True)
        except Exception as exc:  # noqa: BLE001
            self._show_error("绑定代表模式光学厚度失败", exc)

    def _open_validation(self) -> None:
        self.navigation.setCurrentRow(2)

    def _automatically_correct_optical_depth(
        self, comparison: dict[str, Any]
    ) -> dict[str, Any] | None:
        """在已拟合有效云量下修正高层温度和逐通道光学厚度。"""
        parent = QApplication.activeModalWidget() or self
        atmosphere = self.radiation.atmosphere
        grid = self.modis.get_grid()
        if atmosphere._absorption_tau_original is None:
            QMessageBox.information(parent, "自动矫正", "请先导入总气体分子光学厚度。")
            return None
        if grid is None or self.latest_sample is None or self.latest_parameters is None:
            QMessageBox.information(parent, "自动矫正", "请先重新执行一次高分辨率单柱求解。")
            return None
        if not self.latest_spectrum or self.latest_spectrum.get("spectral_quantity") != "radiance":
            QMessageBox.information(parent, "自动矫正", "自动矫正必须基于高分辨率目标单柱谱辐亮度。")
            return None

        try:
            wavelength = np.asarray(comparison["wavelength_um"], dtype=float).reshape(-1)
            observed = np.asarray(comparison["observed"], dtype=float).reshape(-1)
            simulated = np.asarray(comparison["simulated"], dtype=float).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            self._show_error("验证光谱无效", exc)
            return None
        valid = (
            np.isfinite(wavelength)
            & np.isfinite(observed)
            & np.isfinite(simulated)
            & (wavelength > 0.0)
            & (observed >= 0.0)
        )
        wavelength, observed, simulated = (
            wavelength[valid], observed[valid], simulated[valid]
        )
        order = np.argsort(wavelength)
        wavelength, observed, simulated = (
            wavelength[order], observed[order], simulated[order]
        )
        if wavelength.size < 3:
            QMessageBox.warning(parent, "自动矫正", "共同波段有效通道不足。")
            return None

        validator = SpectrumValidationManager()
        temperature_mask = (wavelength >= 14.8) & (wavelength <= 15.1)
        optical_mask = (wavelength < 14.8) | (
            (wavelength > 15.1) & (wavelength <= 15.35)
        )
        if np.count_nonzero(temperature_mask) < 3:
            QMessageBox.warning(parent, "自动矫正", "14.8～15.1 μm 内至少需要3个有效通道。")
            return None
        if np.count_nonzero(optical_mask) < 2:
            QMessageBox.warning(parent, "自动矫正", "可参与光学厚度矫正的通道不足。")
            return None

        previous_corrections = list(self.corrections)
        previous_offset = self.temperature_offset.value()
        temperature_candidates = np.arange(-15.0, 15.0 + 0.5, 1.0)
        factor_candidates = np.unique(np.r_[np.geomspace(0.2, 5.0, 17), 1.0])
        total = int(temperature_candidates.size + factor_candidates.size + 1)
        progress = QProgressDialog("正在按拟合云量自动矫正总光学厚度…", "取消", 0, total, parent)
        progress.setWindowTitle("拟合云量并矫正总光学厚度")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()
        evaluation = 0
        succeeded = False

        base_parameters = dict(self.latest_parameters)
        base_parameters.update({
            "enable_optical_depth_correction": False,
            "optical_depth_corrections": [],
        })

        def keep_running(_step: int, _total: int) -> bool:
            QApplication.processEvents()
            return not progress.wasCanceled()

        def evaluate(parameters: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
            spectrum = self.radiation.compute_spectrum_at_position(
                grid,
                dict(self.latest_sample or {}),
                parameters,
                progress_callback=keep_running,
                spectral_mode="high_resolution",
                wavelength_grid_um=wavelength,
            )
            raw = np.interp(
                wavelength,
                np.asarray(spectrum["wavelength_um"], dtype=float),
                np.asarray(spectrum["earth_total_spectral_irradiance"], dtype=float),
            )
            return spectrum, validator.three_point_hamming(
                raw, 1.0e4 / np.asarray(wavelength, dtype=float)
            )

        try:
            baseline_rmse = float(np.sqrt(np.mean(np.square(simulated - observed))))
            best_offset = 0.0
            best_temperature_rmse = float("inf")
            for offset in temperature_candidates:
                if progress.wasCanceled():
                    raise RuntimeError("自动矫正已取消。")
                trial = dict(base_parameters)
                trial["upper_atmosphere_temperature_offset_k"] = float(offset)
                _, trial_spectrum = evaluate(trial)
                score = float(np.sqrt(np.mean(np.square(
                    trial_spectrum[temperature_mask] - observed[temperature_mask]
                ))))
                if score < best_temperature_rmse:
                    best_temperature_rmse = score
                    best_offset = float(offset)
                evaluation += 1
                progress.setValue(evaluation)
                progress.setLabelText(
                    f"搜索10 km以上温度偏移：{offset:+.0f} K，RMSE={score:.5g}"
                )
                QApplication.processEvents()

            response: list[np.ndarray] = []
            for factor in factor_candidates:
                if progress.wasCanceled():
                    raise RuntimeError("自动矫正已取消。")
                broad: list[dict[str, float]] = []
                if wavelength[0] < 14.8:
                    broad.append({
                        "wavelength_min_um": float(wavelength[0]),
                        "wavelength_max_um": min(14.8, float(wavelength[-1])),
                        "factor": float(factor),
                    })
                lower = max(15.1, float(wavelength[0]))
                upper = min(15.35, float(wavelength[-1]))
                if upper > lower:
                    broad.append({
                        "wavelength_min_um": lower,
                        "wavelength_max_um": upper,
                        "factor": float(factor),
                    })
                trial = dict(base_parameters)
                trial.update({
                    "upper_atmosphere_temperature_offset_k": best_offset,
                    "enable_optical_depth_correction": bool(broad),
                    "optical_depth_corrections": broad,
                })
                _, trial_spectrum = evaluate(trial)
                response.append(trial_spectrum)
                evaluation += 1
                progress.setValue(evaluation)
                progress.setLabelText(f"采样光学厚度倍率：{factor:.5g}")
                QApplication.processEvents()

            response_matrix = np.vstack(response)
            best_indices = np.argmin(np.abs(response_matrix - observed[None, :]), axis=0)
            factors = factor_candidates[best_indices]
            corrections = validator.build_per_wavenumber_corrections(
                wavelength, factors, optical_mask
            )
            final_parameters = dict(base_parameters)
            final_parameters.update({
                "upper_atmosphere_temperature_offset_k": best_offset,
                "enable_optical_depth_correction": bool(corrections),
                "optical_depth_corrections": corrections,
            })
            corrected, corrected_values = evaluate(final_parameters)
            corrected_rmse = float(
                np.sqrt(np.mean(np.square(corrected_values - observed)))
            )
            evaluation += 1
            progress.setValue(evaluation)
            if not np.isfinite(corrected_rmse) or corrected_rmse >= baseline_rmse:
                QMessageBox.information(
                    parent,
                    "自动矫正未保存",
                    "试算没有降低整体 RMSE，已恢复原参数。\n"
                    f"原 RMSE：{baseline_rmse:.6g}\n试算 RMSE：{corrected_rmse:.6g}",
                )
                return None

            atmosphere.apply_optical_depth_corrections(corrections)
            atmosphere.set_upper_atmosphere_temperature_offset(best_offset)
            self.corrections = corrections
            self.temperature_offset.setValue(best_offset)
            self.latest_parameters = final_parameters
            corrected["requested_time"] = self.latest_spectrum.get("requested_time", 0.0)
            corrected["position_source"] = "specified_location"
            corrected["optical_depth_auto_correction"] = {
                "method": "per_wavenumber_hamming3_with_upper_temperature_search",
                "upper_atmosphere_temperature_offset_k": best_offset,
                "temperature_only_band_um": [14.8, 15.1],
                "optical_depth_max_wavelength_um": 15.35,
                "correction_count": len(corrections),
                "rmse_before": baseline_rmse,
                "rmse_after": corrected_rmse,
            }
            self.latest_spectrum = dict(corrected)
            output = self.workspace / "config" / "atmosphere" / "optical_depth_total_corrected.csv"
            atmosphere.save_corrected_absorption_optical_depth(output)
            diagnostics = output.with_name("automatic_correction.json")
            diagnostics.write_text(
                json.dumps(
                    corrected["optical_depth_auto_correction"],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            succeeded = True
            self._log(
                f"自动矫正完成：温度偏移 {best_offset:+.1f} K，"
                f"RMSE {baseline_rmse:.6g} → {corrected_rmse:.6g}，结果={output}"
            )
            self._save_project(silent=True)
            return dict(corrected)
        except Exception as exc:  # noqa: BLE001
            if "取消" not in str(exc):
                self._show_error("自动矫正失败", exc)
            return None
        finally:
            if not succeeded:
                if previous_corrections:
                    atmosphere.apply_optical_depth_corrections(previous_corrections)
                else:
                    atmosphere.clear_optical_depth_corrections()
                atmosphere.set_upper_atmosphere_temperature_offset(previous_offset)
            progress.close()

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and isinstance(
            watched, (QAbstractSpinBox, QComboBox)
        ):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._save_project(silent=True):
            event.accept()
        else:
            choice = QMessageBox.question(
                self,
                "项目自动保存失败",
                "项目未能自动保存，仍要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {message}")

    def _show_error(self, title: str, error: Exception) -> None:
        self._log(f"错误：{title}：{error}")
        QMessageBox.warning(self, title, str(error))
