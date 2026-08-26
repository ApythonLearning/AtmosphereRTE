from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

from PySide6.QtCore import QProcess, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.hapi_optical_depth_manager import (
    AtmosphericProfileReader,
    HapiOpticalDepthManager,
)
from dialogs.window_controls import configure_resizable_dialog


class HapiOpticalDepthDialog(QDialog):
    """从NUCAPS廓线生成HAPI逐层吸收光学厚度。"""

    resultReady = Signal(dict)
    batchResultReady = Signal(dict)

    def __init__(
        self,
        project_dir: str | Path,
        parent: QWidget | None = None,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.embedded = bool(embedded)
        self.project_dir = Path(project_dir).resolve()
        self.manager = HapiOpticalDepthManager()
        self.inspection: dict[str, Any] | None = None
        self.result_data: dict[str, Any] | None = None
        self._process: QProcess | None = None
        self._job_dir: Path | None = None
        self._cancel_path: Path | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._pending_state = ""
        self._working = False
        self._batch_profiles: list[dict[str, Any]] = []

        self.setWindowTitle("HITRAN 分层吸收光学厚度计算")
        if self.embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
            self.setMinimumSize(0, 0)
        else:
            configure_resizable_dialog(self)
            self.resize(920, 860)
            self.setMinimumSize(780, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_profile_group())
        layout.addWidget(self._build_gas_group(), 1)
        layout.addWidget(self._build_spectral_group())
        layout.addWidget(self._build_optimization_group())
        layout.addWidget(self._build_progress_group())

        command_row = QHBoxLayout()
        self.start_button = QPushButton("开始计算并保存到当前项目")
        self.start_button.clicked.connect(self._start_calculation)
        self.cancel_button = QPushButton("取消计算")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_calculation)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self._close_dialog)
        self.close_button.setVisible(not self.embedded)
        command_row.addWidget(self.start_button)
        command_row.addWidget(self.cancel_button)
        command_row.addStretch(1)
        command_row.addWidget(self.close_button)
        layout.addLayout(command_row)

        self._load_default_profile()

    def set_project_dir(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        if not self._working:
            self._batch_profiles = []
            self._update_start_button_text()

    def set_profile_selection(self, source: str | Path, for_index: int) -> bool:
        """Load a NUCAPS product or an exported learned representative profile."""
        if self._working:
            return False
        self._batch_profiles = []
        self._inspect_profile(Path(source).expanduser().resolve())
        if self.inspection is None:
            return False
        loaded = Path(self.profile_path.text()).resolve()
        if loaded != Path(source).expanduser().resolve():
            return False
        if for_index < self.for_index.minimum() or for_index > self.for_index.maximum():
            return False
        self.for_index.setValue(int(for_index))
        self._update_start_button_text()
        return True

    def set_batch_profiles(self, profiles: list[dict[str, Any]]) -> bool:
        """Prepare a sequential HAPI job using one spectral grid for all profiles."""
        if self._working or not profiles:
            return False
        normalized: list[dict[str, Any]] = []
        for position, item in enumerate(profiles):
            source = Path(str(item.get("profile_path", ""))).expanduser().resolve()
            if not source.is_file():
                return False
            normalized.append(
                {
                    "batch_index": int(item.get("batch_index", position)),
                    "profile_path": str(source),
                    "for_index": int(item.get("for_index", 0)),
                }
            )
        first = normalized[0]
        self._batch_profiles = []
        self._inspect_profile(Path(first["profile_path"]))
        if self.inspection is None or Path(self.profile_path.text()).resolve() != Path(
            first["profile_path"]
        ):
            return False
        self.for_index.setValue(int(first["for_index"]))
        self._batch_profiles = normalized
        self._update_profile_summary()
        self._update_start_button_text()
        self.result_summary.setText(
            f"已装载{len(normalized)}个代表廓线。它们将采用相同波数网格顺序计算；"
            "完成后结果会自动返回大气廓线模式库。"
        )
        return True

    @property
    def batch_profile_count(self) -> int:
        return len(self._batch_profiles)

    def _update_start_button_text(self) -> None:
        if not hasattr(self, "start_button"):
            return
        if self._batch_profiles:
            self.start_button.setText(
                f"开始批量计算全部 {len(self._batch_profiles)} 个代表模式"
            )
        else:
            self.start_button.setText("开始计算并保存到当前项目")

    def set_line_table_sources(self, sources: dict[str, str]) -> None:
        self.manager.set_table_sources(sources)
        self._update_database_directory_label()
        if hasattr(self, "gas_table"):
            self._update_gas_table()

    def line_table_sources(self) -> dict[str, str]:
        return self.manager.export_table_sources()

    def _update_database_directory_label(self) -> None:
        external_count = len(self.manager.export_table_sources())
        suffix = f"；外部原位引用 {external_count} 组" if external_count else ""
        self.database_directory_label.setText(
            f"下载缓存目录：{self.manager.database_dir}{suffix}"
        )

    def _build_profile_group(self) -> QGroupBox:
        group = QGroupBox("大气廓线（NUCAPS或最终代表模式）")
        form = QFormLayout(group)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_path = QLineEdit()
        self.profile_path.setReadOnly(True)
        browse_button = QPushButton("选择廓线…")
        browse_button.clicked.connect(self._browse_profile)
        file_layout.addWidget(self.profile_path, 1)
        file_layout.addWidget(browse_button)
        form.addRow("廓线文件", file_row)

        match_row = QWidget()
        match_layout = QHBoxLayout(match_row)
        match_layout.setContentsMargins(0, 0, 0, 0)
        self.target_latitude = QDoubleSpinBox()
        self.target_latitude.setRange(-90.0, 90.0)
        self.target_latitude.setDecimals(7)
        self.target_latitude.setSuffix("°")
        self.target_longitude = QDoubleSpinBox()
        self.target_longitude.setRange(-180.0, 180.0)
        self.target_longitude.setDecimals(7)
        self.target_longitude.setSuffix("°")
        match_button = QPushButton("匹配最近有效FOR")
        match_button.clicked.connect(self._match_nearest_profile)
        match_layout.addWidget(QLabel("纬度"))
        match_layout.addWidget(self.target_latitude)
        match_layout.addWidget(QLabel("经度"))
        match_layout.addWidget(self.target_longitude)
        match_layout.addWidget(match_button)
        form.addRow("目标位置", match_row)

        self.for_index = QSpinBox()
        self.for_index.setRange(0, 0)
        self.for_index.valueChanged.connect(self._update_profile_summary)
        form.addRow("廓线/FOR索引", self.for_index)

        self.profile_summary = QLabel("请选择NUCAPS NetCDF或模式学习导出的35层CSV。")
        self.profile_summary.setWordWrap(True)
        self.profile_summary.setTextInteractionFlags(
            self.profile_summary.textInteractionFlags()
        )
        form.addRow("匹配结果", self.profile_summary)
        return group

    def _build_gas_group(self) -> QGroupBox:
        group = QGroupBox("自动识别的气相吸收组分")
        layout = QVBoxLayout(group)
        note = QLabel(
                    "液态水属于云消光，由现有云辐射模块处理。"
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        layout.addWidget(note)
        self.gas_table = QTableWidget(0, 3)
        self.gas_table.setHorizontalHeaderLabels(["气体", "廓线数据", "HITRAN谱线"])
        self.gas_table.verticalHeader().setVisible(False)
        self.gas_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gas_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.gas_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.gas_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.gas_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.gas_table)
        import_row = QHBoxLayout()
        import_files_button = QPushButton("导入本地HITRAN数据…")
        import_files_button.clicked.connect(self._import_local_hitran_files)
        import_directory_button = QPushButton("批量导入目录…")
        import_directory_button.clicked.connect(self._import_local_hitran_directory)
        self.database_directory_label = QLabel(
            f"下载缓存目录：{self.manager.database_dir}"
        )
        self.database_directory_label.setObjectName("secondaryText")
        self.database_directory_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        import_row.addWidget(import_files_button)
        import_row.addWidget(import_directory_button)
        import_row.addWidget(self.database_directory_label, 1)
        layout.addLayout(import_row)
        return group

    def _import_local_hitran_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择本地HITRAN/HAPI谱线文件",
            "",
            "HITRAN/HAPI谱线文件 (*.par *.data *.header);;所有文件 (*)",
        )
        if paths:
            self._import_local_hitran(paths)

    def _import_local_hitran_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择包含HITRAN .par或HAPI文件对的目录", ""
        )
        if path:
            self._import_local_hitran([path])

    def _import_local_hitran(self, sources: list[str]) -> None:
        try:
            result = self.manager.import_line_tables(sources, overwrite=False)
        except FileExistsError as exc:
            choice = QMessageBox.question(
                self,
                "覆盖已有HITRAN数据",
                f"{exc}\n\n是否使用所选本地文件覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
            try:
                result = self.manager.import_line_tables(sources, overwrite=True)
            except Exception as retry_exc:  # noqa: BLE001
                QMessageBox.warning(self, "导入本地HITRAN数据失败", str(retry_exc))
                return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导入本地HITRAN数据失败", str(exc))
            return
        self._update_gas_table()
        self._update_database_directory_label()
        imported = "、".join(result["imported_gases"])
        overwritten = result.get("overwritten_gases", [])
        overwrite_text = (
            f"\n已覆盖：{'、'.join(overwritten)}" if overwritten else ""
        )
        storage_lines: list[str] = []
        if result.get("referenced_gases"):
            storage_lines.append(
                "HAPI文件对已原地引用，不会复制。\n"
                f"源目录：{'；'.join(result.get('source_directories', []))}"
            )
        if result.get("copied_gases"):
            storage_lines.append("原始.par文件已转换到下载缓存目录。")
        storage_text = "\n".join(storage_lines)
        QMessageBox.information(
            self,
            "本地HITRAN数据导入完成",
            f"已导入：{imported}{overwrite_text}\n"
            f"{storage_text}\n"
            f"下载缓存目录：{result['database_dir']}",
        )

    def _build_spectral_group(self) -> QGroupBox:
        group = QGroupBox("计算波数网格")
        form = QFormLayout(group)
        range_row = QWidget()
        range_layout = QHBoxLayout(range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.wavenumber_min = QDoubleSpinBox()
        self.wavenumber_min.setRange(1.0, 50_000.0)
        self.wavenumber_min.setDecimals(3)
        self.wavenumber_min.setValue(self.manager.DEFAULT_WAVENUMBER_MIN_CM)
        self.wavenumber_min.setSuffix(" cm⁻¹")
        self.wavenumber_max = QDoubleSpinBox()
        self.wavenumber_max.setRange(1.0, 50_000.0)
        self.wavenumber_max.setDecimals(3)
        self.wavenumber_max.setValue(self.manager.DEFAULT_WAVENUMBER_MAX_CM)
        self.wavenumber_max.setSuffix(" cm⁻¹")
        range_layout.addWidget(self.wavenumber_min)
        range_layout.addWidget(QLabel("至"))
        range_layout.addWidget(self.wavenumber_max)
        form.addRow("波数范围", range_row)

        self.wavenumber_step = QDoubleSpinBox()
        self.wavenumber_step.setRange(0.01, 1.0)
        self.wavenumber_step.setDecimals(3)
        self.wavenumber_step.setSingleStep(0.1)
        self.wavenumber_step.setValue(self.manager.DEFAULT_WAVENUMBER_STEP_CM)
        self.wavenumber_step.setSuffix(" cm⁻¹")
        form.addRow("波数间隔", self.wavenumber_step)
        note = QLabel(
            "推荐范围500–33300 cm⁻¹，与现有高分辨率大气辐射求解器一致；"
            "0.5 cm⁻¹约包含65601个采样点，完整计算可能耗时较长。"
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        form.addRow("说明", note)
        return group

    def _build_optimization_group(self) -> QGroupBox:
        group = QGroupBox("性能与输出")
        form = QFormLayout(group)

        self.calculation_mode = QComboBox()
        self.calculation_mode.addItem("快速预览（1.0 cm⁻¹）", "fast")
        self.calculation_mode.addItem("标准计算（0.5 cm⁻¹）", "standard")
        self.calculation_mode.addItem("高精度（0.25 cm⁻¹）", "high")
        self.calculation_mode.setCurrentIndex(1)
        self.calculation_mode.currentIndexChanged.connect(self._apply_calculation_mode)
        form.addRow("计算模式", self.calculation_mode)

        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, min(4, max(os.cpu_count() or 1, 1)))
        self.max_workers.setValue(
            min(self.manager.DEFAULT_MAX_WORKERS, self.max_workers.maximum())
        )
        self.max_workers.setSuffix(" 个气体进程")
        form.addRow("最大并发", self.max_workers)

        option_row = QWidget()
        option_layout = QHBoxLayout(option_row)
        option_layout.setContentsMargins(0, 0, 0, 0)
        self.use_cache = QCheckBox("精确截面缓存")
        self.use_cache.setChecked(True)
        self.prefilter_lines = QCheckBox("按波数范围预筛谱线")
        self.prefilter_lines.setChecked(True)
        self.save_components = QCheckBox("保存各气体分量")
        self.save_components.setChecked(True)
        option_layout.addWidget(self.use_cache)
        option_layout.addWidget(self.prefilter_lines)
        option_layout.addWidget(self.save_components)
        option_layout.addStretch(1)
        form.addRow("选项", option_row)

        note = QLabel(
            "计算始终在独立进程中执行；并发数最多限制为4。关闭气体分量输出可减少磁盘写入，"
            "不会影响逐层总光学厚度。"
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        form.addRow("说明", note)
        return group

    @Slot()
    def _apply_calculation_mode(self) -> None:
        mode = str(self.calculation_mode.currentData())
        step_by_mode = {"fast": 1.0, "standard": 0.5, "high": 0.25}
        self.wavenumber_step.setValue(step_by_mode.get(mode, 0.5))

    def _build_progress_group(self) -> QGroupBox:
        group = QGroupBox("计算状态")
        layout = QVBoxLayout(group)
        self.progress_label = QLabel("等待开始计算。")
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.result_summary = QLabel("")
        self.result_summary.setWordWrap(True)
        self.result_summary.setTextInteractionFlags(
            self.result_summary.textInteractionFlags()
        )
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_summary)
        return group

    def _load_default_profile(self) -> None:
        application_dir = Path(__file__).resolve().parents[1]
        profile_root = application_dir / "resources" / "data" / "atmospheric_profiles"
        candidates = sorted(profile_root.rglob("*.nc")) if profile_root.exists() else []
        if not candidates:
            return
        source = candidates[0]
        match_path = source.parent / "profile_match.json"
        if match_path.exists():
            try:
                match = json.loads(match_path.read_text(encoding="utf-8"))
                target = match.get("target", {})
                self.target_latitude.setValue(float(target.get("latitude_deg", 0.0)))
                self.target_longitude.setValue(float(target.get("longitude_deg", 0.0)))
            except (OSError, TypeError, ValueError):
                pass
        self._inspect_profile(source)

    def _browse_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择大气廓线",
            self.profile_path.text(),
            "大气廓线 (*.nc *.nc4 *.csv);;NUCAPS NetCDF (*.nc *.nc4);;35层CSV (*.csv);;所有文件 (*)",
        )
        if path:
            self._batch_profiles = []
            self._inspect_profile(Path(path))
            self._update_start_button_text()

    def _inspect_profile(self, source: Path) -> None:
        try:
            inspection = AtmosphericProfileReader.inspect(source)
            nearest = AtmosphericProfileReader.nearest_valid_for(
                inspection,
                self.target_latitude.value(),
                self.target_longitude.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "大气廓线读取失败", str(exc))
            return
        self.inspection = inspection
        self.profile_path.setText(str(source.resolve()))
        self.for_index.blockSignals(True)
        self.for_index.setRange(0, max(int(inspection["profile_count"]) - 1, 0))
        self.for_index.setValue(nearest)
        self.for_index.blockSignals(False)
        self._update_profile_summary()
        self._update_gas_table()

    def _match_nearest_profile(self) -> None:
        if self.inspection is None:
            QMessageBox.information(self, "匹配大气廓线", "请先选择有效的大气廓线文件。")
            return
        try:
            nearest = AtmosphericProfileReader.nearest_valid_for(
                self.inspection,
                self.target_latitude.value(),
                self.target_longitude.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "匹配大气廓线失败", str(exc))
            return
        self.for_index.setValue(nearest)

    def _update_profile_summary(self) -> None:
        if self.inspection is None:
            return
        try:
            summary = AtmosphericProfileReader.profile_summary(
                self.inspection, self.for_index.value()
            )
        except Exception as exc:  # noqa: BLE001
            self.profile_summary.setText(str(exc))
            return
        self.profile_summary.setText(
            f"索引 {summary['for_index']}；{summary['observation_time_utc']}；"
            f"{summary['latitude_deg']:.7f}°, {summary['longitude_deg']:.7f}°；"
            f"Quality_Flag={summary['quality_flag']}；"
            f"有效温压层={summary['valid_level_count']}"
            + (
                f"\n批量模式：共{len(self._batch_profiles)}个最终代表廓线，"
                "本页参数将统一应用于全部廓线。"
                if self._batch_profiles
                else ""
            )
        )

    def _update_gas_table(self) -> None:
        gases = list(self.inspection.get("gas_names", [])) if self.inspection else []
        status = self.manager.line_database_status(gases)
        external_gases = set(self.manager.export_table_sources())
        self.gas_table.setRowCount(len(gases))
        for row, name in enumerate(gases):
            self.gas_table.setItem(row, 0, QTableWidgetItem(name))
            self.gas_table.setItem(row, 1, QTableWidgetItem("已识别，将自动计算"))
            self.gas_table.setItem(
                row,
                2,
                QTableWidgetItem(
                    "外部文件可用（原位使用）"
                    if name in external_gases and status.get(name)
                    else "本地缓存可用"
                    if status.get(name)
                    else "缺失，计算时自动下载"
                ),
            )

    def _start_calculation(self) -> None:
        if self._working:
            return
        if self.inspection is None or not self.profile_path.text():
            QMessageBox.information(self, "HAPI光学厚度计算", "请先选择有效的大气廓线。")
            return
        if self.wavenumber_max.value() <= self.wavenumber_min.value():
            QMessageBox.information(self, "HAPI光学厚度计算", "波数上限必须大于下限。")
            return

        output_root = self.project_dir / "config" / "atmosphere" / "hapi_optical_depth"
        arguments = {
            "profile_path": self.profile_path.text(),
            "for_index": self.for_index.value(),
            "wavenumber_min_cm": self.wavenumber_min.value(),
            "wavenumber_max_cm": self.wavenumber_max.value(),
            "wavenumber_step_cm": self.wavenumber_step.value(),
            "output_root": str(output_root),
            "max_workers": self.max_workers.value(),
            "use_cache": self.use_cache.isChecked(),
            "prefilter_lines": self.prefilter_lines.isChecked(),
            "save_components": self.save_components.isChecked(),
            "calculation_mode": str(self.calculation_mode.currentData()),
        }
        self.result_data = None
        self.result_summary.clear()
        self._pending_state = ""
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        worker_root = output_root / ".worker_jobs"
        self._job_dir = worker_root / uuid.uuid4().hex
        self._job_dir.mkdir(parents=True, exist_ok=False)
        self._cancel_path = self._job_dir / "cancel.flag"
        request_path = self._job_dir / "request.json"
        request = {
            "hapi_path": str(self.manager.hapi_path),
            "database_dir": str(self.manager.database_dir),
            "table_sources": self.manager.export_table_sources(),
            "cancel_path": str(self._cancel_path),
            "arguments": arguments,
        }
        if self._batch_profiles:
            request["batch_profiles"] = list(self._batch_profiles)
        request_path.write_text(
            json.dumps(
                request,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self._process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._process.readyReadStandardError.connect(self._read_process_stderr)
        self._process.errorOccurred.connect(self._process_error)
        self._process.finished.connect(self._process_finished)

        self._working = True
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("正在启动独立HAPI计算进程…")
        self._process.start(
            sys.executable,
            ["-m", "core.hapi_optical_depth_worker", str(request_path)],
        )

    @Slot(int, int, str)
    def _update_progress(self, value: int, maximum: int, message: str) -> None:
        if maximum > 0:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(max(0, min(value, maximum)))
        else:
            self.progress_bar.setRange(0, 0)
        self.progress_label.setText(message)

    @Slot()
    def _read_process_stdout(self) -> None:
        if self._process is None:
            return
        self._stdout_buffer += bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._handle_worker_event(line)

    @Slot()
    def _read_process_stderr(self) -> None:
        if self._process is None:
            return
        self._stderr_buffer += bytes(self._process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if len(self._stderr_buffer) > 20_000:
            self._stderr_buffer = self._stderr_buffer[-20_000:]

    def _handle_worker_event(self, line: str) -> None:
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except ValueError:
            self._stderr_buffer += line + "\n"
            return
        event_type = str(event.get("type", ""))
        if event_type == "progress":
            self._update_progress(
                int(event.get("value", 0)),
                int(event.get("maximum", 0)),
                str(event.get("message", "")),
            )
        elif event_type == "result":
            self._calculation_succeeded(event.get("result"))
        elif event_type == "batch_result":
            self._batch_calculation_succeeded(event.get("result"))
        elif event_type == "cancelled":
            self._calculation_cancelled(str(event.get("message", "计算已取消。")))
        elif event_type == "error":
            self._calculation_failed(str(event.get("message", "HAPI计算失败。")))

    @Slot(object)
    def _calculation_succeeded(self, result: object) -> None:
        self.result_data = dict(result) if isinstance(result, dict) else None
        self._pending_state = "success"
        self._update_gas_table()
        if self.result_data is not None:
            gas_text = "、".join(self.result_data.get("gas_names", []))
            downloaded = self.result_data.get("downloaded_gases", [])
            download_text = "、".join(downloaded) if downloaded else "无（全部使用本地谱线）"
            optimization = self.result_data.get("optimization", {})
            performance = self.result_data.get("performance", {})
            self.result_summary.setText(
                f"已生成{self.result_data.get('layer_count', '-')}层、"
                f"{self.result_data.get('spectral_point_count', '-')}个波数点。\n"
                f"吸收气体：{gas_text}\n"
                f"本次下载：{download_text}\n"
                f"并发进程：{optimization.get('max_workers', '-')}；"
                f"截面缓存命中：{optimization.get('coefficient_cache_hits', 0)}\n"
                f"总耗时：{float(performance.get('total_seconds', 0.0)):.1f}秒\n"
                f"总光学厚度：{self.result_data.get('total_optical_depth_file', '')}"
            )
            self.resultReady.emit(dict(self.result_data))

    @Slot(object)
    def _batch_calculation_succeeded(self, result: object) -> None:
        batch_result = dict(result) if isinstance(result, dict) else {}
        results = list(batch_result.get("results", []))
        failures = list(batch_result.get("failures", []))
        self.result_data = batch_result
        self._pending_state = "success"
        elapsed = sum(
            float(dict(item).get("performance", {}).get("total_seconds", 0.0))
            for item in results
            if isinstance(item, dict)
        )
        failure_text = ""
        if failures:
            details = "；".join(
                f"模式{int(item.get('batch_index', -1)) + 1}：{item.get('message', '未知错误')}"
                for item in failures[:5]
            )
            failure_text = f"\n失败{len(failures)}个：{details}"
        first = dict(results[0]) if results else {}
        self.result_summary.setText(
            f"代表廓线批量计算完成：成功{len(results)}个，失败{len(failures)}个；"
            f"统一网格{first.get('spectral_point_count', '-')}点；累计耗时{elapsed:.1f}秒。"
            f"{failure_text}"
        )
        self._update_gas_table()
        self.batchResultReady.emit(batch_result)

    @Slot(str)
    def _calculation_failed(self, message: str) -> None:
        self._pending_state = "failed"
        self.result_summary.setText(message)

    @Slot(str)
    def _calculation_cancelled(self, message: str) -> None:
        self._pending_state = "cancelled"
        self.result_summary.setText(message)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart or not self._working:
            return
        message = self._process.errorString() if self._process is not None else "无法启动计算进程"
        self._calculation_failed(message)
        self._finalize_process()

    def _process_finished(
        self, exit_code: int, _exit_status: QProcess.ExitStatus
    ) -> None:
        self._read_process_stdout()
        self._read_process_stderr()
        if self._stdout_buffer.strip():
            self._handle_worker_event(self._stdout_buffer)
            self._stdout_buffer = ""
        if not self._pending_state:
            detail = self._stderr_buffer.strip().splitlines()
            suffix = detail[-1] if detail else f"子进程退出码：{exit_code}"
            self._calculation_failed(f"HAPI计算进程异常结束：{suffix}")
        self._finalize_process()

    def _finalize_process(self) -> None:
        if not self._working:
            return
        self._working = False
        self.start_button.setEnabled(True)
        self._update_start_button_text()
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        if self._pending_state == "success":
            self.progress_bar.setValue(self.progress_bar.maximum())
            if self._batch_profiles and isinstance(self.result_data, dict):
                success_count = len(self.result_data.get("results", []))
                failure_count = len(self.result_data.get("failures", []))
                self.progress_label.setText(
                    f"批量任务结束：成功{success_count}个，失败{failure_count}个。"
                )
            else:
                self.progress_label.setText(
                    "计算完成，结果已应用到当前项目。"
                    if self.embedded
                    else "计算完成，可关闭窗口并应用到当前项目。"
                )
        elif self._pending_state == "failed":
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("计算失败。")
            QMessageBox.warning(self, "HAPI分层光学厚度计算失败", self.result_summary.text())
        elif self._pending_state == "cancelled":
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText("计算已取消。")
        if self._process is not None:
            self._process.deleteLater()
        self._process = None
        if self._job_dir is not None:
            shutil.rmtree(self._job_dir, ignore_errors=True)
        self._job_dir = None
        self._cancel_path = None

    def _cancel_calculation(self) -> None:
        if self._working and self._cancel_path is not None:
            self._cancel_path.touch(exist_ok=True)
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("正在通知独立计算进程取消，请等待当前谱线层结束…")

    def _close_dialog(self) -> None:
        if self._working:
            self._cancel_calculation()
            return
        if self.result_data is not None:
            self.accept()
        else:
            self.reject()

    def reject(self) -> None:
        if self._working:
            self._cancel_calculation()
            return
        super().reject()
