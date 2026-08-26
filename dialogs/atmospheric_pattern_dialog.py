from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.atmospheric_pattern_manager import AtmosphericPatternManager
from dialogs.atmospheric_pattern_visualization import AtmosphericPatternVisualization


class AtmosphericPatternDialog(QWidget):
    """Embedded workbench for learned atmospheric regimes and optical-depth reuse."""

    opticalDepthReady = Signal(str, dict)
    statusMessage = Signal(str)
    exactCalculationRequested = Signal()
    exactProfileRequested = Signal(str, int)
    batchProfilesRequested = Signal(object)

    def __init__(self, project_dir: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.manager = AtmosphericPatternManager()
        self.last_prediction: dict[str, Any] | None = None
        self._build_ui()
        self.set_project_dir(self.project_dir)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "直接从全球GFS合并文件、NUCAPS产品或35层廓线学习典型大气状态。训练阶段只处理"
            "温度、气压和气体廓线，不调用HAPI；K-means确定最终代表模式后，才将少量代表廓线"
            "送入HAPI计算光学厚度。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("secondaryText")
        layout.addWidget(intro)

        training_group = QGroupBox("1. 建立大气状态模式库")
        training_form = QFormLayout(training_group)
        self.library_path = QLineEdit()
        library_row = QWidget()
        library_layout = QHBoxLayout(library_row)
        library_layout.setContentsMargins(0, 0, 0, 0)
        library_layout.addWidget(self.library_path, 1)
        browse_library = QPushButton("选择目录…")
        browse_library.clicked.connect(self._browse_library)
        library_layout.addWidget(browse_library)
        browse_library_file = QPushButton("选择合并文件…")
        browse_library_file.clicked.connect(self._browse_library_file)
        library_layout.addWidget(browse_library_file)
        training_form.addRow("全球廓线文件/目录", library_row)

        self.method = QComboBox()
        self.method.addItem("非线性自编码器 + K-means", "autoencoder")
        self.method.addItem("PCA/EOF + K-means", "pca")
        self.latent_dimension = QSpinBox()
        self.latent_dimension.setRange(1, 64)
        self.latent_dimension.setValue(8)
        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(2, 512)
        self.cluster_count.setValue(16)
        self.epochs = QSpinBox()
        self.epochs.setRange(20, 5000)
        self.epochs.setValue(300)
        self.maximum_samples = QSpinBox()
        self.maximum_samples.setRange(2, 100000)
        self.maximum_samples.setValue(5000)
        training_form.addRow("学习方法", self.method)
        training_form.addRow("潜在空间维数", self.latent_dimension)
        training_form.addRow("代表状态数量", self.cluster_count)
        training_form.addRow("自编码器训练轮数", self.epochs)
        training_form.addRow("最多读取廓线数", self.maximum_samples)

        train_row = QHBoxLayout()
        train_button = QPushButton("扫描样本并训练")
        train_button.setObjectName("primaryButton")
        train_button.clicked.connect(self._train)
        load_button = QPushButton("读取已有模型…")
        load_button.clicked.connect(self._load_model_dialog)
        train_row.addWidget(train_button)
        train_row.addWidget(load_button)
        train_row.addStretch(1)
        training_form.addRow("", train_row)
        self.model_summary = QLabel("尚未训练或读取模型。")
        self.model_summary.setWordWrap(True)
        self.model_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        training_form.addRow("模型状态", self.model_summary)
        self.pattern_table = QTableWidget(0, 6)
        self.pattern_table.setHorizontalHeaderLabels(
            ["模式", "纬度", "经度", "源索引", "观测时间", "源文件"]
        )
        self.pattern_table.setAlternatingRowColors(True)
        self.pattern_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pattern_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pattern_table.itemSelectionChanged.connect(
            self._synchronize_visualization_selection
        )
        self.pattern_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.pattern_table.setMinimumHeight(150)
        training_form.addRow("代表状态", self.pattern_table)
        layout.addWidget(training_group)

        visualization_group = QGroupBox("2. 训练结果可视化")
        visualization_layout = QVBoxLayout(visualization_group)
        self.visualization = AtmosphericPatternVisualization(self.manager)
        self.visualization.representativeRequested.connect(
            self._open_representative_index
        )
        visualization_layout.addWidget(self.visualization)
        layout.addWidget(visualization_group)

        optical_group = QGroupBox("3. 仅对最终代表模式计算光学厚度")
        optical_layout = QVBoxLayout(optical_group)
        optical_note = QLabel(
            "训练完成后软件会自动导出最终代表模式的35层CSV。可将全部模式送入HAPI工作台，"
            "统一设置一次波数范围并顺序计算；全球训练样本本身不会执行光学厚度计算。"
        )
        optical_note.setWordWrap(True)
        optical_note.setObjectName("secondaryText")
        optical_layout.addWidget(optical_note)
        representative_actions = QHBoxLayout()
        batch_representatives = QPushButton("批量计算全部最终模式")
        batch_representatives.setObjectName("primaryButton")
        batch_representatives.clicked.connect(self._request_all_representatives)
        open_representative = QPushButton("在HAPI中打开选中最终模式")
        open_representative.clicked.connect(self._open_selected_representative)
        export_representatives = QPushButton("导出全部最终模式廓线…")
        export_representatives.clicked.connect(self._export_representatives)
        representative_actions.addWidget(batch_representatives)
        representative_actions.addWidget(open_representative)
        representative_actions.addWidget(export_representatives)
        representative_actions.addStretch(1)
        optical_layout.addLayout(representative_actions)
        self.representative_status = QLabel("尚未生成最终代表模式。")
        self.representative_status.setWordWrap(True)
        optical_layout.addWidget(self.representative_status)
        layout.addWidget(optical_group)

        prediction_group = QGroupBox("可选：运行时新廓线匹配与光学厚度插值")
        prediction_form = QFormLayout(prediction_group)
        self.query_profile = QLineEdit()
        query_row = QWidget()
        query_layout = QHBoxLayout(query_row)
        query_layout.setContentsMargins(0, 0, 0, 0)
        query_layout.addWidget(self.query_profile, 1)
        browse_profile = QPushButton("选择廓线…")
        browse_profile.clicked.connect(self._browse_profile)
        query_layout.addWidget(browse_profile)
        prediction_form.addRow("35层廓线CSV", query_row)
        self.neighbor_count = QSpinBox()
        self.neighbor_count.setRange(1, 32)
        self.neighbor_count.setValue(3)
        prediction_form.addRow("参与插值的近邻数", self.neighbor_count)

        action_row = QHBoxLayout()
        match_button = QPushButton("匹配代表状态")
        match_button.clicked.connect(self._match)
        self.interpolate_button = QPushButton("生成并应用插值光学厚度")
        self.interpolate_button.setObjectName("primaryButton")
        self.interpolate_button.setEnabled(False)
        self.interpolate_button.clicked.connect(self._interpolate)
        exact_button = QPushButton("转到HAPI精算")
        exact_button.clicked.connect(self.exactCalculationRequested.emit)
        action_row.addWidget(match_button)
        action_row.addWidget(self.interpolate_button)
        action_row.addWidget(exact_button)
        action_row.addStretch(1)
        prediction_form.addRow("", action_row)

        self.prediction_summary = QLabel("尚未匹配廓线。")
        self.prediction_summary.setWordWrap(True)
        self.prediction_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        prediction_form.addRow("适用性判定", self.prediction_summary)
        self.representative_table = QTableWidget(0, 6)
        self.representative_table.setHorizontalHeaderLabels(
            ["代表状态", "距离", "权重", "纬度", "经度", "观测时间/来源"]
        )
        self.representative_table.setAlternatingRowColors(True)
        self.representative_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.representative_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.representative_table.setMinimumHeight(150)
        prediction_form.addRow("近邻代表廓线", self.representative_table)
        layout.addWidget(prediction_group)
        layout.addStretch(1)

    def set_project_dir(self, project_dir: str | Path) -> None:
        previous_default = self._default_library_source(self.project_dir)
        current = self.library_path.text().strip() if hasattr(self, "library_path") else ""
        self.project_dir = Path(project_dir).expanduser().resolve()
        new_default = self._default_library_source(self.project_dir)
        if not current or Path(current).expanduser() == previous_default:
            self.library_path.setText(str(new_default))

    @staticmethod
    def _default_library_source(project_dir: Path) -> Path:
        data_root = project_dir / "resources" / "data" / "atmospheric_profiles"
        candidates = sorted(
            data_root.glob("gfs_global_*/gfs_global_profiles_*.nc4")
        )
        if candidates:
            return candidates[-1]
        return project_dir / "config" / "atmosphere" / "hapi_optical_depth"

    def project_state(self) -> dict[str, Any]:
        return {
            "library_path": self.library_path.text().strip(),
            "model_path": str(self.manager.model_path) if self.manager.model_path else "",
            "query_profile": self.query_profile.text().strip(),
            "method": str(self.method.currentData()),
            "latent_dimension": self.latent_dimension.value(),
            "cluster_count": self.cluster_count.value(),
            "epochs": self.epochs.value(),
            "maximum_samples": self.maximum_samples.value(),
            "neighbor_count": self.neighbor_count.value(),
        }

    def restore_project_state(self, state: dict[str, Any]) -> bool:
        self.library_path.setText(str(state.get("library_path", self.library_path.text())))
        self.query_profile.setText(str(state.get("query_profile", "")))
        method_index = self.method.findData(str(state.get("method", "autoencoder")))
        self.method.setCurrentIndex(max(method_index, 0))
        self.latent_dimension.setValue(int(state.get("latent_dimension", 8)))
        self.cluster_count.setValue(int(state.get("cluster_count", 16)))
        self.epochs.setValue(int(state.get("epochs", 300)))
        self.maximum_samples.setValue(int(state.get("maximum_samples", 5000)))
        self.neighbor_count.setValue(int(state.get("neighbor_count", 3)))
        model_path = Path(str(state.get("model_path", ""))).expanduser()
        if not str(state.get("model_path", "")).strip():
            return False
        if not model_path.is_absolute():
            model_path = self.project_dir / model_path
        if not model_path.is_file():
            self.model_summary.setText(f"项目记录的模式模型不存在：{model_path}")
            return False
        self._load_model(model_path)
        return True

    def set_query_profile(self, path: str | Path) -> None:
        source = Path(path).expanduser().resolve()
        if source.is_file():
            self.query_profile.setText(str(source))

    def _browse_library(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择全球廓线或历史结果目录", self.library_path.text()
        )
        if path:
            self.library_path.setText(path)

    def _browse_library_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择全球大气廓线合并文件",
            self.library_path.text() or str(self.project_dir),
            "全球/卫星廓线 (*.nc *.nc4);;所有文件 (*)",
        )
        if path:
            self.library_path.setText(path)

    def _browse_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择35层大气廓线",
            self.query_profile.text() or str(self.project_dir),
            "大气廓线 CSV (*.csv);;所有文件 (*)",
        )
        if path:
            self.query_profile.setText(path)

    def _train(self) -> None:
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            summary = self.manager.fit_directory(
                self.library_path.text(),
                cluster_count=self.cluster_count.value(),
                latent_dimension=self.latent_dimension.value(),
                method=str(self.method.currentData()),
                epochs=self.epochs.value(),
                maximum_samples=self.maximum_samples.value(),
            )
            model_path = (
                self.project_dir
                / "config"
                / "atmosphere"
                / "profile_patterns"
                / "atmospheric_pattern_model.npz"
            )
            self.manager.save(model_path)
            representative_directory = model_path.parent / "representatives"
            exported = self.manager.export_representative_profiles(
                representative_directory
            )
            # 导出路径写入代表模式元数据后再次保存，项目重启可直接发送到HAPI。
            self.manager.save(model_path)
            self._show_model_summary(summary | {"model_path": str(model_path)})
            self._populate_representatives()
            self.visualization.refresh()
            self.representative_status.setText(
                f"已生成{len(exported)}个最终代表模式；仅这些模式需要计算光学厚度。\n"
                f"廓线目录：{representative_directory}"
            )
            self.last_prediction = None
            self.interpolate_button.setEnabled(False)
            message = (
                f"大气状态模式训练完成：{summary['training_sample_count']}条样本，"
                f"{summary['representative_count']}个代表状态。"
            )
            self.statusMessage.emit(message)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "大气状态模式训练失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "读取大气状态模式模型",
            str(self.project_dir),
            "大气状态模型 (*.npz);;所有文件 (*)",
        )
        if path:
            try:
                self._load_model(Path(path))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "模式模型读取失败", str(exc))

    def _load_model(self, path: Path) -> None:
        summary = self.manager.load(path)
        self._show_model_summary(summary)
        self._populate_representatives()
        self.visualization.refresh()
        self.last_prediction = None
        self.interpolate_button.setEnabled(False)
        self.statusMessage.emit(f"已读取大气状态模式模型：{path}")
        available = sum(
            1
            for metadata in self.manager.representative_metadata
            if Path(str(metadata.get("profile_file", ""))).is_file()
        )
        self.representative_status.setText(
            f"模型含{len(self.manager.representative_metadata)}个最终模式；"
            f"其中{available}个已导出为可直接计算的35层CSV；"
            + (
                f"全部代表模式光学厚度库可用（{summary['spectral_point_count']:,}点）。"
                if summary["has_optical_library"]
                else "尚未计算完整代表光学厚度库。"
            )
        )

    def _show_model_summary(self, summary: dict[str, Any]) -> None:
        method = "非线性自编码器" if summary["method"] == "autoencoder" else "PCA/EOF"
        optical = (
            f"另含{summary['spectral_point_count']:,}点历史光学厚度库"
            if summary["has_optical_library"]
            else "训练阶段未计算光学厚度（符合直接廓线学习流程）"
        )
        self.model_summary.setText(
            f"{method}；训练样本={summary['training_sample_count']}；"
            f"代表状态={summary['representative_count']}；潜在维数={summary['latent_dimension']}；"
            f"重建RMSE={summary['reconstruction_rmse']:.4g}；{optical}\n"
            f"模型文件：{summary.get('model_path', '')}"
        )

    def _populate_representatives(self) -> None:
        metadata_list = self.manager.representative_metadata
        self.pattern_table.setRowCount(len(metadata_list))
        for row, metadata in enumerate(metadata_list):
            values = [
                str(row + 1),
                self._format_number(metadata.get("latitude_deg")),
                self._format_number(metadata.get("longitude_deg")),
                str(metadata.get("for_index", "—")),
                str(metadata.get("observation_time_utc", "")),
                str(metadata.get("source_file", metadata.get("profile_file", ""))),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(values[-1])
                self.pattern_table.setItem(row, column, item)
        if metadata_list:
            self.pattern_table.selectRow(0)

    def _open_selected_representative(self) -> None:
        row = self.pattern_table.currentRow()
        if row < 0 or row >= len(self.manager.representative_metadata):
            QMessageBox.information(self, "打开代表廓线", "请先选择一个代表状态。")
            return
        self._open_representative_index(row)

    def _request_all_representatives(self) -> None:
        if not self.manager.is_fitted:
            QMessageBox.information(self, "批量计算代表模式", "请先训练或读取模型。")
            return
        try:
            profiles = self._ensure_representative_profiles()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "准备代表廓线失败", str(exc))
            return
        batch_profiles = [
            {
                "batch_index": index,
                "profile_path": str(path),
                "for_index": 0,
            }
            for index, path in enumerate(profiles)
        ]
        self.batchProfilesRequested.emit(batch_profiles)

    def _ensure_representative_profiles(self) -> list[Path]:
        metadata_list = self.manager.representative_metadata
        existing = [
            Path(str(metadata.get("profile_file", ""))).expanduser()
            for metadata in metadata_list
        ]
        if len(existing) == len(metadata_list) and all(path.is_file() for path in existing):
            return [path.resolve() for path in existing]
        output = (
            self.project_dir
            / "config"
            / "atmosphere"
            / "profile_patterns"
            / "representatives"
        )
        profiles = self.manager.export_representative_profiles(output)
        if self.manager.model_path is not None:
            self.manager.save(self.manager.model_path)
        self._populate_representatives()
        return profiles

    def apply_batch_optical_depth_result(self, batch_result: dict[str, Any]) -> bool:
        """Attach a complete representative HAPI batch to the learned model."""
        expected = len(self.manager.representative_metadata)
        results = [dict(item) for item in batch_result.get("results", [])]
        failures = [dict(item) for item in batch_result.get("failures", [])]
        indexed_results: dict[int, dict[str, Any]] = {}
        for result in results:
            try:
                index = int(result["batch_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < expected:
                indexed_results[index] = result

        if len(indexed_results) != expected or failures:
            failed_indices = {
                index + 1 for index in range(expected) if index not in indexed_results
            }
            for item in failures:
                try:
                    failed_index = int(item.get("batch_index", -1))
                except (TypeError, ValueError):
                    continue
                if failed_index >= 0:
                    failed_indices.add(failed_index + 1)
            failed_indices = sorted(failed_indices)
            suffix = "、".join(str(index) for index in failed_indices[:12]) or "未知"
            self.representative_status.setText(
                f"批量计算未完整完成：成功{len(indexed_results)}/{expected}个；"
                f"缺失或失败模式：{suffix}。成功结果已保留在HAPI结果目录，"
                "但不会替换当前模式插值库。"
            )
            self.statusMessage.emit(
                f"代表模式光学厚度批量计算部分完成：{len(indexed_results)}/{expected}。"
            )
            return False

        optical_paths = [
            str(indexed_results[index]["total_optical_depth_file"])
            for index in range(expected)
        ]
        summary = self.manager.set_representative_optical_depth_files(optical_paths)
        if self.manager.model_path is not None:
            self.manager.save(self.manager.model_path)
        self._show_model_summary(summary)
        self.last_prediction = None
        self.interpolate_button.setEnabled(False)
        output_directories = {
            str(result.get("output_directory", "")) for result in indexed_results.values()
        }
        common_parent = Path(optical_paths[0]).parent.parent
        self.representative_status.setText(
            f"全部{expected}个最终代表模式的光学厚度已计算并写入模式库；"
            f"统一光谱网格共{summary['spectral_point_count']:,}点。\n"
            f"结果根目录：{common_parent}（{len(output_directories)}个计算目录）"
        )
        self.statusMessage.emit(
            f"全部{expected}个代表模式光学厚度计算完成，模式插值库已可用。"
        )
        return True

    def _open_representative_index(self, row: int) -> None:
        if row < 0 or row >= len(self.manager.representative_metadata):
            return
        metadata = self.manager.representative_metadata[row]
        exported_profile = str(metadata.get("profile_file", "")).strip()
        if exported_profile and Path(exported_profile).is_file():
            self.exactProfileRequested.emit(exported_profile, 0)
            return
        source = str(metadata.get("source_file", "")).strip()
        try:
            for_index = int(metadata["for_index"])
        except (KeyError, TypeError, ValueError):
            for_index = -1
        if not source or for_index < 0 or not Path(source).is_file():
            QMessageBox.information(
                self,
                "打开代表廓线",
                "该模型没有可用的代表模式CSV或原始NUCAPS索引。请重新训练或先导出最终模式。",
            )
            return
        self.exactProfileRequested.emit(source, for_index)

    def _synchronize_visualization_selection(self) -> None:
        if not hasattr(self, "visualization"):
            return
        row = self.pattern_table.currentRow()
        if row >= 0:
            self.visualization.set_selected_mode(row)

    def _export_representatives(self) -> None:
        if not self.manager.is_fitted:
            QMessageBox.information(self, "导出代表状态", "请先训练或读取模型。")
            return
        default = (
            self.project_dir
            / "config"
            / "atmosphere"
            / "profile_patterns"
            / "representatives_export"
        )
        path = QFileDialog.getExistingDirectory(
            self, "选择最终代表模式导出目录", str(default)
        )
        if not path:
            return
        try:
            output = Path(path).expanduser().resolve()
            paths = self.manager.export_representative_profiles(output)
            if self.manager.model_path is not None:
                self.manager.save(self.manager.model_path)
            self._populate_representatives()
            self.visualization.refresh()
            self.representative_status.setText(
                f"已导出{len(paths)}个最终代表模式：{output}"
            )
            self.statusMessage.emit(f"最终代表模式廓线已导出：{output}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出代表状态失败", str(exc))

    def _match(self) -> None:
        try:
            result = self.manager.predict(
                self.query_profile.text(), self.neighbor_count.value()
            )
            self.last_prediction = result
            self._show_prediction(result)
        except Exception as exc:  # noqa: BLE001
            self.last_prediction = None
            self.interpolate_button.setEnabled(False)
            QMessageBox.warning(self, "大气状态匹配失败", str(exc))

    def _show_prediction(self, result: dict[str, Any]) -> None:
        outside = bool(result["out_of_distribution"])
        optical = "代表光学厚度库可用" if self.manager.has_optical_library else "代表光学厚度库不可用"
        if outside:
            verdict = "超出训练分布，禁止直接应用插值结果，建议执行HAPI精算"
        elif not self.manager.has_optical_library:
            verdict = "状态匹配有效，但缺少统一光谱网格，需执行HAPI精算"
        else:
            verdict = "处于训练分布内，可以生成分层光学厚度插值结果"
        self.prediction_summary.setText(
            f"{verdict}。置信度={float(result['confidence']):.1%}；"
            f"状态距离={float(result['center_distance']):.4g}，"
            f"阈值={float(result['distance_threshold']):.4g}；"
            f"廓线重建RMSE={float(result['reconstruction_rmse']):.4g}；{optical}。"
        )
        self.interpolate_button.setEnabled(
            self.manager.has_optical_library and not outside
        )
        representatives = list(result["representatives"])
        distances = result["distances"]
        weights = result["weights"]
        indices = result["neighbor_indices"]
        self.representative_table.setRowCount(len(representatives))
        for row, metadata in enumerate(representatives):
            source = str(metadata.get("observation_time_utc", "")) or str(
                metadata.get("profile_file", "")
            )
            values = [
                str(int(indices[row]) + 1),
                f"{float(distances[row]):.5g}",
                f"{float(weights[row]):.2%}",
                self._format_number(metadata.get("latitude_deg")),
                self._format_number(metadata.get("longitude_deg")),
                source,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(metadata.get("profile_file", "")))
                self.representative_table.setItem(row, column, item)

    @staticmethod
    def _format_number(value: Any) -> str:
        try:
            return f"{float(value):.5f}°"
        except (TypeError, ValueError):
            return "—"

    def _interpolate(self) -> None:
        if self.last_prediction is None:
            self._match()
        result = self.last_prediction
        if result is None:
            return
        if bool(result["requires_hapi"]):
            QMessageBox.warning(
                self,
                "不能应用插值结果",
                "当前廓线超出模型适用范围或缺少代表光学厚度，请转到HAPI进行精算。",
            )
            return
        output = (
            self.project_dir
            / "config"
            / "atmosphere"
            / "profile_patterns"
            / "predictions"
            / f"optical_depth_interpolated_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        )
        try:
            saved = self.manager.save_prediction(result, output)
            self.opticalDepthReady.emit(str(saved), dict(result))
            self.statusMessage.emit(f"已生成并应用模式插值光学厚度：{saved}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "光学厚度插值失败", str(exc))
