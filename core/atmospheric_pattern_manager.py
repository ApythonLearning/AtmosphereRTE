from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROFILE_FILENAME = "atmospheric_profile_35_layers.csv"
OPTICAL_DEPTH_FILENAME = "optical_depth_total.csv"
MANIFEST_FILENAME = "calculation_manifest.json"


@dataclass(frozen=True)
class ProfileRecord:
    path: Path
    columns: tuple[str, ...]
    values: np.ndarray
    metadata: dict[str, Any]


class AtmosphericPatternManager:
    """Learn representative atmospheric states and interpolate layered optical depth."""

    SCHEMA_VERSION = 3

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.method = ""
        self.feature_columns: tuple[str, ...] = ()
        self.layer_count = 0
        self.feature_mean = np.empty(0)
        self.feature_scale = np.empty(0)
        self.encoder_weight = np.empty((0, 0))
        self.encoder_bias = np.empty(0)
        self.decoder_weight = np.empty((0, 0))
        self.decoder_bias = np.empty(0)
        self.cluster_centers = np.empty((0, 0))
        self.representative_scores = np.empty((0, 0))
        self.representative_indices = np.empty(0, dtype=int)
        self.representative_metadata: list[dict[str, Any]] = []
        self.representative_columns: tuple[str, ...] = ()
        self.representative_values = np.empty((0, 0, 0))
        self.training_scores = np.empty((0, 0))
        self.training_labels = np.empty(0, dtype=int)
        self.training_latitudes = np.empty(0)
        self.training_longitudes = np.empty(0)
        self.training_times_utc = np.empty(0, dtype="U1")
        self.cluster_counts = np.empty(0, dtype=int)
        self.cluster_mean_distances = np.empty(0)
        self.cluster_profile_mean = np.empty((0, 0, 0))
        self.cluster_profile_p10 = np.empty((0, 0, 0))
        self.cluster_profile_p90 = np.empty((0, 0, 0))
        self.training_loss_history = np.empty(0)
        self.pca_explained_variance_ratio = np.empty(0)
        self.distance_threshold = 0.0
        self.training_sample_count = 0
        self.reconstruction_rmse = 0.0
        self.training_iterations = 0
        self.wavenumber_cm = np.empty(0)
        self.representative_tau = np.empty((0, 0, 0), dtype=np.float32)
        self.model_path: Path | None = None

    @property
    def is_fitted(self) -> bool:
        return self.training_sample_count >= 2 and self.representative_scores.size > 0

    @property
    def has_optical_library(self) -> bool:
        return (
            self.is_fitted
            and self.wavenumber_cm.size > 0
            and self.representative_tau.shape[0] == self.representative_scores.shape[0]
        )

    @property
    def has_visualization_data(self) -> bool:
        return (
            self.is_fitted
            and self.training_scores.ndim == 2
            and self.training_scores.shape[0] == self.training_sample_count
            and self.training_labels.size == self.training_sample_count
        )

    @classmethod
    def discover_profiles(cls, source: str | Path) -> list[Path]:
        root = Path(source).expanduser().resolve()
        if root.is_file():
            return [root] if root.name.lower() == PROFILE_FILENAME.lower() else []
        if not root.exists():
            raise FileNotFoundError(f"训练样本目录不存在：{root}")
        return sorted(root.rglob(PROFILE_FILENAME))

    @classmethod
    def read_profile(cls, path: str | Path, for_index: int = 0) -> ProfileRecord:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"廓线文件不存在：{source}")
        if source.suffix.lower() in {".nc", ".nc4"}:
            from core.hapi_optical_depth_manager import NucapsAtmosphericProfileReader

            profile = NucapsAtmosphericProfileReader.read(source, int(for_index))
            return cls._record_from_layered_profile(profile, "NUCAPS")
        if source.suffix.lower() != ".csv":
            raise ValueError("待匹配廓线仅支持35层CSV或NUCAPS NetCDF（.nc/.nc4）。")
        with source.open("r", encoding="utf-8-sig") as stream:
            header = stream.readline().strip()
        columns = tuple(item.strip() for item in header.split(","))
        values = np.loadtxt(source, delimiter=",", skiprows=1, ndmin=2)
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise ValueError(f"廓线文件列数与表头不一致：{source}")
        if values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError(f"廓线文件包含无效数据：{source}")
        required = {"pressure(hPa)", "temperature(K)"}
        if not required.issubset(columns):
            raise ValueError(f"廓线文件缺少 pressure(hPa) 或 temperature(K)：{source}")
        metadata: dict[str, Any] = {"profile_file": str(source)}
        manifest_path = source.parent / MANIFEST_FILENAME
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                metadata.update(dict(manifest.get("profile", {})))
                metadata["manifest_file"] = str(manifest_path)
            except (OSError, ValueError, TypeError):
                pass
        return ProfileRecord(source, columns, values, metadata)

    def fit_directory(
        self,
        source: str | Path,
        *,
        cluster_count: int = 16,
        latent_dimension: int = 8,
        method: str = "autoencoder",
        epochs: int = 300,
        random_seed: int = 42,
        maximum_samples: int = 5000,
    ) -> dict[str, Any]:
        paths = self.discover_profiles(source)
        if len(paths) >= 2:
            records = [self.read_profile(path) for path in paths[:maximum_samples]]
        else:
            records = self._read_netcdf_directory(
                source, maximum_samples, random_seed=random_seed
            )
        if len(records) < 2:
            raise ValueError(
                "至少需要2条35层CSV廓线，或包含有效FOR的NUCAPS NetCDF文件。"
            )
        return self.fit(
            records,
            cluster_count=cluster_count,
            latent_dimension=latent_dimension,
            method=method,
            epochs=epochs,
            random_seed=random_seed,
        )

    @classmethod
    def _read_netcdf_directory(
        cls,
        source: str | Path,
        maximum_samples: int,
        *,
        random_seed: int,
    ) -> list[ProfileRecord]:
        from core.gfs_atmospheric_profile_reader import (
            GfsGlobalAtmosphericProfileReader,
        )

        root = Path(source).expanduser().resolve()
        if root.is_file():
            candidates = [root] if root.suffix.lower() in {".nc", ".nc4"} else []
        elif root.exists():
            candidates = sorted(
                path
                for suffix in ("*.nc", "*.nc4")
                for path in root.rglob(suffix)
            )
        else:
            raise FileNotFoundError(f"训练样本路径不存在：{root}")
        if not candidates:
            raise ValueError("所选路径中没有35层CSV、NUCAPS或GFS NetCDF廓线。")

        gfs_files = [
            path for path in candidates if GfsGlobalAtmosphericProfileReader.is_supported(path)
        ]
        if gfs_files:
            per_file = max(2, int(np.ceil(maximum_samples / len(gfs_files))))
            profiles = []
            for file_index, path in enumerate(gfs_files):
                profiles.extend(
                    GfsGlobalAtmosphericProfileReader.read_sampled(
                        path,
                        per_file,
                        random_seed=int(random_seed) + file_index,
                    )
                )
            if len(profiles) > maximum_samples:
                selected = np.linspace(0, len(profiles) - 1, maximum_samples, dtype=int)
                profiles = [profiles[int(index)] for index in selected]
            return [cls._record_from_layered_profile(profile, "GFS") for profile in profiles]
        return cls._read_nucaps_directory(root, maximum_samples)

    @staticmethod
    def _record_from_layered_profile(profile: Any, source_type: str) -> ProfileRecord:
        columns = (
            "altitude_mid(km)",
            "pressure(hPa)",
            "temperature(K)",
            *(f"column_{name}(molec_cm-2)" for name in profile.gas_names),
        )
        values = np.column_stack(
            [
                profile.altitude_mid_km,
                profile.pressure_hpa,
                profile.temperature_k,
                *(profile.gas_columns_molec_cm2[name] for name in profile.gas_names),
            ]
        )
        metadata = {
            "source_type": source_type,
            "source_file": str(profile.source_path),
            "for_index": int(profile.for_index),
            "observation_time_utc": profile.observation_time_utc,
            "latitude_deg": profile.latitude_deg,
            "longitude_deg": profile.longitude_deg,
            "quality_flag": profile.quality_flag,
            "profile_file": "",
        }
        for attribute, key in (
            ("gfs_time_index", "time_index"),
            ("gfs_latitude_index", "latitude_index"),
            ("gfs_longitude_index", "longitude_index"),
            ("terrain_height_m", "terrain_height_m"),
        ):
            if hasattr(profile, attribute):
                metadata[key] = getattr(profile, attribute)
        return ProfileRecord(Path(profile.source_path), columns, values, metadata)

    @staticmethod
    def _read_nucaps_directory(
        source: str | Path, maximum_samples: int
    ) -> list[ProfileRecord]:
        from core.hapi_optical_depth_manager import NucapsAtmosphericProfileReader

        root = Path(source).expanduser().resolve()
        if root.is_file():
            candidates = [root] if root.suffix.lower() in {".nc", ".nc4"} else []
        else:
            candidates = sorted(
                path
                for suffix in ("*.nc", "*.nc4")
                for path in root.rglob(suffix)
            )
        available: list[tuple[Path, int]] = []
        for path in candidates:
            inspection = NucapsAtmosphericProfileReader.inspect(path)
            quality = np.asarray(inspection["quality_flag"], dtype=float)
            levels = np.asarray(inspection["valid_level_count"], dtype=int)
            valid = np.flatnonzero(np.isfinite(quality) & (quality == 0) & (levels >= 2))
            if not valid.size:
                valid = np.flatnonzero(levels >= 2)
            available.extend((path, int(index)) for index in valid)
        if len(available) > maximum_samples:
            selected = np.linspace(0, len(available) - 1, maximum_samples, dtype=int)
            available = [available[int(index)] for index in selected]

        records: list[ProfileRecord] = []
        for path, for_index in available:
            profile = NucapsAtmosphericProfileReader.read(path, for_index)
            records.append(AtmosphericPatternManager._record_from_layered_profile(profile, "NUCAPS"))
        return records

    def fit(
        self,
        records: Iterable[ProfileRecord],
        *,
        cluster_count: int = 16,
        latent_dimension: int = 8,
        method: str = "autoencoder",
        epochs: int = 300,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        samples = list(records)
        if len(samples) < 2:
            raise ValueError("至少需要2条廓线才能训练大气状态模式。")
        layer_count = samples[0].values.shape[0]
        common = set(samples[0].columns)
        for record in samples[1:]:
            if record.values.shape[0] != layer_count:
                raise ValueError("训练廓线的层数必须一致。")
            common.intersection_update(record.columns)
        gas_columns = sorted(name for name in common if name.startswith("column_"))
        if all(str(record.metadata.get("source_type", "")).upper() == "GFS" for record in samples):
            # GFS只真实提供H2O和O3；CO2/CH4/N2O/CO固定背景仅供最终HAPI计算，
            # 不让这些由气压推导出的重复列影响全球模式聚类。
            observed_gfs_gases = {
                "column_H2O(molec_cm-2)",
                "column_O3(molec_cm-2)",
            }
            gas_columns = [name for name in gas_columns if name in observed_gfs_gases]
        feature_columns = ("pressure(hPa)", "temperature(K)", *gas_columns)
        if not set(feature_columns).issubset(common):
            raise ValueError("训练廓线缺少共同的温度或气压字段。")

        matrix = np.vstack([
            self._profile_feature_vector(record, feature_columns, layer_count)
            for record in samples
        ])
        feature_mean = np.mean(matrix, axis=0)
        feature_scale = np.std(matrix, axis=0)
        feature_scale = np.where(feature_scale > 1.0e-10, feature_scale, 1.0)
        standardized = (matrix - feature_mean) / feature_scale
        maximum_latent = max(1, min(standardized.shape[0] - 1, standardized.shape[1]))
        latent_dimension = max(1, min(int(latent_dimension), maximum_latent))
        method = str(method).strip().lower()
        if method not in {"pca", "autoencoder"}:
            raise ValueError("模式学习方法必须为 pca 或 autoencoder。")

        _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
        variance = singular_values**2
        explained_variance_ratio = variance / max(float(np.sum(variance)), 1.0e-30)
        pca_components = vt[:latent_dimension]
        if method == "pca":
            encoder_weight = pca_components.T
            encoder_bias = np.zeros(latent_dimension)
            decoder_weight = pca_components
            decoder_bias = np.zeros(standardized.shape[1])
            scores = standardized @ encoder_weight
            reconstruction = scores @ decoder_weight
            iterations = 0
            loss_history = np.empty(0)
        else:
            (
                encoder_weight,
                encoder_bias,
                decoder_weight,
                decoder_bias,
                iterations,
                loss_history,
            ) = self._train_autoencoder(
                standardized,
                pca_components,
                max(1, int(epochs)),
                int(random_seed),
            )
            scores = np.tanh(standardized @ encoder_weight + encoder_bias)
            reconstruction = scores @ decoder_weight + decoder_bias

        cluster_count = max(1, min(int(cluster_count), len(samples)))
        centers, labels = self._kmeans(scores, cluster_count, int(random_seed))
        representative_indices: list[int] = []
        for cluster_index in range(cluster_count):
            members = np.flatnonzero(labels == cluster_index)
            distances = np.linalg.norm(scores[members] - centers[cluster_index], axis=1)
            representative_indices.append(int(members[int(np.argmin(distances))]))
        representative_indices_array = np.asarray(representative_indices, dtype=int)
        cluster_counts = np.bincount(labels, minlength=cluster_count).astype(int)
        assigned_distances = np.linalg.norm(scores - centers[labels], axis=1) / np.sqrt(
            max(latent_dimension, 1)
        )
        cluster_mean_distances = np.asarray(
            [
                float(np.mean(assigned_distances[labels == cluster_index]))
                for cluster_index in range(cluster_count)
            ],
            dtype=float,
        )
        nearest_center = np.min(
            np.linalg.norm(scores[:, None, :] - centers[None, :, :], axis=2), axis=1
        ) / np.sqrt(max(latent_dimension, 1))
        threshold = max(float(np.percentile(nearest_center, 99.0)) * 1.25, 1.0e-6)

        representative_columns = tuple(
            name for name in samples[0].columns if name in common
        )
        sample_column_indices = [
            {name: index for index, name in enumerate(sample.columns)}
            for sample in samples
        ]
        all_profile_values = np.stack(
            [
                sample.values[
                    :, [indices[name] for name in representative_columns]
                ]
                for sample, indices in zip(samples, sample_column_indices)
            ]
        )
        cluster_profile_mean = np.stack(
            [np.mean(all_profile_values[labels == index], axis=0) for index in range(cluster_count)]
        )
        cluster_profile_p10 = np.stack(
            [np.percentile(all_profile_values[labels == index], 10.0, axis=0) for index in range(cluster_count)]
        )
        cluster_profile_p90 = np.stack(
            [np.percentile(all_profile_values[labels == index], 90.0, axis=0) for index in range(cluster_count)]
        )

        self.clear()
        self.method = method
        self.feature_columns = feature_columns
        self.layer_count = layer_count
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.encoder_weight = encoder_weight
        self.encoder_bias = encoder_bias
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias
        self.cluster_centers = centers
        self.representative_scores = scores[representative_indices_array]
        self.representative_indices = representative_indices_array
        self.representative_metadata = [
            dict(samples[index].metadata) for index in representative_indices
        ]
        self.representative_columns = representative_columns
        representative_column_indices = [sample_column_indices[index] for index in representative_indices]
        self.representative_values = np.stack(
            [
                samples[sample_index].values[
                    :, [indices[name] for name in self.representative_columns]
                ]
                for sample_index, indices in zip(
                    representative_indices, representative_column_indices
                )
            ]
        )
        self.training_scores = np.asarray(scores, dtype=np.float32)
        self.training_labels = np.asarray(labels, dtype=int)
        self.training_latitudes = np.asarray(
            [self._metadata_float(sample.metadata, "latitude_deg") for sample in samples],
            dtype=float,
        )
        self.training_longitudes = np.asarray(
            [self._metadata_float(sample.metadata, "longitude_deg") for sample in samples],
            dtype=float,
        )
        self.training_times_utc = np.asarray(
            [str(sample.metadata.get("observation_time_utc", "")) for sample in samples],
            dtype=str,
        )
        self.cluster_counts = cluster_counts
        self.cluster_mean_distances = cluster_mean_distances
        self.cluster_profile_mean = cluster_profile_mean
        self.cluster_profile_p10 = cluster_profile_p10
        self.cluster_profile_p90 = cluster_profile_p90
        self.training_loss_history = np.asarray(loss_history, dtype=float)
        self.pca_explained_variance_ratio = np.asarray(
            explained_variance_ratio, dtype=float
        )
        self.distance_threshold = threshold
        self.training_sample_count = len(samples)
        self.reconstruction_rmse = float(np.sqrt(np.mean((reconstruction - standardized) ** 2)))
        self.training_iterations = iterations
        self._load_representative_optical_depth(samples, representative_indices)
        return self.summary()

    def export_representative_profiles(self, output_directory: str | Path) -> list[Path]:
        """Export only the final learned atmospheric modes as reusable 35-layer CSVs."""

        if not self.is_fitted or self.representative_values.ndim != 3:
            raise RuntimeError("当前模型没有可导出的最终代表廓线。")
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, (values, metadata) in enumerate(
            zip(self.representative_values, self.representative_metadata), start=1
        ):
            path = output / f"atmospheric_pattern_{index:03d}_35_layers.csv"
            np.savetxt(
                path,
                values,
                delimiter=",",
                header=",".join(self.representative_columns),
                comments="",
                fmt="%.8e",
            )
            sidecar = path.with_suffix(".json")
            exported_metadata = dict(metadata)
            exported_metadata.update(
                {
                    "pattern_index": index,
                    "profile_file": str(path),
                    "model_path": str(self.model_path or ""),
                }
            )
            sidecar.write_text(
                json.dumps(
                    exported_metadata,
                    ensure_ascii=False,
                    indent=2,
                    default=self._json_default,
                ),
                encoding="utf-8",
            )
            metadata["profile_file"] = str(path)
            metadata["pattern_index"] = index
            paths.append(path)
        (output / "representative_profiles.json").write_text(
            json.dumps(
                self.representative_metadata,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            ),
            encoding="utf-8",
        )
        return paths

    @staticmethod
    def _train_autoencoder(
        values: np.ndarray,
        pca_components: np.ndarray,
        epochs: int,
        random_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        del random_seed  # PCA initialization makes training deterministic.
        latent = pca_components.shape[0]
        encoder_weight = pca_components.T * 0.25
        encoder_bias = np.zeros(latent)
        decoder_weight = pca_components / 0.25
        decoder_bias = np.zeros(values.shape[1])
        parameters = [encoder_weight, encoder_bias, decoder_weight, decoder_bias]
        first_moments = [np.zeros_like(item) for item in parameters]
        second_moments = [np.zeros_like(item) for item in parameters]
        learning_rate = 0.01
        best_loss = np.inf
        stale = 0
        completed = 0
        loss_history: list[float] = []
        for iteration in range(1, epochs + 1):
            latent_values = np.tanh(values @ encoder_weight + encoder_bias)
            reconstructed = latent_values @ decoder_weight + decoder_bias
            error = reconstructed - values
            loss = float(np.mean(error * error))
            loss_history.append(loss)
            output_gradient = 2.0 * error / error.size
            gradients = [
                values.T
                @ ((output_gradient @ decoder_weight.T) * (1.0 - latent_values**2)),
                np.sum(
                    (output_gradient @ decoder_weight.T) * (1.0 - latent_values**2),
                    axis=0,
                ),
                latent_values.T @ output_gradient,
                np.sum(output_gradient, axis=0),
            ]
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                gradient = np.clip(gradient + 1.0e-7 * parameter, -1.0, 1.0)
                first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
                second_moments[index] = 0.999 * second_moments[index] + 0.001 * gradient**2
                first_hat = first_moments[index] / (1.0 - 0.9**iteration)
                second_hat = second_moments[index] / (1.0 - 0.999**iteration)
                parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + 1.0e-8)
            completed = iteration
            if loss < best_loss - 1.0e-7:
                best_loss = loss
                stale = 0
            else:
                stale += 1
                if stale >= 40:
                    break
        return (*parameters, completed, np.asarray(loss_history, dtype=float))

    @staticmethod
    def _kmeans(
        scores: np.ndarray, cluster_count: int, random_seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        generator = np.random.default_rng(random_seed)
        centers = [scores[int(generator.integers(0, scores.shape[0]))].copy()]
        for _ in range(1, cluster_count):
            squared = np.min(
                np.sum((scores[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2),
                axis=1,
            )
            total = float(np.sum(squared))
            if total <= 1.0e-20:
                unused = [i for i in range(scores.shape[0]) if not any(np.array_equal(scores[i], c) for c in centers)]
                centers.append(scores[unused[0] if unused else len(centers) % scores.shape[0]].copy())
            else:
                centers.append(scores[int(generator.choice(scores.shape[0], p=squared / total))].copy())
        center_array = np.asarray(centers)
        labels = np.zeros(scores.shape[0], dtype=int)
        for _ in range(100):
            distances = np.linalg.norm(scores[:, None, :] - center_array[None, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            new_centers = center_array.copy()
            for index in range(cluster_count):
                members = scores[new_labels == index]
                if members.size:
                    new_centers[index] = np.mean(members, axis=0)
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, center_array):
                labels = new_labels
                center_array = new_centers
                break
            labels = new_labels
            center_array = new_centers
        return center_array, labels

    @staticmethod
    def _profile_feature_vector(
        record: ProfileRecord, feature_columns: tuple[str, ...], layer_count: int
    ) -> np.ndarray:
        if record.values.shape[0] != layer_count:
            raise ValueError("待匹配廓线的层数与模型不一致。")
        indices = {name: index for index, name in enumerate(record.columns)}
        missing = [name for name in feature_columns if name not in indices]
        if missing:
            raise ValueError(f"廓线缺少模型所需字段：{', '.join(missing)}")
        transformed: list[np.ndarray] = []
        for name in feature_columns:
            values = np.asarray(record.values[:, indices[name]], dtype=float)
            if name == "pressure(hPa)":
                values = np.log(np.maximum(values, 1.0e-12))
            elif name.startswith("column_"):
                values = np.log1p(np.maximum(values, 0.0))
            transformed.append(values)
        result = np.concatenate(transformed)
        if not np.isfinite(result).all():
            raise ValueError(f"廓线包含非有限特征：{record.path}")
        return result

    def _encode(self, feature_vector: np.ndarray) -> tuple[np.ndarray, float]:
        standardized = (feature_vector - self.feature_mean) / self.feature_scale
        if self.method == "autoencoder":
            score = np.tanh(standardized @ self.encoder_weight + self.encoder_bias)
        else:
            score = standardized @ self.encoder_weight + self.encoder_bias
        reconstructed = score @ self.decoder_weight + self.decoder_bias
        rmse = float(np.sqrt(np.mean((reconstructed - standardized) ** 2)))
        return score, rmse

    def predict(
        self,
        profile: ProfileRecord | str | Path,
        neighbor_count: int = 3,
        *,
        for_index: int = 0,
    ) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("尚未训练或读取大气状态模式模型。")
        record = (
            self.read_profile(profile, for_index=for_index)
            if not isinstance(profile, ProfileRecord)
            else profile
        )
        vector = self._profile_feature_vector(record, self.feature_columns, self.layer_count)
        score, reconstruction_rmse = self._encode(vector)
        representative_distances = np.linalg.norm(self.representative_scores - score, axis=1)
        count = max(1, min(int(neighbor_count), representative_distances.size))
        selected = np.argsort(representative_distances)[:count]
        selected_distances = representative_distances[selected] / np.sqrt(max(score.size, 1))
        if selected_distances[0] <= 1.0e-12:
            weights = np.zeros(count)
            weights[0] = 1.0
        else:
            inverse = 1.0 / np.maximum(selected_distances, 1.0e-12) ** 2
            weights = inverse / np.sum(inverse)
        center_distance = float(
            np.min(np.linalg.norm(self.cluster_centers - score, axis=1))
            / np.sqrt(max(score.size, 1))
        )
        out_of_distribution = center_distance > self.distance_threshold
        confidence = float(np.exp(-center_distance / max(self.distance_threshold, 1.0e-12)))
        result: dict[str, Any] = {
            "profile_file": str(record.path),
            "profile_for_index": int(record.metadata.get("for_index", for_index)),
            "profile_metadata": dict(record.metadata),
            "neighbor_indices": selected.astype(int),
            "distances": selected_distances,
            "weights": weights,
            "representatives": [self.representative_metadata[int(index)] for index in selected],
            "center_distance": center_distance,
            "distance_threshold": self.distance_threshold,
            "confidence": confidence,
            "reconstruction_rmse": reconstruction_rmse,
            "out_of_distribution": out_of_distribution,
            "requires_hapi": out_of_distribution or not self.has_optical_library,
        }
        if self.has_optical_library:
            selected_tau = self.representative_tau[selected]
            log_tau = np.log1p(np.maximum(selected_tau, 0.0))
            result["wavenumber_cm"] = self.wavenumber_cm.copy()
            result["optical_depth_layers"] = np.maximum(
                np.expm1(np.tensordot(weights, log_tau, axes=(0, 0))), 0.0
            )
        return result

    def _load_representative_optical_depth(
        self, samples: list[ProfileRecord], representative_indices: list[int]
    ) -> None:
        wavenumber: np.ndarray | None = None
        optical: np.ndarray | None = None
        for position, index in enumerate(representative_indices):
            if samples[index].path.name.lower() != PROFILE_FILENAME.lower():
                return
            path = samples[index].path.parent / OPTICAL_DEPTH_FILENAME
            if not path.is_file():
                return
            values = np.loadtxt(
                path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float32
            )
            if values.shape[1] != self.layer_count + 1:
                return
            current_wavenumber = np.asarray(values[:, 0], dtype=float)
            if wavenumber is None:
                wavenumber = current_wavenumber
                optical = np.empty(
                    (
                        len(representative_indices),
                        self.layer_count,
                        current_wavenumber.size,
                    ),
                    dtype=np.float32,
                )
            elif current_wavenumber.shape != wavenumber.shape or not np.allclose(
                current_wavenumber, wavenumber, rtol=0.0, atol=1.0e-9
            ):
                return
            assert optical is not None
            optical[position] = np.maximum(values[:, 1:].T, 0.0)
        if wavenumber is not None and optical is not None:
            self.wavenumber_cm = wavenumber
            self.representative_tau = optical

    def set_representative_optical_depth_files(
        self, optical_depth_files: Iterable[str | Path]
    ) -> dict[str, Any]:
        """Attach one HAPI optical-depth result to every representative pattern."""
        if not self.is_fitted:
            raise RuntimeError("尚未训练或读取大气状态模式模型。")
        paths = [Path(value).expanduser().resolve() for value in optical_depth_files]
        expected_count = int(self.representative_scores.shape[0])
        if len(paths) != expected_count:
            raise ValueError(
                f"代表光学厚度数量不一致：需要{expected_count}个，实际{len(paths)}个。"
            )

        wavenumber: np.ndarray | None = None
        optical: np.ndarray | None = None
        for index, path in enumerate(paths, start=1):
            if not path.is_file():
                raise FileNotFoundError(f"第{index}个代表光学厚度文件不存在：{path}")
            values = np.loadtxt(
                path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float32
            )
            if values.ndim != 2 or values.shape[1] != self.layer_count + 1:
                raise ValueError(
                    f"第{index}个代表光学厚度必须包含波数列和{self.layer_count}个大气层：{path}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"第{index}个代表光学厚度包含无效数值：{path}")
            current_wavenumber = np.asarray(values[:, 0], dtype=float)
            if current_wavenumber.size < 2 or np.any(np.diff(current_wavenumber) <= 0.0):
                raise ValueError(f"第{index}个代表光学厚度的波数网格无效：{path}")
            if wavenumber is None:
                wavenumber = current_wavenumber
                optical = np.empty(
                    (expected_count, self.layer_count, current_wavenumber.size),
                    dtype=np.float32,
                )
            elif current_wavenumber.shape != wavenumber.shape or not np.allclose(
                current_wavenumber, wavenumber, rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(
                    f"第{index}个代表光学厚度与前序结果的波数网格不一致：{path}"
                )
            assert optical is not None
            optical[index - 1] = np.maximum(values[:, 1:].T, 0.0)

        if wavenumber is None or optical is None:
            raise ValueError("没有可加载的代表光学厚度文件。")
        self.wavenumber_cm = wavenumber
        self.representative_tau = optical
        for metadata, path in zip(self.representative_metadata, paths):
            metadata["optical_depth_file"] = str(path)
        return self.summary()

    def save_prediction(self, result: dict[str, Any], path: str | Path) -> Path:
        if "optical_depth_layers" not in result:
            raise ValueError("模型没有可用于插值的代表光学厚度库。")
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        tau = np.asarray(result["optical_depth_layers"], dtype=float)
        wavenumber = np.asarray(result["wavenumber_cm"], dtype=float)
        layer_headers = [f"layer_{index + 1:02d}" for index in range(tau.shape[0])]
        np.savetxt(
            output,
            np.column_stack([wavenumber, tau.T]),
            delimiter=",",
            header="wavenumber(cm-1)," + ",".join(layer_headers),
            comments="",
            fmt="%.8e",
        )
        metadata = {
            key: value
            for key, value in result.items()
            if key not in {"wavenumber_cm", "optical_depth_layers"}
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8",
        )
        return output

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted:
            raise RuntimeError("尚未训练模型。")
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": self.method,
            "feature_columns": list(self.feature_columns),
            "layer_count": self.layer_count,
            "distance_threshold": self.distance_threshold,
            "training_sample_count": self.training_sample_count,
            "reconstruction_rmse": self.reconstruction_rmse,
            "training_iterations": self.training_iterations,
            "representative_metadata": self.representative_metadata,
            "representative_columns": list(self.representative_columns),
        }
        with output.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                feature_mean=self.feature_mean,
                feature_scale=self.feature_scale,
                encoder_weight=self.encoder_weight,
                encoder_bias=self.encoder_bias,
                decoder_weight=self.decoder_weight,
                decoder_bias=self.decoder_bias,
                cluster_centers=self.cluster_centers,
                representative_scores=self.representative_scores,
                representative_indices=self.representative_indices,
                representative_values=self.representative_values,
                training_scores=self.training_scores,
                training_labels=self.training_labels,
                training_latitudes=self.training_latitudes,
                training_longitudes=self.training_longitudes,
                training_times_utc=self.training_times_utc,
                cluster_counts=self.cluster_counts,
                cluster_mean_distances=self.cluster_mean_distances,
                cluster_profile_mean=self.cluster_profile_mean,
                cluster_profile_p10=self.cluster_profile_p10,
                cluster_profile_p90=self.cluster_profile_p90,
                training_loss_history=self.training_loss_history,
                pca_explained_variance_ratio=self.pca_explained_variance_ratio,
                wavenumber_cm=self.wavenumber_cm,
                representative_tau=self.representative_tau,
            )
        self.model_path = output
        return output

    def load(self, path: str | Path) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            schema_version = int(metadata.get("schema_version", 0))
            if schema_version not in {1, 2, self.SCHEMA_VERSION}:
                raise ValueError("不支持的大气状态模式模型版本。")
            arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "metadata_json"}
        self.clear()
        self.method = str(metadata["method"])
        self.feature_columns = tuple(metadata["feature_columns"])
        self.layer_count = int(metadata["layer_count"])
        self.distance_threshold = float(metadata["distance_threshold"])
        self.training_sample_count = int(metadata["training_sample_count"])
        self.reconstruction_rmse = float(metadata["reconstruction_rmse"])
        self.training_iterations = int(metadata.get("training_iterations", 0))
        self.representative_metadata = list(metadata["representative_metadata"])
        self.representative_columns = tuple(metadata.get("representative_columns", []))
        for name, value in arrays.items():
            setattr(self, name, value)
        self.representative_indices = self.representative_indices.astype(int)
        self.training_labels = self.training_labels.astype(int)
        self.cluster_counts = self.cluster_counts.astype(int)
        self.model_path = source
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "training_sample_count": self.training_sample_count,
            "layer_count": self.layer_count,
            "feature_count": len(self.feature_columns),
            "latent_dimension": int(self.representative_scores.shape[1]) if self.representative_scores.ndim == 2 else 0,
            "representative_count": int(self.representative_scores.shape[0]) if self.representative_scores.ndim == 2 else 0,
            "reconstruction_rmse": self.reconstruction_rmse,
            "training_iterations": self.training_iterations,
            "distance_threshold": self.distance_threshold,
            "has_optical_library": self.has_optical_library,
            "has_representative_profiles": bool(self.representative_values.size),
            "has_visualization_data": self.has_visualization_data,
            "geolocated_sample_count": int(
                np.count_nonzero(
                    np.isfinite(self.training_latitudes)
                    & np.isfinite(self.training_longitudes)
                )
            ),
            "spectral_point_count": int(self.wavenumber_cm.size),
            "model_path": str(self.model_path) if self.model_path else "",
        }

    @staticmethod
    def _metadata_float(metadata: dict[str, Any], key: str) -> float:
        try:
            value = float(metadata.get(key, np.nan))
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"不可序列化的对象：{type(value).__name__}")
