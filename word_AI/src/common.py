"""Word GRU modeling shared utilities."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - runtime dependency guard
    tf = None


DEFAULT_SEQ_LEN = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 42
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_NOISE_STD = 0.01
DEFAULT_TIME_JITTER = 2
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_EPOCHS = 80
DEFAULT_PATIENCE = 12
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_THRESHOLD = 0.75


@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    label_map: dict[str, Any]
    meta: dict[str, Any]
    feature_dim: int
    num_classes: int
    seq_len: int


def ensure_tensorflow() -> Any:
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed in the current environment. "
            "Install tensorflow or tensorflow-cpu before running GRU training."
        )
    return tf


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_word_data_dir() -> Path:
    # 우선순위 1: 현재 스크립트 기준 상위 디렉토리를 순회하며 실제 폴더를 찾는다.
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "전처리 완료 데이터(단어)"
        if candidate.exists() and candidate.is_dir():
            return candidate

    # 우선순위 2: 기존 상대 경로 fallback (호환성 유지)
    return get_project_root() / "전처리 완료 데이터(단어)"


def get_artifact_dir() -> Path:
    # Allow per-run override so model-specific scripts can place artifacts
    # inside their own folder (e.g., word_AI/GRU/artifacts).
    env = os.environ.get("WORD_AI_ARTIFACTS_ROOT")
    if env:
        return Path(env)

    return Path(__file__).resolve().parents[1] / "artifacts"


def get_model_dir() -> Path:
    return get_artifact_dir() / "models"


def get_result_dir() -> Path:
    return get_artifact_dir() / "results"


def get_log_dir() -> Path:
    return get_artifact_dir() / "logs"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_dataset(data_dir: Path | None = None) -> DatasetBundle:
    data_dir = Path(data_dir) if data_dir is not None else get_word_data_dir()

    X_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    label_map_path = data_dir / "label_map.json"
    meta_path = data_dir / "preprocess_meta.json"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing file: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing file: {label_map_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")

    X = np.load(X_path)
    y = np.load(y_path)
    label_map = load_json(label_map_path)
    meta = load_json(meta_path)

    if X.ndim != 3:
        raise ValueError(f"X must be a 3D array shaped (N, 30, F); got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if X.shape[1] != DEFAULT_SEQ_LEN:
        raise ValueError(f"Expected seq_len={DEFAULT_SEQ_LEN}, got {X.shape[1]}")

    y = y.astype(np.int64)
    feature_dim = int(X.shape[-1])
    idx_to_label = label_map.get("idx_to_label", {})
    num_classes = len(idx_to_label) if idx_to_label else int(np.max(y)) + 1

    return DatasetBundle(
        X=X.astype(np.float32),
        y=y,
        label_map=label_map,
        meta=meta,
        feature_dim=feature_dim,
        num_classes=num_classes,
        seq_len=int(X.shape[1]),
    )


def slice_feature_channels(X: np.ndarray, feature_indices: list[int] | None = None) -> np.ndarray:
    """Return the selected feature channels.

    The current dataset is already preprocessed to (N, 30, 411), so the default
    behavior is to keep the full feature tensor unchanged.

    If a future raw-feature version is supplied, pass explicit channel indices
    here or replace them from a config file.
    """
    X = np.asarray(X, dtype=np.float32)

    if feature_indices is None:
        return X

    feature_indices = [int(idx) for idx in feature_indices]
    return X[:, :, feature_indices]


def stratified_split(X: np.ndarray, y: np.ndarray, seed: int = DEFAULT_SEED):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=DEFAULT_VALIDATION_SPLIT, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y))
    return train_idx, val_idx


def build_feature_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_buffer: int | None = None,
    seed: int = DEFAULT_SEED,
    augment: bool = True,
):
    tf_mod = ensure_tensorflow()

    train_ds = tf_mod.data.Dataset.from_tensor_slices((X_train, y_train))
    if shuffle_buffer is None:
        shuffle_buffer = min(len(X_train), 8192)

    if augment:
        train_ds = train_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        train_ds = train_ds.map(lambda x, y: augment_sequence(x, y, seed=seed), num_parallel_calls=tf_mod.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf_mod.data.AUTOTUNE)

    val_ds = tf_mod.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf_mod.data.AUTOTUNE)

    return train_ds, val_ds


def augment_sequence(sequence, label, seed: int = DEFAULT_SEED):
    tf_mod = ensure_tensorflow()

    sequence = tf_mod.cast(sequence, tf_mod.float32)
    label = tf_mod.cast(label, tf_mod.int64)

    noise = tf_mod.random.normal(tf_mod.shape(sequence), mean=0.0, stddev=DEFAULT_NOISE_STD)
    sequence = sequence + noise
    sequence = time_jitter_sequence(sequence, max_shift=DEFAULT_TIME_JITTER)

    return sequence, label


def time_jitter_sequence(sequence, max_shift: int = DEFAULT_TIME_JITTER):
    tf_mod = ensure_tensorflow()

    seq_len = tf_mod.shape(sequence)[0]
    shift = tf_mod.random.uniform([], minval=-max_shift, maxval=max_shift + 1, dtype=tf_mod.int32)

    indices = tf_mod.range(seq_len, dtype=tf_mod.int32) + shift
    indices = tf_mod.clip_by_value(indices, 0, seq_len - 1)

    return tf_mod.gather(sequence, indices)


def create_training_summary(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)


def current_platform() -> str:
    return platform.platform()
