"""Shared helpers for the 1D-CNN pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.common import ensure_tensorflow, load_dataset, save_json, stratified_split, create_training_summary


CNN_SEQ_LEN = 30
CNN_BATCH_SIZE = 256
CNN_SEED = 42
CNN_EPOCHS = 80
CNN_LR = 1e-3
CNN_MIN_LR = 1e-5
CNN_REDUCE_PATIENCE = 5
CNN_EARLY_STOP_PATIENCE = 15
CNN_GAUSSIAN_STD = 0.005  # Refined from 0.01 for more stable augmentation
CNN_CONFIDENCE_THRESHOLD = 0.7

# Temporary: Use 216D baseline (skip core_indices) for diagnostics
USE_CORE_INDICES = False  # Set to True to use 126D "Hands Only"
CNN_FEATURE_DIM = 216 if not USE_CORE_INDICES else 126


@dataclass(frozen=True)
class CnnDatasetBundle:
    X: np.ndarray
    y: np.ndarray
    seq_len: int
    feature_dim: int
    num_classes: int
    label_map: dict[str, Any]
    meta: dict[str, Any]


def load_core_indices(cnn_dir: Path | None = None) -> np.ndarray:
    """Load hand-specific feature indices from core_indices.txt (126D 'Hands Only' subset).
    
    Args:
        cnn_dir: Path to 1D-CNN directory. Defaults to current script's parent.
    
    Returns:
        np.ndarray: Integer indices of core features (shape: (126,)).
    """
    if cnn_dir is None:
        cnn_dir = Path(__file__).resolve().parent
    
    core_file = cnn_dir / "core_indices.txt"
    if not core_file.exists():
        raise FileNotFoundError(f"core_indices.txt not found at {core_file}. Please run find_hand_indices.py first.")
    
    indices = np.loadtxt(core_file, dtype=np.int32)
    print(f"✓ Loaded {len(indices)} core feature indices from {core_file.name}")
    return indices


def resolve_feature_indices(total_dim: int, target_dim: int | None = None, cnn_dir: Path | None = None) -> np.ndarray:
    """Resolve feature indices: first tries core_indices.txt (126D 'Hands Only'),
    falls back to first target_dim features if file not found or disabled.
    
    Args:
        total_dim: Total feature dimension (should be 411).
        target_dim: Target dimension. Defaults to CNN_FEATURE_DIM (126 or 216).
        cnn_dir: Path to 1D-CNN directory for core_indices.txt.
    
    Returns:
        np.ndarray: Integer indices to use for feature slicing.
    """
    if target_dim is None:
        target_dim = CNN_FEATURE_DIM
    
    if total_dim < target_dim:
        raise ValueError(f"Need at least {target_dim} features, got {total_dim}")
    
    # Skip core_indices if disabled for 216D baseline diagnostics
    if not USE_CORE_INDICES:
        print(f"⚠ 216D baseline mode: using first {target_dim} features (core_indices skipped)")
        return np.arange(target_dim, dtype=np.int32)
    
    try:
        return load_core_indices(cnn_dir)
    except FileNotFoundError:
        print(f"⚠ core_indices.txt not found; using first {target_dim} features as fallback")
        return np.arange(target_dim, dtype=np.int32)


def slice_cnn_features(X: np.ndarray, feature_indices: np.ndarray | None = None, cnn_dir: Path | None = None) -> np.ndarray:
    """Slice features by indices and apply Min-Max normalization to [-1.0, 1.0].
    
    Args:
        X: Input array (samples, seq_len, features).
        feature_indices: Feature indices to extract. If None, loads core_indices.txt.
        cnn_dir: Path to 1D-CNN directory for core_indices.txt.
    
    Returns:
        np.ndarray: Sliced and normalized features (samples, seq_len, len(feature_indices)).
    """
    X = np.asarray(X, dtype=np.float32)
    
    if feature_indices is None:
        feature_indices = resolve_feature_indices(X.shape[-1], target_dim=CNN_FEATURE_DIM, cnn_dir=cnn_dir)
    
    # Slice to selected features
    X_sliced = X[:, :, feature_indices]
    
    # Min-Max Scaling: normalize to [-1.0, 1.0]
    # Reshape for sklearn: (samples*seq_len, features)
    n_samples, n_frames, n_features = X_sliced.shape
    X_reshaped = X_sliced.reshape(-1, n_features)
    
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Reshape back
    X_final = X_scaled.reshape(n_samples, n_frames, n_features)
    
    print(f"✓ Sliced to {n_features} features and applied Min-Max normalization to [-1.0, 1.0]")
    print(f"  Final shape: {X_final.shape}")
    
    return X_final


def load_cnn_dataset(data_dir: Path, feature_indices: np.ndarray | None = None, cnn_dir: Path | None = None) -> CnnDatasetBundle:
    """Load dataset and apply 1D-CNN-specific preprocessing (126D slicing + Min-Max normalization).
    
    Args:
        data_dir: Path to dataset directory (containing X.npy, y.npy, etc.).
        feature_indices: Feature indices to extract. If None, loads core_indices.txt.
        cnn_dir: Path to 1D-CNN directory for core_indices.txt.
    
    Returns:
        CnnDatasetBundle: Processed data with 126D 'Hands Only' features.
    """
    bundle = load_dataset(data_dir)
    X = slice_cnn_features(bundle.X, feature_indices=feature_indices, cnn_dir=cnn_dir)
    if X.shape[1] != CNN_SEQ_LEN:
        raise ValueError(f"Expected seq_len={CNN_SEQ_LEN}, got {X.shape[1]}")

    return CnnDatasetBundle(
        X=X,
        y=bundle.y,
        seq_len=int(X.shape[1]),
        feature_dim=int(X.shape[-1]),
        num_classes=int(bundle.num_classes),
        label_map=bundle.label_map,
        meta=bundle.meta,
    )


def one_hot_labels(y: np.ndarray, num_classes: int):
    tf = ensure_tensorflow()
    return tf.one_hot(tf.cast(y, tf.int32), depth=num_classes, dtype=tf.float32)


def build_cnn_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    batch_size: int = CNN_BATCH_SIZE,
    seed: int = CNN_SEED,
    gaussian_std: float = CNN_GAUSSIAN_STD,
):
    tf = ensure_tensorflow()

    def augment_train(sequence, label):
        sequence = tf.cast(sequence, tf.float32)
        noise = tf.random.normal(tf.shape(sequence), mean=0.0, stddev=gaussian_std)
        sequence = sequence + noise
        # Keep label as integer for sparse_categorical_crossentropy
        return sequence, label

    def prepare_eval(sequence, label):
        sequence = tf.cast(sequence, tf.float32)
        # Keep label as integer for sparse_categorical_crossentropy
        return sequence, label

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(min(len(X_train), 8192), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def safe_stratified_split(X: np.ndarray, y: np.ndarray, seed: int = CNN_SEED):
    try:
        return stratified_split(X, y, seed=seed)
    except ValueError:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        val_size = max(1, int(round(len(indices) * 0.2)))
        val_idx = np.sort(indices[:val_size])
        train_idx = np.sort(indices[val_size:])
        return train_idx, val_idx


def save_cnn_json(path: Path, data: dict[str, Any]) -> None:
    save_json(path, data)


def save_cnn_summary(path: Path, **payload: Any) -> None:
    create_training_summary(path, **payload)
