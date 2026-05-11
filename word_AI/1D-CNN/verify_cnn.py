"""1D-CNN verification script: IO checks, confidence checks, and latency benchmark."""

from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "1D-CNN" / "artifacts"))

from cnn_common import CNN_CONFIDENCE_THRESHOLD, CNN_SEED, load_cnn_dataset, load_core_indices, save_cnn_json, slice_cnn_features  # noqa: E402
from src.common import ensure_tensorflow, get_model_dir, get_result_dir, get_word_data_dir  # noqa: E402


def load_best_cnn_model(model_path: Path):
    tf = ensure_tensorflow()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return tf.keras.models.load_model(model_path)


def assert_model_input(model, X: np.ndarray) -> dict[str, Any]:
    """Verify model input shape matches actual data shape.
    
    Args:
        model: Trained model.
        X: Validation data.
    
    Returns:
        dict: Validation report.
    
    Raises:
        AssertionError: If model input shape doesn't match data shape.
    """
    model_input_shape = tuple(model.input_shape[1:])  # (seq_len, feature_dim)
    data_shape = (X.shape[1], X.shape[2])  # (seq_len, feature_dim)
    
    assert model_input_shape == data_shape, (
        f"Model input shape {model_input_shape} does not match data shape {data_shape}. "
        f"Ensure model and data are processed with the same feature dimension."
    )
    
    seq_len, feature_dim = model_input_shape
    spec_label = f"30F × {feature_dim}D"
    if feature_dim == 126:
        spec_label += " (Hands Only)"
    elif feature_dim == 216:
        spec_label += " (Baseline 216D)"
    
    return {
        "model_input_shape": list(model_input_shape),
        "data_input_shape": list(data_shape),
        "passes": True,
        "spec": spec_label,
    }


def overconfidence_check(model, sample: np.ndarray, high_threshold: float = CNN_CONFIDENCE_THRESHOLD) -> dict[str, Any]:
    probs = model.predict(sample, verbose=0)[0]
    max_prob = float(np.max(probs))
    top_prob = float(np.max(probs))
    return {
        "max_probability": max_prob,
        "top_probability": top_prob,
        "prediction_label": "Unknown" if top_prob < high_threshold else int(np.argmax(probs)),
        "over_confidence": bool(max_prob >= high_threshold),
        "threshold": high_threshold,
        "top_class": int(np.argmax(probs)),
    }


def benchmark_latency(model, input_shape: tuple[int, int], repeats: int = 500, device_name: str | None = None) -> dict[str, Any]:
    """Benchmark inference latency on CPU. Goal: achieve 30ms average latency with 126D input.
    
    Args:
        model: Trained model.
        input_shape: Expected input shape (seq_len, feature_dim).
        repeats: Number of iterations for latency measurement.
        device_name: Device to use ("/CPU:0" or "/GPU:0").
    
    Returns:
        dict: Latency statistics and pass/fail against 30ms target.
    """
    tf = ensure_tensorflow()
    dummy = tf.random.uniform((1, input_shape[0], input_shape[1]), dtype=tf.float32)

    # Warmup: 20 runs to stabilize GPU/CPU
    for _ in range(20):
        if device_name:
            with tf.device(device_name):
                _ = model.predict(dummy, verbose=0)
        else:
            _ = model.predict(dummy, verbose=0)

    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        if device_name:
            with tf.device(device_name):
                _ = model.predict(dummy, verbose=0)
        else:
            _ = model.predict(dummy, verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    latencies = np.array(latencies)
    avg_ms = float(np.mean(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    p99_ms = float(np.percentile(latencies, 99))

    # Target: 30ms average latency (reduced from 40ms with 216D)
    target_latency = 30.0
    
    return {
        "repeats": repeats,
        "average_latency_ms": avg_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "p50_latency_ms": p50_ms,
        "p99_latency_ms": p99_ms,
        "target_latency_ms": target_latency,
        "pass_target": bool(avg_ms <= target_latency),
        "note": "126D input should meet 30ms target (improved from 216D baseline)",
    }


def sample_predictions(model, X: np.ndarray, idx_to_label: dict[str, str], sample_count: int = 5) -> list[dict[str, Any]]:
    sample_count = min(sample_count, len(X))
    if sample_count <= 0:
        return []

    indices = np.linspace(0, len(X) - 1, sample_count, dtype=int)
    picked = X[indices]
    probs = model.predict(picked, verbose=0)
    preds = np.argmax(probs, axis=1)

    rows: list[dict[str, Any]] = []
    for order, (idx, pred) in enumerate(zip(indices, preds), start=1):
        rows.append(
            {
                "order": order,
                "sample_index": int(idx),
                "pred_idx": int(pred),
                "pred_label": idx_to_label.get(str(int(pred)), f"class_{int(pred)}"),
                "confidence": float(np.max(probs[order - 1])),
            }
        )
    return rows


def verify_cnn(data_dir: Path, model_path: Path, seed: int = CNN_SEED) -> dict[str, Any]:
    np.random.seed(seed)

    # Load model FIRST to get expected input shape
    model = load_best_cnn_model(model_path)
    seq_len, feature_dim = tuple(model.input_shape[1:])
    
    # Load dataset with feature_indices matching model's input shape
    if feature_dim == 126:
        # Use core_indices for 126D "Hands Only"
        feature_indices = load_core_indices()
    elif feature_dim == 216:
        # Use first 216 features for baseline
        feature_indices = np.arange(216, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported feature dimension: {feature_dim}. Expected 126 or 216.")
    
    bundle = load_cnn_dataset(data_dir, feature_indices=feature_indices)
    X = bundle.X
    idx_to_label = bundle.label_map.get("idx_to_label", {})

    io_report = assert_model_input(model, X)
    one_sample_report = overconfidence_check(model, X[:1], high_threshold=CNN_CONFIDENCE_THRESHOLD)
    input_shape = tuple(model.input_shape[1:])
    tf = ensure_tensorflow()
    gpu_devices = tf.config.list_physical_devices("GPU")
    cpu_latency = benchmark_latency(model, input_shape=input_shape, repeats=200, device_name="/CPU:0")
    gpu_latency = None
    if gpu_devices:
        gpu_latency = benchmark_latency(model, input_shape=input_shape, repeats=200, device_name="/GPU:0")

    p99_latency = benchmark_latency(model, input_shape=input_shape, repeats=1000, device_name="/CPU:0")
    samples = sample_predictions(model, X, idx_to_label=idx_to_label, sample_count=5)

    result = {
        "model_path": str(model_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "io_report": io_report,
        "one_sample_overconfidence": one_sample_report,
        "cpu_latency_benchmark": cpu_latency,
        "gpu_latency_benchmark": gpu_latency,
        "p99_latency_benchmark": p99_latency,
        "sample_predictions_for_gru_comparison": samples,
    }

    result_dir = get_result_dir()
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "verification_cnn.json"
    txt_path = result_dir / "verification_cnn.txt"

    save_cnn_json(json_path, result)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("1D-CNN Verification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model path: {result['model_path']}\n")
        f.write(f"Data path : {result['data_dir']}\n\n")
        f.write(f"IO check  : {io_report}\n")
        f.write(f"One-sample confidence check: {one_sample_report}\n\n")
        f.write(f"CPU latency benchmark (200 runs): {cpu_latency}\n")
        f.write(f"GPU latency benchmark (200 runs): {gpu_latency}\n")
        f.write(f"P99 latency benchmark (1000 runs, CPU): {p99_latency}\n\n")
        f.write("Sample predictions (for GRU comparison):\n")
        for row in samples:
            f.write(
                f"- #{row['order']} idx={row['sample_index']} -> {row['pred_label']} "
                f"(class={row['pred_idx']}, conf={row['confidence']:.4f})\n"
            )

    print("\nCNN verification finished.")
    print(f"Report JSON : {json_path}")
    print(f"Report TXT  : {txt_path}")
    print(f"CPU avg     : {cpu_latency['average_latency_ms']:.2f} ms")
    print(f"GPU avg     : {gpu_latency['average_latency_ms']:.2f} ms" if gpu_latency else "GPU avg     : unavailable")
    print(f"P99 latency : {p99_latency['average_latency_ms']:.2f} ms")
    print(f"40ms target : {bool(cpu_latency['average_latency_ms'] <= 40.0)}")

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument("--model_path", type=str, default=str(get_model_dir() / "best_cnn_model.keras"))
    parser.add_argument("--seed", type=int, default=CNN_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    verify_cnn(data_dir=Path(args.data_dir), model_path=Path(args.model_path), seed=args.seed)


if __name__ == "__main__":
    main()
