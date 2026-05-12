"""Verification script for Hybrid (1D-CNN + GRU) model latency and input spec."""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

# 경로 설정
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "Hybrid" / "artifacts"))

from src.common import ensure_tensorflow, get_model_dir, get_result_dir, save_json

def benchmark_latency(model, input_shape: tuple, num_runs: int = 300):
    """300회 반복 추론을 통해 평균 지연 시간(ms) 측정"""
    tf = ensure_tensorflow()
    # 더미 데이터 생성 (Batch size = 1)
    dummy_input = np.random.random((1, *input_shape)).astype(np.float32)
    
    # 워밍업 (Warm-up)
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
    
    print(f"Benchmarking latency over {num_runs} runs...")
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = model.predict(dummy_input, verbose=0)
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) / num_runs) * 1000
    return avg_latency

def verify_hybrid():
    tf = ensure_tensorflow()
    model_path = get_model_dir() / "best_Hybrid_model.keras"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded: {model_path}")
    
    # [1] 입력 규격 검증 (84차원 확인)
    
    expected_shape = (30, 134) #411차원 테스트 포인트!!!
    actual_shape = tuple(model.input_shape[1:])
    print(f"Expected Input Shape: {expected_shape}")
    print(f"Actual Input Shape  : {actual_shape}")
    assert actual_shape == expected_shape, f"Input shape mismatch! Expected {expected_shape}, got {actual_shape}"
    
    # [2] 환경 체크 (GPU 사용 여부)
    gpu_devices = tf.config.list_physical_devices('GPU')
    device_name = "GPU" if gpu_devices else "CPU"
    print(f"Running on: {device_name}")
    
    # [3] 성능 벤치마크
    avg_ms = benchmark_latency(model, actual_shape)
    print(f"\nAverage Latency: {avg_ms:.2f} ms")
    
    # 결과 저장
    results = {
        "model_name": "Hybrid_1DCNN_GRU",
        "input_shape": actual_shape,
        "device": device_name,
        "average_latency_ms": avg_ms,
        "meets_40ms_target": avg_ms <= 40.0
    }
    
    result_dir = get_result_dir()
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "verification_hybrid.json", results)
    print(f"Verification report saved at: {result_dir / 'verification_hybrid.json'}")

if __name__ == "__main__":
    verify_hybrid()