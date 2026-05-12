"""Performance report generator for Hybrid Model (84D slicing)."""
from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "Hybrid" / "artifacts"))

# 134차원 필터 임포트
from hybrid_common import extract_pose_hands_and_remove_c
from src.common import (
    ensure_tensorflow, get_result_dir, get_word_data_dir, 
    load_dataset, load_json, stratified_split
)

def run_report():
    tf = ensure_tensorflow()
    model_path = Path(os.environ["WORD_AI_ARTIFACTS_ROOT"]) / "models" / "best_Hybrid_model.keras"
    
    # 데이터 로드 및 134D 슬라이싱
    bundle = load_dataset(get_word_data_dir())
    X, y = bundle.X, bundle.y
    
    # 134차원 필터 적용
    X = extract_pose_hands_and_remove_c(X)
    
    # 검증셋 분리
    _, val_idx = stratified_split(X, y, seed=42)
    X_val, y_val = X[val_idx], y[val_idx]
    
    # 모델 로드 (추론 전용이므로 compile=False)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # 예측 수행
    print("Evaluating Hybrid model performance...")
    y_pred_prob = model.predict(X_val, batch_size=256)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # 지표 계산
    acc = accuracy_score(y_val, y_pred)
    cls_report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    
    # Latency 정보 가져오기 (verify_hybrid 결과 참조)
    verify_path = get_result_dir() / "verification_hybrid.json"
    latency_ms = load_json(verify_path).get("average_latency_ms", 0.0) if verify_path.exists() else 0.0
    
    # 성적표 구성 (객관적 수치 원칙)
    metrics = {
        #411차원 테스트 포인트!!! "입력 구조": f"30F × 84D",
        "입력 구조": f"30F x 134D", #411차원 테스트 포인트!!!
        "모델명": "Hybrid (1D-CNN+GRU)",
        "Accuracy": round(acc, 4),
        "Macro F1": round(cls_report['macro avg']['f1-score'], 4),
        "Weighted F1": round(cls_report['weighted avg']['f1-score'], 4),
        "Avg Latency(ms)": round(latency_ms, 2)
    }
    
    df = pd.DataFrame([metrics])
    
    # 파일 저장
    output_path = get_result_dir() / "performance_hybrid.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[Hybrid Model 자체 성능 리포트]\n")
        f.write(df.to_string(index=False))
        f.write(f"\n\nValidation Samples: {len(y_val)}")
    
    print("\n" + df.to_string(index=False))
    print(f"\nFull report saved at: {output_path}")

if __name__ == "__main__":
    run_report()