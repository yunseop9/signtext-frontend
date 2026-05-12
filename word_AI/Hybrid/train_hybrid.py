"""Hybrid (1D-CNN + GRU) training entry point for word-level sign language modeling."""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트 경로 설정 및 모듈 경로 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 하이브리드 모델 전용 결과물 저장 경로 강제 고정 (경로 무결성 원칙)
os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "Hybrid" / "artifacts"))

# 기존 공용 모듈에서 필수 유틸리티 임포트 (기존 코드 스타일 계승)
from src.common import (
    ensure_tensorflow, 
    get_model_dir, 
    get_result_dir, 
    get_word_data_dir,
    load_dataset, 
    slice_feature_channels, 
    stratified_split, 
    create_training_summary,
    save_json,
    DEFAULT_EPOCHS, 
    DEFAULT_SEED, 
    DEFAULT_BATCH_SIZE
)

def set_global_seed(seed: int) -> None:
    """재현성을 위한 글로벌 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    tf = ensure_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def build_hybrid_model(input_shape: tuple[int, int], num_classes: int):
    """
    1D-CNN(공간적 특징) + GRU(시간적 흐름) 하이브리드 모델 설계
    """
    tf = ensure_tensorflow()
    inputs = tf.keras.Input(shape=input_shape, name="hybrid_input")
    
    # ----------------------------------------------------
    # [1] 1D-CNN Block : 프레임 내/인접 프레임 간 국소 특징 추출
    # ----------------------------------------------------
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="cnn_conv1d")(inputs)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn")(x)
    x = tf.keras.layers.Activation('relu', name="cnn_relu")(x)
    
    # ----------------------------------------------------
    # [2] GRU Block : 전체 동작의 시계열적 맥락(문맥) 파악
    # ----------------------------------------------------
    x = tf.keras.layers.GRU(128, return_sequences=False, name="gru_layer")(x)
    
    # ----------------------------------------------------
    # [3] Classifier Block : 특징 통합 및 최종 분류
    # HeNormal 초기화 적용 (Loss 8.0 정체 현상 방지)
    # ----------------------------------------------------
    x = tf.keras.layers.Dense(
        64, 
        activation='relu', 
        kernel_initializer='he_normal', 
        name="dense_features"
    )(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="classifier")(x)
    
    model = tf.keras.Model(inputs, outputs, name="Hybrid_1DCNN_GRU")
    
    # 최적화 및 컴파일
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_hybrid(data_dir: Path, seed: int = DEFAULT_SEED, batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS):
    tf = ensure_tensorflow()
    set_global_seed(seed)
    
    from hybrid_common import extract_pose_hands_and_remove_c

    print("Loading dataset for Hybrid Model...")
    bundle = load_dataset(data_dir)
    X, y = bundle.X, bundle.y
    
    # [핵심 최적화] 독립적인 모듈을 통해 포즈와 양손 슬라이싱 및 노이즈(c) 제거 동시 수행
    
    # print(f"Data shape for 411D test: {X.shape}") #411차원 테스트 포인트!!!
    # [최적화 적용] 포즈+양손 슬라이싱 및 c 채널 제거 (134D)
    X = extract_pose_hands_and_remove_c(X)
    print(f"Data refined to Pose+Hands (x, y). Final shape: {X.shape}")

    # Train / Validation 분할
    train_idx, val_idx = stratified_split(X, y, seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # tf.data.Dataset 구축
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(min(len(X_train), 8192), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 모델 구축
    input_shape = (bundle.seq_len, X_train.shape[-1]) # (30, 126) 예상
    model = build_hybrid_model(input_shape, bundle.num_classes)
    model.summary()
    
    # 콜백 설정
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_Hybrid_model.keras" # 하이브리드 고정 경로
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    # 모델 학습
    print("Starting Hybrid Model Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # 학습 결과 저장
    result_dir = get_result_dir()
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "history.json", history.history)
    
    # [추가] 학습 과정 시각화 및 이미지 저장 (최소 수정 원칙)
    plt.figure(figsize=(12, 4))
    
    # 1. Accuracy 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Hybrid Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Loss 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Hybrid Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 3. 이미지 저장
    plot_path = result_dir / "training_history_hybrid.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"History plot saved at: {plot_path}")


    print("\nHybrid Training Process Completed Successfully!")
    print(f"Best Hybrid Model saved at: {best_model_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_hybrid(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs
    )