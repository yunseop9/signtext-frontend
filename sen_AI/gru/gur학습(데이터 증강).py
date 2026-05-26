import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks
import matplotlib.pyplot as plt

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 증강된 훈련 데이터 경로와 원본 검증 데이터 경로를 분리합니다.
AUG_DATA_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료_증강\train"
ORIG_DATA_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\validation"
CLASSES_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\processed\classes.npy"

MODEL_SAVE_DIR = Path(r"C:\Users\wolah\Desktop\학습모델")
RESULTS_DIR = MODEL_SAVE_DIR / "results" / "gru_augmented"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128 # 데이터가 늘어났으므로 배치 사이즈 증가
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1

# ==========================================
# 2. 데이터 로드 (훈련은 증강 데이터, 검증은 원본 데이터)
# ==========================================
print("증강된 훈련 데이터와 원본 검증 데이터를 불러오는 중입니다...")
X_train = np.load(f"{AUG_DATA_PATH}\\X_train_aug.npy").astype(np.float32)
y_train = np.load(f"{AUG_DATA_PATH}\\y_train_aug.npy").astype(np.int64)

X_val = np.load(f"{ORIG_DATA_PATH}\\X_validation.npy").astype(np.float32)
y_val = np.load(f"{ORIG_DATA_PATH}\\y_validation.npy").astype(np.int64)

classes = np.load(CLASSES_PATH, allow_pickle=True)
NUM_CLASSES = len(classes)
SEQ_LEN = X_train.shape[1]
FEATURE_DIM = X_train.shape[2]

print(f"X_train(증강): {X_train.shape}, y_train(증강): {y_train.shape}")
print(f"X_val(원본):   {X_val.shape}, y_val(원본):   {y_val.shape}")

# ==========================================
# 3. 모델 정의 (Softmax 기반 순수 GRU)
# ==========================================
def build_gru_softmax(seq_len, feat_dim, num_classes, dropout_rate):
    model = Sequential([
        tf.keras.Input(shape=(seq_len, feat_dim)),
        
        layers.GRU(256, return_sequences=True, activation='tanh'),
        layers.Dropout(dropout_rate),
        
        layers.GRU(256, activation='tanh'),
        layers.Dropout(dropout_rate),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_gru_softmax(SEQ_LEN, FEATURE_DIM, NUM_CLASSES, DROPOUT_RATE)

# ==========================================
# 4. Label Smoothing & Optimizer 설정
# ==========================================
def sparse_cce_with_label_smoothing(label_smoothing, num_classes):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        smooth = tf.cast(label_smoothing, tf.float32)
        num_classes_f = tf.cast(num_classes, tf.float32)
        y_true_smoothed = y_true_one_hot * (1.0 - smooth) + (smooth / num_classes_f)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        return cce(y_true_smoothed, y_pred)
    return loss_fn

try:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    print("내장 SparseCategoricalCrossentropy(label_smoothing) 사용")
except Exception:
    loss_fn = sparse_cce_with_label_smoothing(LABEL_SMOOTHING, NUM_CLASSES)
    print("커스텀 Label Smoothing Loss 함수 사용")

try:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("AdamW 옵티마이저 사용")
except Exception:
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    print("Adam 옵티마이저 사용")

# ==========================================
# 5. 모델 컴파일
# ==========================================
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]
)

# ==========================================
# 6. 콜백 설정 및 훈련 시작
# ==========================================
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = MODEL_SAVE_DIR / f"gru_augmented_{timestamp}.keras"
history_csv = RESULTS_DIR / f"history_{timestamp}.csv"

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(filepath=str(model_filename), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

print(f"\n[TRAIN] 증강된 데이터로 GRU 모델 학습 시작 (Batch: {BATCH_SIZE}, Epochs: {EPOCHS})")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1
)

# ==========================================
# 7. 결과 저장 및 시각화
# ==========================================
pd.DataFrame(history.history).to_csv(history_csv, index=False, encoding="utf-8-sig")
print(f"\n성적표 저장 완료: {history_csv}")
print(f"최적 모델 저장 완료: {model_filename}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history.get("accuracy", []), label="Train Accuracy", color="blue")
plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy", color="orange")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history.get("loss", []), label="Train Loss", color="blue")
plt.plot(history.history.get("val_loss", []), label="Val Loss", color="orange")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()