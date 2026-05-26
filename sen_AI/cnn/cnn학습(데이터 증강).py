import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUG_DATA_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료_증강\train"
ORIG_DATA_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\validation"
CLASSES_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\processed\classes.npy"

MODEL_SAVE_DIR = Path(r"C:\Users\wolah\Desktop\학습모델")
RESULTS_DIR = MODEL_SAVE_DIR / "results" / "cnn_augmented"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1

# ==========================================
# 2. 데이터 로드
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

# ==========================================
# 3. 모델 정의 (Pure CNN 구조)
# ==========================================
def build_cnn_softmax(seq_len, feat_dim, num_classes, dropout_rate):
    model = Sequential([
        tf.keras.Input(shape=(seq_len, feat_dim)),
        
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn_softmax(SEQ_LEN, FEATURE_DIM, NUM_CLASSES, DROPOUT_RATE)

# ==========================================
# 4. 커스텀 손실 함수 및 옵티마이저 설정
# ==========================================
def sparse_categorical_crossentropy_with_label_smoothing(y_true, y_pred):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.cast(num_classes, tf.int32))
    y_true = y_true * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.compile(
    optimizer=optimizer,
    loss=sparse_categorical_crossentropy_with_label_smoothing,
    metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
)
model.summary()

# ==========================================
# 5. 학습 시작
# ==========================================
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = MODEL_SAVE_DIR / f"cnn_augmented_{timestamp}.keras"

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(filepath=str(model_filename), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

print(f"\n🚀 [TRAIN] CNN 모델 학습 시작")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1
)