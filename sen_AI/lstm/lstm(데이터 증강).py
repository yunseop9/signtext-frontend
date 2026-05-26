import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks

# ==========================================
# 1. 설정 및 경로
# ==========================================
AUG_DATA_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료_증강\train"
ORIG_DATA_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\validation"
CLASSES_PATH = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\processed\classes.npy"
MODEL_SAVE_DIR = Path(r"C:\Users\wolah\Desktop\학습모델")
RESULTS_DIR = MODEL_SAVE_DIR / "results" / "lstm_augmented"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1

# ==========================================
# 2. 데이터 로드
# ==========================================
X_train = np.load(f"{AUG_DATA_PATH}\\X_train_aug.npy").astype(np.float32)
y_train = np.load(f"{AUG_DATA_PATH}\\y_train_aug.npy").astype(np.int64)
X_val = np.load(f"{ORIG_DATA_PATH}\\X_validation.npy").astype(np.float32)
y_val = np.load(f"{ORIG_DATA_PATH}\\y_validation.npy").astype(np.int64)
classes = np.load(CLASSES_PATH, allow_pickle=True)

# ==========================================
# 3. 모델 정의 (Pure LSTM)
# ==========================================
def build_lstm_softmax(seq_len, feat_dim, num_classes, dropout_rate):
    model = Sequential([
        tf.keras.Input(shape=(seq_len, feat_dim)),
        
        # LSTM은 복잡한 게이트 구조로 긴 시퀀스 기억에 유리
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(dropout_rate),
        
        layers.LSTM(256),
        layers.Dropout(dropout_rate),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_lstm_softmax(X_train.shape[1], X_train.shape[2], len(classes), DROPOUT_RATE)

# ==========================================
# 4. 손실 함수 (Label Smoothing)
# ==========================================
def loss_fn(y_true, y_pred):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.cast(num_classes, tf.int32))
    y_true = y_true * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

model.compile(optimizer='adam', loss=loss_fn, metrics=["accuracy"])

# ==========================================
# 5. 학습
# ==========================================
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = MODEL_SAVE_DIR / f"lstm_augmented_{timestamp}.keras"

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath=str(model_filename), monitor="val_loss", save_best_only=True)
]

print("\n🚀 [TRAIN] LSTM 모델 학습 시작")
model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_list, verbose=1)