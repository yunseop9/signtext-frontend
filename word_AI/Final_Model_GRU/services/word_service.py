import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd

class GRUWordInference:
    def __init__(self, model_path, label_path, mapping_path, seq_len=30, threshold=0.6):
        if not all([os.path.exists(model_path), os.path.exists(label_path), os.path.exists(mapping_path)]):
            raise FileNotFoundError("필수 모델/라벨/매핑 파일 중 일부를 찾을 수 없습니다.")

        print("AI 엔진 및 한국어 사전 로딩 중...")
        # 추론 전용이므로 compile=False로 로드하여 속도와 안정성 확보
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.idx_to_id = data.get('idx_to_label', data)
            
        mapping_df = pd.read_csv(mapping_path)
        self.id_to_korean = dict(zip(mapping_df['label'], mapping_df['korean_word']))
        
        self.seq_len = seq_len
        self.threshold = threshold
        self.expected_input_dim = 411
        self.frame_buffer = []
        print(f"✅ 준비 완료! (임계값: {threshold}, 입력규격: {seq_len}F x 411D -> 126D)")

    def clear_buffer(self):
        """사용자 변경이나 영상 종료 시 버퍼 비우기"""
        self.frame_buffer.clear()

    def get_model_info(self):
        return {
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "seq_len": self.seq_len,
            "threshold": self.threshold
        }

    def _extract_input_126d(self, full_landmarks):
        if full_landmarks is None:
            return None

        full_landmarks = np.asarray(full_landmarks, dtype=np.float32).flatten()
        if full_landmarks.size != self.expected_input_dim:
            return None

        # `full_landmarks`는 이미 run_word_level_eval.py / process_realtime()에서
        # extract_frame_feature()로 전처리된 411D 벡터입니다.
        # 따라서 GRU 입력은 추가 정규화 없이 손 구간(75:201)만 슬라이스해야 합니다.
        hands_126d = full_landmarks[75:201]
        if hands_126d.size != 126:
            return None
        return hands_126d

    def add_frame(self, full_landmarks):
        processed_frame = self._extract_input_126d(full_landmarks)
        if processed_frame is None:
            return False
        
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        return True

    def predict(self):
        if len(self.frame_buffer) < self.seq_len:
            return None

        input_data = np.expand_dims(np.array(self.frame_buffer), axis=0)
        
        # [피드백 반영] prediction shape 명시 (첫 번째 배치 결과만 사용)
        prediction = self.model.predict(input_data, verbose=0)[0]
        
        idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        if confidence < self.threshold:
            return {
                "text": "인식불가",
                "confidence": confidence,
                "word_id": None,
                "status": "low_confidence"
            }
        
        word_id = self.idx_to_id.get(str(idx), "UNKNOWN")
        korean_text = self.id_to_korean.get(word_id, word_id)
        
        # [피드백 반영] Top-3 결과에 한국어 단어 포함하여 상세 구성
        top3_indices = np.argsort(prediction)[-3:][::-1]
        top3_results = []
        for i in top3_indices:
            t_id = self.idx_to_id.get(str(int(i)), "UNKNOWN")
            t_ko = self.id_to_korean.get(t_id, t_id)
            top3_results.append({
                "word_id": t_id,
                "text": t_ko,
                "confidence": float(prediction[i])
            })
        
        return {
            "text": korean_text,
            "confidence": confidence,
            "word_id": word_id,
            "status": "success",
            "top3": top3_results
        }

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "artifacts", "Final_GRU_HANDS_126D", "models")
    
    MODEL_PATH = os.path.join(base_path, "best_model.keras")
    LABEL_PATH = os.path.join(base_path, "label_map.json")
    MAPPING_PATH = os.path.join(base_path, "word_label_mapping.csv")
    
    try:
        # 1. 기본 설정(0.6)으로 엔진 초기화
        inference = GRUWordInference(MODEL_PATH, LABEL_PATH, MAPPING_PATH, threshold=0.6)
        
        print("🔍 [검증 시작] 가상 데이터 주입 중...")
        for _ in range(35):
            dummy = np.random.random(411)
            inference.add_frame(dummy)

        # --- 시나리오 1: 표준 임계값 적용 (포인트 2 확인) ---
        print("\n==================================================")
        print("📡 시나리오 1: 표준 임계값(0.6) 적용 결과 (안정성 테스트)")
        result_std = inference.predict()
        print(f"결과: {result_std}")
        print("=> 랜덤 데이터이므로 '인식불가'가 뜨는 것이 정상입니다.")

        # --- 시나리오 2: 임계값 해제 (포인트 1 확인) ---
        print("\n==================================================")
        print("📡 시나리오 2: 임계값(0.0) 강제 출력 결과 (매핑 테스트)")
        
        # 임계값을 잠시 0으로 변경
        original_threshold = inference.threshold
        inference.threshold = 0.0
        
        result_forced = inference.predict()
        
        # 테스트 후 다시 원복 (매너 코딩)
        inference.threshold = original_threshold
        
        print(f"결과: {result_forced}")
        print(f"=> 'text' 필드에 '고민', '슬프다' 같은 한글이 나오는지 확인하세요!")
        print("==================================================\n")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

def extract_hands_126_from_411d(sequence_411d):
    """
    word_AI 보고서 기준 입력:
    30F x 126D hands keypoint

    현재 411D 구성:
    pose 0:75
    left hand 75:138
    right hand 138:201
    face 201:411

    hands 126D = left hand 63D + right hand 63D
    """
    arr = np.asarray(sequence_411d, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 411:
        raise ValueError(f"411D sequence shape 오류: {arr.shape}")

    hands_126 = arr[:, 75:201].astype(np.float32)

    if hands_126.shape != (30, 126):
        raise ValueError(f"word_AI 입력 shape 오류: {hands_126.shape}, expected (30, 126)")

    return hands_126


def extract_sentence_120_from_411d(sequence_411d):
    """
    sentence_AI 입력용 30F x 120D sequence 생성.

    구성:
    left hand 21점 x,y = 42
    right hand 21점 x,y = 42
    pose 18점 x,y = 36
    total = 120D
    """
    arr = np.asarray(sequence_411d, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 411:
        raise ValueError(f"411D sequence shape 오류: {arr.shape}")

    pose_75 = arr[:, 0:75].reshape(arr.shape[0], 25, 3)
    left_63 = arr[:, 75:138].reshape(arr.shape[0], 21, 3)
    right_63 = arr[:, 138:201].reshape(arr.shape[0], 21, 3)

    left_xy = left_63[:, :, :2].reshape(arr.shape[0], 42)
    right_xy = right_63[:, :, :2].reshape(arr.shape[0], 42)
    pose18_xy = pose_75[:, :18, :2].reshape(arr.shape[0], 36)

    sentence_120 = np.concatenate([left_xy, right_xy, pose18_xy], axis=1).astype(np.float32)

    if sentence_120.shape != (30, 120):
        raise ValueError(f"sentence_AI 입력 shape 오류: {sentence_120.shape}, expected (30, 120)")

    return sentence_120


def extract_degree_280_from_411d(sequence_411d):
    """
    degree_AI 보고서 기준 입력:
    1F x 280D

    구성:
    16D 얼굴 요약 feature
    + 132D 정규화 얼굴 landmark
    + 132D delta feature
    = 280D

    현재 411D의 face 영역은 201:411, 70점 x (x,y,c) = 210D.
    이 중 앞 66개 얼굴점의 x,y를 사용해 132D를 만든다.
    """
    arr = np.asarray(sequence_411d, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 411:
        raise ValueError(f"411D sequence shape 오류: {arr.shape}")

    face = arr[:, 201:411].reshape(arr.shape[0], 70, 3)
    face_xy = face[:, :66, :2]  # 30F x 66 x 2

    # confidence가 전부 0이면 얼굴 검출 실패로 판단
    face_conf = face[:, :, 2]
    has_face = bool(np.any(face_conf > 0))

    # 대표 프레임: 중앙 프레임
    mid_idx = len(face_xy) // 2
    current_xy = face_xy[mid_idx]

    # 기준 프레임: 첫 프레임
    base_xy = face_xy[0]

    # 정규화: 얼굴 중심과 scale 기준
    center = np.mean(current_xy, axis=0)
    centered = current_xy - center

    scale = np.std(centered)
    if scale < 1e-6:
        scale = 1.0

    norm_xy = centered / scale
    norm_132 = norm_xy.reshape(-1)

    base_center = np.mean(base_xy, axis=0)
    base_centered = base_xy - base_center
    base_scale = np.std(base_centered)
    if base_scale < 1e-6:
        base_scale = 1.0

    base_norm = base_centered / base_scale
    delta_132 = (norm_xy - base_norm).reshape(-1)

    # 16D summary feature
    x = norm_xy[:, 0]
    y = norm_xy[:, 1]
    dx = delta_132[0::2]
    dy = delta_132[1::2]

    summary_16 = np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.mean(y),
        np.std(y),
        np.min(y),
        np.max(y),
        np.mean(dx),
        np.std(dx),
        np.min(dx),
        np.max(dx),
        np.mean(dy),
        np.std(dy),
        np.min(dy),
        np.max(dy),
    ], dtype=np.float32)

    degree_280 = np.concatenate([
        summary_16,
        norm_132.astype(np.float32),
        delta_132.astype(np.float32),
    ]).astype(np.float32)

    if degree_280.shape[0] != 280:
        raise ValueError(f"degree_AI 입력 shape 오류: {degree_280.shape}, expected (280,)")

    return degree_280, has_face