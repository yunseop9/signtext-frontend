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
        self.frame_buffer = []
        print(f"✅ 준비 완료! (임계값: {threshold}, 입력규격: {seq_len}F x 126D)")

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

    def _extract_hands_126d(self, full_landmarks):
        if full_landmarks is None or len(full_landmarks) < 201:
            return None
        return full_landmarks[75:201]

    def _normalize_shoulder(self, landmarks_126d, pose_landmarks):
        """어깨 기준 정규화 (안정성 및 리턴 타입 일관성 강화)"""
        l_sh = pose_landmarks[11]
        r_sh = pose_landmarks[12]
        
        shoulder_center = (l_sh[:2] + r_sh[:2]) / 2
        shoulder_dist = np.linalg.norm(l_sh[:2] - r_sh[:2]) + 1e-6
        
        # [피드백 반영] return 타입 일관성 유지 (flattened numpy array)
        if shoulder_dist < 1e-3:
            return np.array(landmarks_126d).flatten()
            
        reshaped = landmarks_126d.reshape(-1, 3)
        reshaped[:, :2] = (reshaped[:, :2] - shoulder_center) / shoulder_dist
        return reshaped.flatten()

    def add_frame(self, full_landmarks):
        hands_data = self._extract_hands_126d(full_landmarks)
        if hands_data is None:
            return False
            
        pose_data = full_landmarks[0:75].reshape(-1, 3)
        processed_frame = self._normalize_shoulder(hands_data, pose_data)
        
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