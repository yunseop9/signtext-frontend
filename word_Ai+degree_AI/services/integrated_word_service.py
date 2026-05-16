import sys
import os
import numpy as np
from pathlib import Path

# --- 경로 설정 (word_AI 폴더의 서비스를 가져오기 위함) ---
CURRENT_DIR = Path(__file__).resolve().parent
# 99. 깃헙 코드 -sign_language 폴더를 루트로 설정
PROJECT_ROOT = CURRENT_DIR.parents[1] 

# word_AI 경로를 시스템 경로에 추가하여 임포트 가능하게 함
WORD_AI_PATH = PROJECT_ROOT / "word_AI"
if str(WORD_AI_PATH) not in sys.path:
    sys.path.insert(0, str(WORD_AI_PATH))

# 개별 서비스 임포트
from services.word_service import GRUWordInference
from semantic_word_postprocessor import SemanticWordPostprocessor

class IntegratedWordService:
    def __init__(self, model_path, label_path, mapping_path):
        print("🚀 [통합 서비스] 엔진 초기화 중...")
        # 1. 단어 인식 AI 엔진 로드
        self.word_engine = GRUWordInference(model_path, label_path, mapping_path)
        # 2. LLM 후처리 엔진 로드
        self.llm_engine = SemanticWordPostprocessor()
        print("✨ 모든 엔진이 성공적으로 로드되었습니다.")

    def process_realtime(self, full_landmarks, degree_ko="보통"):
        """
        좌표 데이터를 입력받아 [전처리 -> 단어인식 -> LLM 보정] 과정을 수행
        """
        # 1. 단어 AI 버퍼에 프레임 추가
        self.word_engine.add_frame(full_landmarks)
        
        # 2. 단어 추론 수행
        word_result = self.word_engine.predict()
        
        # 3. 인식에 성공한 경우에만 LLM 후처리 진행
        if word_result and word_result['status'] == 'success':
            # word_result['text'] 에는 이미 '슬프다' 같은 한국어 단어가 들어있음
            final_output = self.llm_engine.process(word_result['text'], degree_ko)
            
            return {
                "status": "success",
                "original_word": word_result['text'],
                "confidence": word_result['confidence'],
                "final_sentence": final_output['final_text'],
                "modifier": final_output['modifier'],
                "reason": final_output['reason']
            }
        
        return {"status": "processing", "message": "데이터 축적 중이거나 확신도가 낮음"}

if __name__ == "__main__":
    # --- 통합 테스트 코드 ---
    # 준혁님의 실제 경로 구조에 맞춰 artifacts 경로 설정
    base_model_path = PROJECT_ROOT / "word_AI" / "artifacts" / "Final_GRU_HANDS_126D" / "models"
    
    MODEL = str(base_model_path / "best_model.keras")
    LABEL = str(base_model_path / "label_map.json")
    MAP = str(base_model_path / "word_label_mapping.csv")

    # 서비스 생성
    service = IntegratedWordService(MODEL, LABEL, MAP)
    
    # 가상 데이터 테스트 (포인트: 단어 AI + LLM이 한 번에 도는지 확인)
    print("\n🔥 [최종 검증] 통합 추론 테스트 시작...")
    service.word_engine.threshold = 0.0 # 테스트를 위해 모든 결과를 통과시킴
    
    # 35프레임 주입 (버퍼 채우기)
    for _ in range(35):
        dummy_data = np.random.random(411)
        service.process_realtime(dummy_data)
        
    # 마지막 프레임과 함께 감정 강도 "강함" 전달
    final_res = service.process_realtime(np.random.random(411), degree_ko="강함")
    
    import json
    print(f"\n✅ 통합 서비스 최종 응답:\n{json.dumps(final_res, indent=2, ensure_ascii=False)}")