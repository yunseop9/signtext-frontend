import json
import requests
import re

class SemanticWordPostprocessor:
    def __init__(self, model_name="qwen2.5:7b"):
        self.url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        # [피드백 3 반영] 규칙 기반 Fallback 사전
        self.fallback_mapping = {
            "강함": "매우",
            "약함": "조금",
            "보통": ""
        }

    def _clean_json_string(self, raw_str):
        """[피드백 1 반영] LLM 응답에서 마크다운 태그 등을 제거하고 순수 JSON만 추출"""
        # ```json { ... } ``` 형태 제거
        clean_str = re.sub(r'```json|```', '', raw_str).strip()
        return clean_str

    def process(self, word_text, degree_ko):
        """
        단어와 감정 강도를 받아 LLM 후처리를 수행합니다.
        강력한 예외 처리와 검증 로직이 포함되어 있습니다.
        """
        prompt = f"""
        당신은 수어 통역 전문가입니다. 
        입력된 단어와 감정의 강도를 조합하여 가장 자연스러운 한국어 표현을 만드세요.

        입력 데이터:
        - 인식된 단어: {word_text}
        - 감정 강도: {degree_ko}

        출력은 반드시 아래 JSON 형식만 반환하세요:
        {{
          "apply_degree": true,
          "final_text": "수정된 최종 텍스트",
          "target_expression": "{word_text}",
          "modifier": "추가된 수식어",
          "reason": "반영 이유"
        }}
        """

        try:
            response = requests.post(self.url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }, timeout=15)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            # 1. 원본 응답 추출 및 청소
            raw_response = response.json().get('response', '')
            clean_response = self._clean_json_string(raw_response)
            
            # 2. JSON 파싱
            result = json.loads(clean_response)
            
            # [피드백 2 반영] 필수 Key 검증
            required_keys = ["apply_degree", "final_text", "target_expression", "modifier", "reason"]
            if not all(key in result for key in required_keys):
                raise ValueError("Missing required keys in LLM response")

            # [피드백 4 반영] 성공 상태 추가
            result["status"] = "success"
            return result

        except Exception as e:
            # [피드백 3 반영] 강화된 Fallback 로직
            modifier = self.fallback_mapping.get(degree_ko, "")
            final_text = f"{modifier} {word_text}".strip() if modifier else word_text
            
            return {
                "status": "fallback", # [피드백 4 반영]
                "apply_degree": True if modifier else False,
                "final_text": final_text,
                "target_expression": word_text,
                "modifier": modifier,
                "reason": f"LLM 호출 실패 또는 응답 오류로 인한 규칙 기반 처리 ({str(e)})"
            }

if __name__ == "__main__":
    processor = SemanticWordPostprocessor()
    print("🚀 [고도화 버전] Ollama 후처리 테스트 시작...")
    
    # 테스트 시나리오
    test_word = "슬프다"
    test_degree = "강함"
    
    result = processor.process(test_word, test_degree)
    print(f"\n✅ 최종 결과:\n{json.dumps(result, indent=2, ensure_ascii=False)}")