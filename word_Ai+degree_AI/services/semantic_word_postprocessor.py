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
        당신은 청각장애인의 수어를 자연스러운 일상 대화로 번역해주는 '친근한 대화형 수어 통역사'입니다.
        입력된 단어와 감정 강도를 조합하여 문장을 만들되, 반드시 아래의 [3단계 프로세스]를 거쳐 검토한 후 최종 결과를 출력하세요.

        [입력 데이터]
        - 인식된 단어: {word_text}
        - 표현의 강도: {degree_ko}

        [⚠️ 필수 준수 규칙]
        1. 문어체(책, 뉴스, 사전에만 나오는 딱딱한 표현)는 절대 금지합니다.
        - ❌ 나쁜 예: 극도로 슬프다, 심각하게 슬프다, 지대하게 기쁘다 (일상 대화에서 쓰지 않음)
        2. 일상 구어체(사람들이 평소에 진짜 말로 주고받는 친근하고 자연스러운 표현)를 사용하세요.
        -  좋은 예: 정말 너무 슬프다, 진짜 많이 슬프다, 너무나도 슬프다
        3. 강도가 '강함'일 때는 단순 기계적 결합을 넘어, 일상적으로 감정을 강하게 표현하는 단어를 고르세요.
        4. JSON 내의 모든 필드값("final_text", "modifier", "reason")은 100% 한국어로만 작성하세요. 중국어나 영어 등 외국어 번역 표기는 절대 절대 금지합니다. 특히 "modifier"에는 문장에 사용한 한국어 부사만 넣으세요.

        [⚙️ 3단계 처리 프로세스]
        - 1단계 [초안 작성]: 단어와 강도를 조합하여 1차 문장 후보를 마음속으로 작성합니다.
        - 2단계 [비판적 검토]: 자기가 만든 초안을 다시 읽으며 "이게 일상 대화에서 실제 사람이 쓰는 자연스러운 말인가? 너무 딱딱하거나 낯설지 않은가?"라고 스스로 비판해보세요. 만약 '극도로', '심각하게' 같은 딱딱한 단어가 들어갔다면 감점 요인입니다.
        - 3단계 [수정 및 출력]: 검토를 거쳐 가장 사람 냄새 나는 자연스러운 문장으로 수정한 뒤, 최종 JSON 형식으로 변환하여 출력하세요. 부연 설명은 절대 금지합니다.

        [출력 JSON 규격]
        출력은 반드시 아래 JSON 형식만 반환하세요:
        {{
          "apply_degree": true,
          "final_text": "수정된 최종 텍스트",
          "target_expression": "{word_text}",
          "modifier": "추가된 수식어",
          "reason": "반영 이유, 1단계 초안에서 어떤 점이 부자연스러워 2, 3단계를 통해 어떻게 고쳤는지 비판적 검토 과정을 기술"
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