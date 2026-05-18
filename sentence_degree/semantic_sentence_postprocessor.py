import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "sentence_degree_prompt.txt"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(text: str, degree: str, degree_ko: str) -> str:
    template = load_prompt_template()
    return template.replace("{text}", text).replace("{degree}", degree).replace("{degree_ko}", degree_ko)


def extract_json(response_text: str) -> Dict[str, Any]:
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError("JSON object not found in LLM response")

    return json.loads(match.group(0))


def call_ollama(prompt: str, timeout_sec: int = 30) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result.get("response", "")


def fallback_sentence_degree(text: str, degree: str, reason: str) -> Dict[str, Any]:
    return {
        "apply_degree": False,
        "final_text": text,
        "target_expression": "",
        "modifier": "",
        "reason": reason,
        "source": "fallback"
    }


def validate_llm_result(result: Dict[str, Any], original_text: str, degree: str) -> Dict[str, Any]:
    required_keys = [
        "apply_degree",
        "final_text",
        "target_expression",
        "modifier",
        "reason"
    ]

    for key in required_keys:
        if key not in result:
            raise ValueError(f"missing key: {key}")

    if not isinstance(result["apply_degree"], bool):
        raise ValueError("apply_degree must be boolean")

    if not str(result["final_text"]).strip():
        raise ValueError("final_text is empty")

    if degree == "normal" and result["final_text"] != original_text:
        raise ValueError("degree is normal but final_text was modified")

    return result


def semantic_sentence_postprocess(
    sentence_result: Dict[str, Any],
    degree_result: Dict[str, Any],
    use_llm: bool = True
) -> Dict[str, Any]:
    text = str(sentence_result.get("text", "")).strip()
    degree = str(degree_result.get("degree", "normal")).strip().lower()
    degree_ko = str(degree_result.get("degree_ko", "보통")).strip()

    if not text:
        return fallback_sentence_degree("", degree, "empty sentence text")

    if degree not in {"weak", "normal", "strong"}:
        return fallback_sentence_degree(text, degree, "invalid degree value")

    if degree == "normal":
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "degree가 normal이므로 원문 유지",
            "source": "rule"
        }

    if not use_llm:
        return fallback_sentence_degree(text, degree, "LLM disabled")

    start_time = time.time()

    try:
        prompt = build_prompt(text=text, degree=degree, degree_ko=degree_ko)
        raw_response = call_ollama(prompt)
        parsed = extract_json(raw_response)
        validated = validate_llm_result(parsed, text, degree)

        validated["source"] = "ollama"
        validated["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return validated

    except (urllib.error.URLError, TimeoutError) as e:
        return fallback_sentence_degree(text, degree, f"Ollama connection failed: {e}")

    except Exception as e:
        return fallback_sentence_degree(text, degree, f"LLM response error: {e}")


def build_backend_response(sentence_result: Dict[str, Any], degree_result: Dict[str, Any]) -> Dict[str, Any]:
    semantic_result = semantic_sentence_postprocess(
        sentence_result=sentence_result,
        degree_result=degree_result,
        use_llm=True
    )

    return {
        "mode": "sentence",
        "raw_ai_result": sentence_result,
        "degree_result": degree_result,
        "semantic_llm_result": semantic_result,
        "final_result": {
            "text": semantic_result["final_text"],
            "modified": semantic_result["apply_degree"]
        }
    }


if __name__ == "__main__":
    test_sentence_result = {
        "text": "나는 화났다",
        "confidence": 0.91
    }

    test_degree_result = {
        "degree": "strong",
        "degree_ko": "강함",
        "confidence": 0.82
    }

    result = build_backend_response(test_sentence_result, test_degree_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))