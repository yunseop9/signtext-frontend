import { OUTPUT_MODES } from "../constants/outputModes";

const HISTORY = [
  { text: "감사합니다.", degree: "normal", confidence: 0.92 },
  { text: "잠깐만요.", degree: "weak", confidence: 0.74 },
  { text: "괜찮아요.", degree: "strong", confidence: 0.81 },
];

export function createMockAnalysisResult(outputMode, source = "webcam") {
  const fallback = outputMode === OUTPUT_MODES.SENTENCE_DEGREE;
  const text = fallback ? "지금 몸이 너무 아파서 빨리 도와주세요." : "안녕하세요.";

  return {
    status: "success",
    source,
    mode: outputMode,
    text,
    word: fallback ? "도와주세요" : "안녕하세요",
    sentence: text,
    degree: fallback ? "strong" : "normal",
    confidence: fallback ? 0.71 : 0.87,
    llmFallback: fallback,
    fallbackReason: fallback ? "LLM 후처리에 실패하여 원문 결과를 표시합니다." : "",
    finalText: text,
    originalText: text,
    history: fallback ? HISTORY.slice(0, 2) : HISTORY,
    keypoints: {
      hands: true,
      face: true,
      pose: source === "upload" ? true : false,
    },
  };
}
