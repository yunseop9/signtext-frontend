import { OUTPUT_MODES } from "../constants/outputModes";
import { createMockAnalysisResult } from "../mocks/mockAnalysisResult";

const API_BASE_URL = "http://127.0.0.1:8000";

function getBackendMode(outputMode) {
  if (
    outputMode === OUTPUT_MODES.SENTENCE ||
    outputMode === OUTPUT_MODES.SENTENCE_DEGREE
  ) {
    return "sentence";
  }

  return "word";
}

function mapBackendResultToFrontendResult(backendResult, outputMode, source = "upload") {
  const raw = backendResult.raw_ai_result ?? {};
  const degreeResult = backendResult.degree_result ?? {};
  const semantic = backendResult.semantic_llm_result ?? {};
  const final = backendResult.final_result ?? {};

  const rawText = raw.text ?? "";
  const finalText = final.text ?? semantic.final_text ?? rawText;
  const degree = degreeResult.degree ?? "normal";
  const confidence = raw.confidence ?? degreeResult.confidence ?? 0;

  return {
    status: backendResult.status ?? "success",
    source,
    mode: outputMode,

    text: finalText,
    word: rawText,
    sentence: rawText,

    degree,
    confidence,

    llmFallback: semantic.apply_degree === false && outputMode === OUTPUT_MODES.SENTENCE_DEGREE,
    fallbackReason: semantic.reason ?? "",

    finalText,
    originalText: rawText,

    history: [
      {
        text: finalText,
        degree,
        confidence,
      },
    ],

    keypoints: {
      hands: true,
      face: true,
      pose: true,
    },

    backendRaw: backendResult,
  };
}

export async function analyzeUpload({ file, outputMode }) {
  if (!file) {
    throw new Error("분석할 영상 파일을 먼저 선택해 주세요.");
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", getBackendMode(outputMode));

  const response = await fetch(`${API_BASE_URL}/api/predict/upload`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok || data.status === "error") {
    throw new Error(data.detail ?? data.message ?? "업로드한 영상 분석에 실패했습니다.");
  }

  return mapBackendResultToFrontendResult(data, outputMode, "upload");
}

export async function analyzeWebcam({ imageBlob, outputMode }) {
  if (!imageBlob) {
    // 웹캠 캡처가 실패했을 때만 임시 mock으로 fallback
    return createMockAnalysisResult(outputMode, "webcam");
  }

  const formData = new FormData();
  formData.append("file", imageBlob, "webcam-frame.jpg");
  formData.append("mode", getBackendMode(outputMode));

  const response = await fetch(`${API_BASE_URL}/api/predict/webcam-frame`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok || data.status === "error") {
    throw new Error(data.detail ?? data.message ?? "웹캠 분석에 실패했습니다.");
  }

  return mapBackendResultToFrontendResult(data, outputMode, "webcam");
}