import { createMockAnalysisResult } from "../mocks/mockAnalysisResult";

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export async function analyzeUpload({ file, outputMode }) {
  if (!file) {
    throw new Error("분석할 영상 파일을 먼저 선택해 주세요.");
  }

  await wait(900);
  return createMockAnalysisResult(outputMode, "upload");
}

export async function analyzeWebcam({ outputMode }) {
  await wait(1200);
  return createMockAnalysisResult(outputMode, "webcam");
}
