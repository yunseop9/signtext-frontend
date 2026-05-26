import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { OUTPUT_MODES } from "../constants/outputModes";
import { formatConfidence } from "../utils/formatConfidence";
import { getDegreeLabel } from "../utils/getDegreeLabel";
import { StatusBadge } from "./StatusBadge";

function getPrimaryText(result, outputMode) {
  if (!result) return "";

  if (outputMode === OUTPUT_MODES.WORD) return result.word;
  if (outputMode === OUTPUT_MODES.SENTENCE) return result.sentence;
  if (outputMode === OUTPUT_MODES.DEGREE) return getDegreeLabel(result.degree);
  if (outputMode === OUTPUT_MODES.WORD_DEGREE) return result.word;
  return result.finalText || result.sentence || result.text;
}

function shouldShowDegree(outputMode) {
  return [
    OUTPUT_MODES.DEGREE,
    OUTPUT_MODES.WORD_DEGREE,
    OUTPUT_MODES.SENTENCE_DEGREE,
  ].includes(outputMode);
}

function getDegreeBadgeLabel(degree) {
  return getDegreeLabel(degree);
}

function EmptyState({ status, errorMessage }) {
  const messages = {
    [ANALYSIS_STATUS.IDLE]: ["대기 중", "웹캠을 선택하거나 영상을 업로드해 주세요."],
    [ANALYSIS_STATUS.CAMERA_READY]: ["대기 중", "손, 얼굴, 신체가 모두 인식되면 자동으로 분석을 시작합니다."],
    [ANALYSIS_STATUS.WAITING_KEYPOINTS]: ["인식 대기 중", "손, 얼굴, 신체가 모두 화면 안에 들어오도록 자세와 거리를 조정해 주세요."],
    [ANALYSIS_STATUS.LOADING]: ["분석 중", "수어 동작을 분석하고 있습니다"],
    [ANALYSIS_STATUS.ERROR]: ["분석 실패", errorMessage || "결과를 가져올 수 없습니다."],
    [ANALYSIS_STATUS.CAMERA_DENIED]: ["카메라 권한이 거부되었습니다", "브라우저 주소창 옆 카메라 아이콘에서 권한을 허용해 주세요."],
  };

  const [title, description] = messages[status] ?? messages[ANALYSIS_STATUS.IDLE];

  return (
    <div className="empty-result">
      <StatusBadge status={status} />
      <h2>{title}</h2>
      <p>{description}</p>
      {status === ANALYSIS_STATUS.LOADING && <span className="loading-line" />}
    </div>
  );
}

export function ResultPanel({ status, result, outputMode, errorMessage }) {
  const resultReady = [ANALYSIS_STATUS.SUCCESS, ANALYSIS_STATUS.LLM_FALLBACK].includes(status) && result;

  if (!resultReady) {
    return (
      <aside className="result-panel" aria-label="분석 결과">
        <EmptyState status={status} errorMessage={errorMessage} />
      </aside>
    );
  }

  const primaryText = getPrimaryText(result, outputMode);
  const degree = getDegreeLabel(result.degree);
  const confidence = formatConfidence(result.confidence);
  const showDegree = shouldShowDegree(outputMode);

  return (
    <aside className="result-panel" aria-label="분석 결과">
      <div className="result-main">
        <h2>{primaryText}</h2>
        {showDegree && (
          <div className={`degree-badge ${degree}`}>
            {getDegreeBadgeLabel(result.degree)}
          </div>
        )}
        <div className="result-meta">
          <span className="confidence-bar">
            <span style={{ width: confidence === "-" ? "0%" : confidence }} />
          </span>
          <strong>{confidence}</strong>
          <span>신뢰도</span>
        </div>
      </div>
    </aside>
  );
}
