import { ANALYSIS_STATUS } from "../constants/analysisStatus";

export function WebcamPreview({ videoRef, cameraStatus, cameraError, onRetry, status }) {
  const denied = status === ANALYSIS_STATUS.CAMERA_DENIED;

  return (
    <div className="preview-frame">
      <video ref={videoRef} className="webcam-video" autoPlay playsInline muted />
      {!denied && cameraStatus !== "connected" && (
        <div className="preview-placeholder">
          <div className="person-mark" />
          <p>카메라를 준비하고 있습니다</p>
        </div>
      )}
      {denied && (
        <div className="permission-panel">
          <p>{cameraError || "카메라 권한이 거부되었습니다."}</p>
          <button type="button" onClick={onRetry}>
            권한 재요청
          </button>
        </div>
      )}
    </div>
  );
}
