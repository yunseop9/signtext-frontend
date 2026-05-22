import { useEffect, useRef } from "react";
import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { INPUT_MODES } from "../constants/inputModes";

export function useAutoWebcamAnalysis({
  inputMode,
  cameraStatus,
  connectedStatus,
  keypoints,
  status,
  isAnalyzing,
  setStatus,
  runWebcamAnalysis,
}) {
  const hasStartedRef = useRef(false);

  useEffect(() => {
    if (inputMode !== INPUT_MODES.WEBCAM) {
      hasStartedRef.current = false;
      return;
    }

    if (cameraStatus !== connectedStatus) {
      hasStartedRef.current = false;
      return;
    }

    const ready = keypoints.hands && keypoints.face && keypoints.pose;

    if (!ready) {
      hasStartedRef.current = false;
      if (status !== ANALYSIS_STATUS.LOADING && status !== ANALYSIS_STATUS.CAMERA_DENIED) {
        setStatus(ANALYSIS_STATUS.WAITING_KEYPOINTS);
      }
      return;
    }

    if (!hasStartedRef.current && !isAnalyzing) {
      hasStartedRef.current = true;
      runWebcamAnalysis();
    }
  }, [
    cameraStatus,
    connectedStatus,
    inputMode,
    isAnalyzing,
    keypoints.face,
    keypoints.hands,
    keypoints.pose,
    runWebcamAnalysis,
    setStatus,
    status,
  ]);
}
