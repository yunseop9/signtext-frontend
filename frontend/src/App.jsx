import { useCallback, useEffect, useMemo, useState } from "react";
import { AppHeader } from "./components/AppHeader";
import { MediaWorkspace } from "./components/MediaWorkspace";
import { ResultPanel } from "./components/ResultPanel";
import { ANALYSIS_STATUS } from "./constants/analysisStatus";
import { INPUT_MODES } from "./constants/inputModes";
import { OUTPUT_MODES } from "./constants/outputModes";
import { useAnalysisFlow } from "./hooks/useAnalysisFlow";
import { useAutoWebcamAnalysis } from "./hooks/useAutoWebcamAnalysis";
import { useKeypointReadiness } from "./hooks/useKeypointReadiness";
import { useWebcamStream } from "./hooks/useWebcamStream";

export default function App() {
  const [inputMode, setInputMode] = useState(INPUT_MODES.WEBCAM);
  const [outputMode, setOutputMode] = useState(OUTPUT_MODES.SENTENCE_DEGREE);
  const [selectedFile, setSelectedFile] = useState(null);

  const {
    status,
    setStatus,
    result,
    errorMessage,
    setErrorMessage,
    uploadKeypoints,
    isAnalyzing,
    resetAnalysis,
    runUploadAnalysis,
    runWebcamAnalysis,
  } = useAnalysisFlow();

  const webcam = useWebcamStream(inputMode === INPUT_MODES.WEBCAM);
  const webcamConnected = webcam.cameraStatus === webcam.CAMERA_STATUS.CONNECTED;
  const keypointReadiness = useKeypointReadiness(inputMode === INPUT_MODES.WEBCAM && webcamConnected);

  const previewUrl = useMemo(() => {
    if (!selectedFile) return "";
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (inputMode !== INPUT_MODES.WEBCAM) return;

    if (webcam.cameraStatus === webcam.CAMERA_STATUS.PERMISSION_DENIED) {
      setStatus(ANALYSIS_STATUS.CAMERA_DENIED);
      setErrorMessage(webcam.cameraError);
      return;
    }

    if (webcamConnected && status === ANALYSIS_STATUS.IDLE) {
      setStatus(ANALYSIS_STATUS.CAMERA_READY);
    }
  }, [
    inputMode,
    setErrorMessage,
    setStatus,
    status,
    webcam.CAMERA_STATUS.PERMISSION_DENIED,
    webcam.cameraError,
    webcam.cameraStatus,
    webcamConnected,
  ]);

  const effectiveKeypoints =
    inputMode === INPUT_MODES.WEBCAM ? keypointReadiness.keypoints : uploadKeypoints;

  const handleInputModeChange = useCallback((nextMode) => {
    setInputMode(nextMode);
    resetAnalysis(ANALYSIS_STATUS.IDLE);
    if (nextMode === INPUT_MODES.UPLOAD) {
      setSelectedFile(null);
    }
  }, [resetAnalysis]);

  const handleUploadStart = useCallback(() => {
    runUploadAnalysis(selectedFile, outputMode);
  }, [outputMode, runUploadAnalysis, selectedFile]);

  const handleAutoWebcamAnalysis = useCallback(() => {
    runWebcamAnalysis(outputMode);
  }, [outputMode, runWebcamAnalysis]);

  useAutoWebcamAnalysis({
    inputMode,
    cameraStatus: webcam.cameraStatus,
    connectedStatus: webcam.CAMERA_STATUS.CONNECTED,
    keypoints: keypointReadiness.keypoints,
    status,
    isAnalyzing,
    setStatus,
    runWebcamAnalysis: handleAutoWebcamAnalysis,
  });

  return (
    <main className="app-shell">
      <AppHeader inputMode={inputMode} onInputModeChange={handleInputModeChange} />

      <div className="workspace-grid">
        <MediaWorkspace
          inputMode={inputMode}
          status={status}
          keypoints={effectiveKeypoints}
          outputMode={outputMode}
          onOutputModeChange={setOutputMode}
          videoRef={webcam.videoRef}
          cameraStatus={webcam.cameraStatus}
          cameraError={webcam.cameraError}
          onCameraRetry={webcam.requestCamera}
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onFileChange={setSelectedFile}
          onUploadStart={handleUploadStart}
          isAnalyzing={isAnalyzing}
        />

        <ResultPanel
          status={status}
          result={result}
          outputMode={outputMode}
          errorMessage={errorMessage}
        />
      </div>
    </main>
  );
}
