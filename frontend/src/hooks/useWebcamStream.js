import { useCallback, useEffect, useRef, useState } from "react";

const CAMERA_STATUS = {
  NOT_REQUESTED: "not_requested",
  REQUESTING: "requesting",
  CONNECTED: "connected",
  PERMISSION_DENIED: "permission_denied",
  DISCONNECTED: "disconnected",
};

export function useWebcamStream(enabled) {
  const [stream, setStream] = useState(null);
  const [cameraStatus, setCameraStatus] = useState(CAMERA_STATUS.NOT_REQUESTED);
  const [cameraError, setCameraError] = useState("");
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setStream(null);
    setCameraStatus(CAMERA_STATUS.DISCONNECTED);
  }, []);

  const requestCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus(CAMERA_STATUS.PERMISSION_DENIED);
      setCameraError("이 브라우저에서는 카메라 기능을 사용할 수 없습니다.");
      return;
    }

    setCameraStatus(CAMERA_STATUS.REQUESTING);
    setCameraError("");

    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = nextStream;
      setStream(nextStream);
      setCameraStatus(CAMERA_STATUS.CONNECTED);
    } catch (error) {
      const denied = error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError";
      setStream(null);
      setCameraStatus(denied ? CAMERA_STATUS.PERMISSION_DENIED : CAMERA_STATUS.DISCONNECTED);
      setCameraError(
        denied
          ? "카메라 권한이 거부되었습니다. 브라우저 설정에서 권한을 허용해 주세요."
          : "카메라를 연결할 수 없습니다.",
      );
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      stopCamera();
      setCameraStatus(CAMERA_STATUS.NOT_REQUESTED);
      return;
    }

    requestCamera();

    return () => {
      stopCamera();
    };
  }, [enabled, requestCamera, stopCamera]);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return {
    videoRef,
    stream,
    cameraStatus,
    cameraError,
    requestCamera,
    stopCamera,
    CAMERA_STATUS,
  };
}
