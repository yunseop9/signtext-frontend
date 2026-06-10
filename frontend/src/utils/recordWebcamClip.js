const DEFAULT_RECORDING_DURATION_MS = 5000;


function getSupportedMimeType() {
  const candidates = [
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];

  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}


function createMirroredStream(sourceStream) {
  const sourceVideo = document.createElement("video");
  sourceVideo.srcObject = sourceStream;
  sourceVideo.muted = true;
  sourceVideo.playsInline = true;

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const sourceTrack = sourceStream.getVideoTracks()[0];
  const settings = sourceTrack?.getSettings?.() ?? {};
  const fps = settings.frameRate || 30;

  let animationFrameId = null;
  let stopped = false;

  const drawFrame = () => {
    if (stopped) return;

    if (sourceVideo.videoWidth && sourceVideo.videoHeight && context) {
      if (canvas.width !== sourceVideo.videoWidth) {
        canvas.width = sourceVideo.videoWidth;
      }
      if (canvas.height !== sourceVideo.videoHeight) {
        canvas.height = sourceVideo.videoHeight;
      }

      context.save();
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
      context.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
      context.restore();
    }

    animationFrameId = window.requestAnimationFrame(drawFrame);
  };

  const mirroredStream = canvas.captureStream(fps);
  sourceVideo.play().catch(() => {});
  drawFrame();

  return {
    stream: mirroredStream,
    stop: () => {
      stopped = true;
      if (animationFrameId) {
        window.cancelAnimationFrame(animationFrameId);
      }
      mirroredStream.getTracks().forEach((track) => track.stop());
      sourceVideo.srcObject = null;
    },
  };
}


export function recordWebcamClip(
  stream,
  durationMs = DEFAULT_RECORDING_DURATION_MS,
  signal,
) {
  if (!stream?.active) {
    return Promise.reject(new Error("연결된 웹캠 스트림이 없습니다."));
  }

  if (!window.MediaRecorder) {
    return Promise.reject(
      new Error("현재 브라우저에서는 웹캠 영상 녹화를 지원하지 않습니다."),
    );
  }

  const mimeType = getSupportedMimeType();
  const options = mimeType ? { mimeType } : undefined;

  return new Promise((resolve, reject) => {
    const mirrored = createMirroredStream(stream);
    const chunks = [];
    const recorder = new MediaRecorder(mirrored.stream, options);
    let timerId;
    let aborted = false;

    const abortRecording = () => {
      aborted = true;
      window.clearTimeout(timerId);
      if (recorder.state !== "inactive") {
        recorder.stop();
      }
    };

    const cleanup = () => {
      window.clearTimeout(timerId);
      signal?.removeEventListener("abort", abortRecording);
      mirrored.stop();
    };

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    });

    recorder.addEventListener("error", (event) => {
      cleanup();
      reject(event.error ?? new Error("웹캠 녹화 중 오류가 발생했습니다."));
    });

    recorder.addEventListener("stop", () => {
      cleanup();

      if (aborted) {
        reject(new DOMException("Recording was stopped.", "AbortError"));
        return;
      }

      if (chunks.length === 0) {
        reject(new Error("웹캠 녹화 영상이 비어 있습니다."));
        return;
      }

      resolve(
        new Blob(chunks, {
          type: recorder.mimeType || mimeType || "video/webm",
        }),
      );
    });

    recorder.start(250);
    timerId = window.setTimeout(() => {
      if (recorder.state !== "inactive") {
        recorder.stop();
      }
    }, durationMs);

    signal?.addEventListener("abort", abortRecording, { once: true });
  });
}
