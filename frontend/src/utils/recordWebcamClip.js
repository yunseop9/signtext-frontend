const DEFAULT_RECORDING_DURATION_MS = 5000;


function getSupportedMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];

  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}


function waitForVideoReady(video) {
  if (video.videoWidth && video.videoHeight) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      cleanup();
      reject(new Error("웹캠 영상 준비 시간이 초과되었습니다."));
    }, 3000);

    const cleanup = () => {
      window.clearTimeout(timeoutId);
      video.removeEventListener("loadedmetadata", handleReady);
      video.removeEventListener("canplay", handleReady);
    };

    const handleReady = () => {
      if (!video.videoWidth || !video.videoHeight) return;
      cleanup();
      resolve();
    };

    video.addEventListener("loadedmetadata", handleReady);
    video.addEventListener("canplay", handleReady);
  });
}


async function createMirroredStream(sourceStream) {
  const sourceVideo = document.createElement("video");
  sourceVideo.srcObject = sourceStream;
  sourceVideo.muted = true;
  sourceVideo.playsInline = true;
  await sourceVideo.play();
  await waitForVideoReady(sourceVideo);

  const canvas = document.createElement("canvas");
  canvas.width = sourceVideo.videoWidth;
  canvas.height = sourceVideo.videoHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("좌우반전 녹화 캔버스를 생성하지 못했습니다.");
  }

  const sourceTrack = sourceStream.getVideoTracks()[0];
  const settings = sourceTrack?.getSettings?.() ?? {};
  const fps = Math.min(Math.max(Math.round(settings.frameRate || 30), 10), 30);
  let timerId = null;

  const drawFrame = () => {
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
    context.restore();
  };

  drawFrame();
  timerId = window.setInterval(drawFrame, Math.round(1000 / fps));
  const mirroredStream = canvas.captureStream(fps);

  return {
    stream: mirroredStream,
    stop: () => {
      if (timerId) {
        window.clearInterval(timerId);
      }
      mirroredStream.getTracks().forEach((track) => track.stop());
      sourceVideo.pause();
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

  return new Promise((resolve, reject) => {
    let recorder = null;
    let mirrored = null;
    let timerId = null;
    let aborted = false;
    const chunks = [];

    const cleanup = () => {
      window.clearTimeout(timerId);
      signal?.removeEventListener("abort", abortRecording);
      mirrored?.stop();
    };

    const abortRecording = () => {
      aborted = true;
      window.clearTimeout(timerId);
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      } else {
        cleanup();
        reject(new DOMException("Recording was stopped.", "AbortError"));
      }
    };

    const startRecording = async () => {
      try {
        mirrored = await createMirroredStream(stream);
        const mimeType = getSupportedMimeType();
        const options = mimeType ? { mimeType } : undefined;
        recorder = new MediaRecorder(mirrored.stream, options);

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

        signal?.addEventListener("abort", abortRecording, { once: true });
        recorder.start(250);
        timerId = window.setTimeout(() => {
          if (recorder.state !== "inactive") {
            recorder.stop();
          }
        }, durationMs);
      } catch (error) {
        cleanup();
        reject(error);
      }
    };

    startRecording();
  });
}
