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
    const chunks = [];
    const recorder = new MediaRecorder(stream, options);
    let timerId;
    let aborted = false;

    const abortRecording = () => {
      aborted = true;
      window.clearTimeout(timerId);
      if (recorder.state !== "inactive") {
        recorder.stop();
      }
    };

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    });

    recorder.addEventListener("error", (event) => {
      window.clearTimeout(timerId);
      signal?.removeEventListener("abort", abortRecording);
      reject(event.error ?? new Error("웹캠 녹화 중 오류가 발생했습니다."));
    });

    recorder.addEventListener("stop", () => {
      window.clearTimeout(timerId);
      signal?.removeEventListener("abort", abortRecording);

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
