# SignText Frontend

React + Vite 기반 수어 텍스트 변환 웹 UI입니다.

## Folder Structure

```text
frontend/
  package.json
  vite.config.js
  index.html
  src/
    main.jsx
    App.jsx
    styles.css

    api/
      analysisApi.js

    constants/
      inputModes.js
      outputModes.js
      analysisStatus.js

    components/
      AppHeader.jsx
      InputModeTabs.jsx
      MediaWorkspace.jsx
      WebcamPreview.jsx
      UploadPreview.jsx
      KeypointOverlay.jsx
      OutputModeSelector.jsx
      ResultPanel.jsx
      StatusBadge.jsx
      StartIndicatorButton.jsx

    hooks/
      useAnalysisFlow.js
      useWebcamStream.js

    utils/
      formatConfidence.js
      getStatusLabel.js
      getDegreeLabel.js
      recordWebcamClip.js
```

## Current Behavior

- 웹캠 모드에서는 카메라 권한을 요청하고, 버튼을 누르면 약 3초간 녹화한 영상을 백엔드 `/api/predict/webcam`으로 전송합니다.
- 업로드 모드에서는 파일을 선택한 뒤 백엔드 `/api/predict/upload`로 전송합니다.
- 단어, 문장, 표현 정도, 결합 출력 모드는 모두 백엔드의 실제 모델 응답을 사용합니다.
- keypoint 상태, 신뢰도, 모델 상태는 백엔드 MediaPipe 및 모델 응답에서 가져옵니다.
- 개발 서버에서는 `/api` 요청을 `http://127.0.0.1:8000`으로 프록시합니다. 다른 주소는 `VITE_API_BASE_URL`로 설정할 수 있습니다.

## Commands

백엔드가 `http://127.0.0.1:8000`에서 실행 중이고 MediaPipe가 설치되어 있어야 실제 분석 결과를 받을 수 있습니다. 최신 MediaPipe Tasks 환경에서는 `holistic_landmarker.task`를 내려받은 뒤 `MEDIAPIPE_HOLISTIC_TASK_PATH`로 경로를 지정해야 합니다.

```bash
npm install
npm run dev
npm run build
```
