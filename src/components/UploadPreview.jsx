export function UploadPreview({ previewUrl, file }) {
  return (
    <div className="preview-frame">
      {previewUrl ? (
        <video className="upload-video" src={previewUrl} controls muted />
      ) : (
        <div className="preview-placeholder">
          <div className="person-mark" />
          <p>분석할 수어 영상을 선택해 주세요</p>
        </div>
      )}
      {file && <span className="file-name">{file.name}</span>}
    </div>
  );
}
