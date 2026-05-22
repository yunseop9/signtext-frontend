import { InputModeTabs } from "./InputModeTabs";

export function AppHeader({ inputMode, onInputModeChange }) {
  return (
    <header className="app-header">
      <nav className="main-navigation" aria-label="상단 네비게이션">
        <a className="nav-link active" href="#converter">수어 변환</a>
        <a className="nav-link" href="#service-intro">서비스 소개</a>
        <a className="nav-link" href="#usage-guide">사용 방법</a>
        <a className="nav-link" href="#conversation-history">대화 기록 저장</a>
      </nav>
      <InputModeTabs value={inputMode} onChange={onInputModeChange} />
    </header>
  );
}
