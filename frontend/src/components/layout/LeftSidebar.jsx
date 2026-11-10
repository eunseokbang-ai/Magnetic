import { useApp } from "../../AppContext";
import AutosaveBanner from "../AutosaveBanner";
import WorkflowProgress from "../WorkflowProgress";
import WorkflowSteps from "../WorkflowSteps";

// The left column: project save / load / report, the workflow progress strip,
// and the numbered processing steps (components/WorkflowSteps.jsx).
// A region of App's screen; it reads App's state with useApp().
export default function LeftSidebar() {
  const {
    backendVersion, baseSummary, droneSummary, eulerResult, exportingReport, handleError, handleExportReport,
    handleLoadProject, handleRestoreAutosave, handleSaveProject, handleSetPurposeMode,
    handleWorkflowStepClick, inversionSummary, leftSidebarOpen, loadingProject, overlay, processSummary,
    processing, purposeMode, saveFilename, savingProject, setSaveFilename,
  } = useApp();

  return (
    <div
      className={`app-sidebar-left${leftSidebarOpen ? " open" : ""}`}
      style={{ width: 320, borderRight: "1px solid #e6dac0", overflowY: "auto", padding: 12, background: "#faf6ec" }}
    >
      <h1 style={{ fontSize: 16, margin: "4px 0 0 0" }}>DroneMag Studio</h1>
      <div style={{ fontSize: 11, color: "#8a7a5c", margin: "0 0 2px 0" }}>드론 자력탐사 자료처리</div>
      <div
        style={{ fontSize: 10, color: "#ab9a78", margin: "0 0 12px 0" }}
        title="이 화면이 백엔드 서버로부터 실제로 받은 버전 정보입니다 - '분명히 고쳐달라고 한 문제가 그대로'라면 run.bat을 다시 실행해 이 커밋 해시가 바뀌는지부터 확인하세요"
      >
        {backendVersion?.commit
          ? `버전: ${backendVersion.version ? `v${backendVersion.version} · ` : ""}${backendVersion.commit} (${backendVersion.commit_date})`
          : ""}
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <button
          style={{
            flex: 1,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: purposeMode === "mineral" ? "1px solid #a9631f" : "1px solid #ddd0b2",
            background: purposeMode === "mineral" ? "#a9631f" : "white",
            color: purposeMode === "mineral" ? "white" : "#4a3d28",
            cursor: "pointer",
          }}
          onClick={() => handleSetPurposeMode("mineral")}
          title="넓은 지역의 완만한 지질체 - 3차원 역산/오일러 디컨볼루션 중심, 격자 크기 기본 10m"
        >
          🪨 광물자원탐사
        </button>
        <button
          style={{
            flex: 1,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: purposeMode === "target" ? "1px solid #b91c1c" : "1px solid #ddd0b2",
            background: purposeMode === "target" ? "#b91c1c" : "white",
            color: purposeMode === "target" ? "white" : "#4a3d28",
            cursor: "pointer",
          }}
          onClick={() => handleSetPurposeMode("target")}
          title="작고 국지적인 근지표 표적(지뢰/불발탄/은닉차량) - 쌍극자 피팅 표적탐지 중심, 격자 크기 기본 1m"
        >
          🎯 근지표 표적탐지
        </button>
      </div>
      <WorkflowProgress
        droneSummary={droneSummary}
        baseSummary={baseSummary}
        processSummary={processSummary}
        overlay={overlay}
        inversionSummary={inversionSummary}
        eulerResult={eulerResult}
        onStepClick={handleWorkflowStepClick}
      />
      <AutosaveBanner
        busy={loadingProject || processing}
        hasProject={!!droneSummary}
        onRestore={handleRestoreAutosave}
        onError={handleError}
      />
      <input
        type="text"
        value={saveFilename}
        onChange={(e) => setSaveFilename(e.target.value)}
        placeholder="저장할 파일명 (확장자 제외, 여러 프로젝트는 다른 이름으로 저장)"
        title="여러 프로젝트를 구분해 저장/불러오려면 프로젝트마다 다른 파일명을 지정하세요"
        style={{
          width: "100%",
          marginBottom: 6,
          padding: "5px 8px",
          fontSize: 12,
          borderRadius: 6,
          border: "1px solid #ddd0b2",
          boxSizing: "border-box",
        }}
      />
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <button
          style={{
            flex: 1,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: "1px solid #a9631f",
            background: "white",
            color: "#a9631f",
            cursor: droneSummary && !savingProject ? "pointer" : "default",
            opacity: droneSummary && !savingProject ? 1 : 0.5,
          }}
          disabled={!droneSummary || savingProject}
          onClick={handleSaveProject}
          title="업로드한 자료, 처리 파라미터, 수동 편집, DEM, 역산 결과를 모두 담아 저장합니다"
        >
          {savingProject ? "저장 중..." : "💾 프로젝트 저장"}
        </button>
        <label
          style={{
            flex: 1,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: "1px solid #a9631f",
            background: "white",
            color: "#a9631f",
            cursor: "pointer",
            textAlign: "center",
            opacity: loadingProject ? 0.5 : 1,
          }}
          title="이전에 저장한 프로젝트(.zip)를 불러와 이어서 작업합니다"
        >
          {loadingProject ? "불러오는 중..." : "📂 프로젝트 불러오기"}
          <input
            type="file"
            accept=".zip"
            style={{ display: "none" }}
            disabled={loadingProject}
            onChange={async (e) => {
              const f = e.target.files[0];
              if (!f) return;
              await handleLoadProject(f);
              e.target.value = "";
            }}
          />
        </label>
      </div>
      <button
        style={{
          width: "100%",
          marginBottom: 12,
          padding: "6px 8px",
          fontSize: 12,
          borderRadius: 6,
          border: "1px solid #a9631f",
          background: "white",
          color: "#a9631f",
          cursor: processSummary && !exportingReport ? "pointer" : "default",
          opacity: processSummary && !exportingReport ? 1 : 0.5,
        }}
        disabled={!processSummary || exportingReport}
        onClick={() => handleExportReport("md")}
        title="지금까지 적용된 보정, 측선/통계 요약, 역산·오일러 결과를 정리한 Markdown 보고서를 다운로드합니다"
      >
        {exportingReport ? "생성 중..." : "📄 처리 보고서 다운로드 (.md)"}
      </button>
      <div style={{ display: "flex", gap: 6, margin: "6px 0 10px 0" }} role="group" aria-label="그림이 들어간 보고서">
        {[["docx", "Word"], ["html", "HTML"], ["pdf", "PDF"]].map(([format, name]) => (
          <button
            key={format}
            style={{ flex: 1, padding: "6px 4px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "white", color: "#a9631f",
                     cursor: processSummary && !exportingReport ? "pointer" : "default", opacity: processSummary && !exportingReport ? 1 : 0.5 }}
            disabled={!processSummary || exportingReport}
            onClick={() => handleExportReport(format)}
            title={format === "pdf" ? "지도 그림 모음과 요약 한 장 (보정 내역 전체는 Word·HTML에 있습니다)" : format === "html" ? "그림이 들어간 보고서 한 파일 - 브라우저에서 인쇄하면 PDF가 됩니다" : "그림·표가 들어간 Word 보고서"}
          >
            그림 포함 {name}
          </button>
        ))}
      </div>
      <WorkflowSteps />
    </div>
  );
}
