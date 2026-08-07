const STEPS = [
  { key: "drone", label: "드론 자료 업로드", sectionId: "wf-section-drone" },
  { key: "base", label: "베이스 자료 업로드", sectionId: "wf-section-base" },
  { key: "process", label: "자료 처리 실행", sectionId: "wf-section-process" },
  { key: "grid", label: "그리드/시각화 생성", sectionId: "wf-section-grid" },
  { key: "advanced", label: "(선택) 3차원 역산 · 오일러 디컨볼루션", sectionId: "wf-section-advanced" },
];

export default function WorkflowProgress({ droneSummary, baseSummary, processSummary, overlay, inversionSummary, eulerResult, onStepClick }) {
  const done = {
    drone: !!droneSummary,
    base: !!baseSummary,
    process: !!processSummary,
    grid: !!overlay,
    advanced: !!inversionSummary || !!eulerResult,
  };
  // the first not-yet-done step is "current"; once everything through grid
  // is done, the optional advanced step just stays available (not "next")
  const currentKey = STEPS.find((s) => !done[s.key] && s.key !== "advanced")?.key;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 12, fontSize: 11 }}>
      {STEPS.map((s, i) => {
        const isDone = done[s.key];
        const isCurrent = s.key === currentKey;
        return (
          <div
            key={s.key}
            onClick={() => onStepClick && onStepClick(s.sectionId)}
            title="클릭하면 해당 단계로 이동합니다"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              cursor: onStepClick ? "pointer" : "default",
              borderRadius: 4,
              padding: "2px 4px",
              marginLeft: -4,
            }}
            onMouseEnter={(e) => onStepClick && (e.currentTarget.style.background = "#eef2ff")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <div
              style={{
                width: 16,
                height: 16,
                borderRadius: "50%",
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 10,
                fontWeight: 700,
                background: isDone ? "#22c55e" : isCurrent ? "#2563eb" : "#e5e7eb",
                color: isDone || isCurrent ? "white" : "#9ca3af",
              }}
            >
              {isDone ? "✓" : i + 1}
            </div>
            <span style={{ color: isDone ? "#16a34a" : isCurrent ? "#2563eb" : "#9ca3af", fontWeight: isCurrent ? 600 : 400 }}>
              {s.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
