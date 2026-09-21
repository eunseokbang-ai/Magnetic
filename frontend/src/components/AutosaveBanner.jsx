import { useEffect, useState } from "react";
import * as api from "../api";

function when(saved_at) {
  if (!saved_at) return "";
  const d = new Date(saved_at * 1000);
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  const stamp = d.toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  if (mins < 1) return `방금 · ${stamp}`;
  if (mins < 60) return `${mins}분 전 · ${stamp}`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `${hours}시간 전 · ${stamp}`;
  return stamp;
}

/**
 * Offers the projects the backend saved by itself, so a session that
 * ended badly (closed window, crash, restart) can be picked up where it
 * stopped. Hidden once there is a project in progress on screen - the
 * point is to catch the empty start, not to nag mid-work.
 */
export default function AutosaveBanner({ busy, hasProject, onRestore, onError }) {
  const [saves, setSaves] = useState([]);
  const [dismissed, setDismissed] = useState(false);
  const [working, setWorking] = useState(null);

  useEffect(() => {
    api
      .listAutosaves()
      .then((r) => setSaves(r.autosaves || []))
      .catch(() => setSaves([]));
  }, []);

  if (dismissed || hasProject || saves.length === 0) return null;

  const restore = async (s) => {
    try {
      setWorking(s.project_id);
      await onRestore(s.project_id);
      setDismissed(true);
    } catch (e) {
      onError && onError(e);
    } finally {
      setWorking(null);
    }
  };

  const remove = async (s) => {
    try {
      const r = await api.deleteAutosave(s.project_id);
      setSaves(r.autosaves || []);
    } catch (e) {
      onError && onError(e);
    }
  };

  return (
    <div
      style={{
        border: "1px solid #ddd0b2",
        background: "#fdf8ee",
        borderRadius: 8,
        padding: 10,
        marginBottom: 12,
        fontSize: 12,
        color: "#4a3d28",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <b>이전 작업 이어서 하기</b>
        <span style={{ color: "#8a7a5c" }}>자동 저장된 프로젝트 {saves.length}개</span>
        <button
          style={{ marginLeft: "auto", border: "none", background: "none", cursor: "pointer", color: "#8a7a5c" }}
          onClick={() => setDismissed(true)}
          title="이번 세션에서는 다시 묻지 않습니다"
        >
          ✕
        </button>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {saves.map((s) => (
          <div key={s.project_id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ flex: 1, lineHeight: 1.35 }}>
              <div>{when(s.saved_at)}</div>
              <div style={{ color: "#8a7a5c", fontSize: 11 }}>
                측점 {s.n_points ? s.n_points.toLocaleString() : 0}개
                {s.time_range ? ` · ${s.time_range[0].slice(0, 10)}` : ""}
                {s.processed ? " · 처리 완료" : " · 처리 전"}
                {s.n_source_removals ? ` · 구조물 ${s.n_source_removals}건 제거` : ""}
                {s.n_smoothed_points ? ` · 스무딩 ${s.n_smoothed_points}점` : ""}
              </div>
            </div>
            <button
              style={{
                padding: "4px 8px",
                fontSize: 12,
                borderRadius: 6,
                border: "1px solid #a9631f",
                background: "white",
                color: "#a9631f",
                cursor: busy || working ? "default" : "pointer",
                opacity: busy || working ? 0.5 : 1,
              }}
              disabled={busy || !!working}
              onClick={() => restore(s)}
            >
              {working === s.project_id ? "복구 중..." : "복구"}
            </button>
            <button
              style={{ border: "none", background: "none", cursor: "pointer", color: "#8a7a5c", fontSize: 12 }}
              disabled={busy || !!working}
              onClick={() => remove(s)}
              title="이 자동 저장본을 지웁니다"
            >
              삭제
            </button>
          </div>
        ))}
      </div>
      <div style={{ color: "#8a7a5c", fontSize: 11, marginTop: 6 }}>
        복구하면 저장 당시의 측점 자료·처리 설정·수동 편집·구조물 모델이 그대로 돌아오고, 처리는 다시 실행됩니다.
      </div>
    </div>
  );
}
