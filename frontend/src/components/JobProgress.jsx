import { useEffect, useRef, useState } from "react";
import * as api from "../api";

// What the backend is doing right now, with a way to stop it.
//
// Processing, gridding, inversion and restoring a project are single
// blocking requests. While one is in flight (api.subscribeBusy) this asks
// GET /api/jobs twice a second and shows the running job's stage, a bar
// when the stage can say how far along it is, the elapsed time, and a
// cancel button. See magcore/progress.py for the other half.
//
// Nothing is shown for the first second of a job: most requests finish
// well inside that, and a bar that flashes for 200 ms is noise.
const SHOW_AFTER_S = 1.0;
const POLL_MS = 500;

export default function JobProgress() {
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState(null);
  const [cancelling, setCancelling] = useState(null); // job id the user asked to stop
  const timer = useRef(null);

  useEffect(() => api.subscribeBusy(setBusy), []);

  useEffect(() => {
    if (!busy) {
      setJob(null);
      return undefined;
    }
    let stopped = false;
    const poll = async () => {
      try {
        const { jobs } = await api.listJobs();
        if (!stopped) setJob(jobs[0] || null);
      } catch {
        // the backend is busy or gone; the request that is in flight will say so
      }
      if (!stopped) timer.current = setTimeout(poll, POLL_MS);
    };
    timer.current = setTimeout(poll, POLL_MS);
    return () => {
      stopped = true;
      clearTimeout(timer.current);
    };
  }, [busy]);

  if (!job || job.elapsed_s < SHOW_AFTER_S) return null;

  const percent = job.fraction == null ? null : Math.round(job.fraction * 100);
  const asked = cancelling === job.id || job.cancel_requested;
  const elapsed = job.elapsed_s < 60 ? `${Math.round(job.elapsed_s)}초` : `${Math.floor(job.elapsed_s / 60)}분 ${Math.round(job.elapsed_s % 60)}초`;

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: "fixed",
        left: "50%",
        bottom: 24,
        transform: "translateX(-50%)",
        zIndex: 3000,
        width: "min(440px, calc(100vw - 32px))",
        background: "#fffaf0",
        border: "1px solid #ddd0b2",
        borderRadius: 10,
        boxShadow: "0 6px 24px rgba(44, 36, 24, 0.22)",
        padding: "10px 12px",
        fontSize: 12,
        color: "#2c2418",
      }}
    >
      <style>{"@keyframes job-progress-slide { 0% { left: -35%; } 100% { left: 100%; } }"}</style>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <strong style={{ fontSize: 13 }}>{job.name}</strong>
        <span style={{ flex: 1, color: "#6b5c42", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {job.stage !== job.name ? job.stage : ""}
        </span>
        <span style={{ color: "#8a7a5c", fontVariantNumeric: "tabular-nums" }}>
          {percent != null ? `${percent}% · ` : ""}
          {elapsed}
        </span>
      </div>
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
        aria-label={`${job.name} 진행률`}
        style={{ position: "relative", height: 6, borderRadius: 3, background: "#eadfc8", overflow: "hidden", margin: "8px 0" }}
      >
        {percent != null ? (
          <div style={{ width: `${percent}%`, height: "100%", background: "#a9631f", borderRadius: 3, transition: "width 0.4s ease" }} />
        ) : (
          // this stage cannot say how far along it is: a moving band, no number
          <div style={{ position: "absolute", top: 0, width: "35%", height: "100%", background: "#a9631f", borderRadius: 3, animation: "job-progress-slide 1.4s linear infinite" }} />
        )}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ flex: 1, color: "#8a7a5c", fontSize: 11 }}>
          {asked ? "취소를 요청했습니다 — 지금 단계가 끝나는 대로 멈춥니다." : "취소하면 이전 결과가 그대로 남습니다."}
        </span>
        <button
          type="button"
          disabled={asked}
          onClick={() => {
            setCancelling(job.id);
            api.cancelJob(job.id).catch(() => setCancelling(null));
          }}
          style={{
            padding: "4px 12px",
            borderRadius: 6,
            border: "1px solid #b45309",
            background: asked ? "#eadfc8" : "white",
            color: asked ? "#8a7a5c" : "#b45309",
            cursor: asked ? "default" : "pointer",
            fontSize: 12,
          }}
        >
          {asked ? "취소 중…" : "취소"}
        </button>
      </div>
    </div>
  );
}
