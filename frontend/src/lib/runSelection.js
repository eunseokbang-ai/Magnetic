// What a click on a survey point selects, and what the card that opens
// says about it.
//
// The backend numbers every point with the stretch of flight it belongs
// to (run_id - one pass along a line, one ferry leg, one turn, one ramp;
// see project_common.point_run_ids). Selecting by that rather than by line
// id is the point of the tool: a line resumed in a second flight is one
// id and two passes, and what was excluded has no line id at all.

const KIND_LABEL = {
  line: "측선 구간",
  transit: "이동 구간 (자동 제외)",
  ramp: "이착륙 구간",
  other: "선회·제외 구간",
};

// Everything the card and the highlight need about one run, from the
// point list the map already holds. `transitRuns` is processSummary.transit.runs.
export function summariseRun(points, runId, transitRuns = []) {
  if (runId == null || !points) return null;
  const own = [];
  for (let i = 0; i < points.length; i++) if (points[i].run_id === runId) own.push(points[i]);
  if (own.length === 0) return null;

  const first = own[0];
  const lineId = first.line_id;
  let lengthM = 0;
  let nExcluded = 0;
  for (let i = 0; i < own.length; i++) {
    if (own[i].excluded) nExcluded++;
    if (i > 0) lengthM += Math.hypot(own[i].x - own[i - 1].x, own[i].y - own[i - 1].y);
  }
  // how much of the same line lies outside this run (other passes, other flights)
  let nLinePoints = 0;
  if (lineId >= 0) for (let i = 0; i < points.length; i++) if (points[i].line_id === lineId) nLinePoints++;

  const transit = transitRuns.find((r) => r.run_id === runId) || null;
  const kind = lineId >= 0 ? "line" : transit && transit.verdict === "transit" ? "transit" : first.is_ramp ? "ramp" : "other";
  return {
    runId,
    kind,
    lineId: lineId >= 0 ? lineId : null,
    isRamp: !!first.is_ramp,
    pointIds: own.map((p) => p.point_id),
    nPoints: own.length,
    nExcluded,
    lengthM,
    nLinePoints,
    latlngs: own.map((p) => [p.lat, p.lon]),
    timeStart: first.timestamp,
    timeEnd: own[own.length - 1].timestamp,
    transit,
  };
}

export function runTitle(run) {
  if (run.kind === "line") return `측선 #${run.lineId} 구간`;
  return KIND_LABEL[run.kind];
}

// "why" codes of a ferry leg or a suspect run, in words
export function transitReason(run) {
  if (!run) return "";
  const parts = [];
  const ratio = run.line_speed_mps ? run.speed_mps / run.line_speed_mps : null;
  if (run.why.includes("fast")) {
    parts.push(`속도 ${run.speed_mps.toFixed(1)} m/s` + (ratio ? ` (측선 순항 속도 ${run.line_speed_mps.toFixed(1)} m/s의 ${ratio.toFixed(2)}배)` : ""));
  }
  if (run.why.includes("off_direction")) parts.push(`측선 방향에서 ${Math.abs(run.azimuth_deviation_deg).toFixed(1)}° 벗어남`);
  if (run.why.includes("whole_flight")) parts.push("이 비행 전체가 해당 — 다른 방향으로 비행한 구역일 수 있어 그대로 둠");
  return parts.join(", ");
}

const WHERE_LABEL = { start: "측선 시작 전", end: "측선 종료 후", middle: "측선 사이", flight: "비행 전체" };

export function transitWhere(run) {
  return WHERE_LABEL[run.where] || "";
}

// hh:mm:ss out of the timestamp strings the points carry
export function clock(timestamp) {
  const m = /(\d{2}:\d{2}:\d{2})/.exec(String(timestamp || ""));
  return m ? m[1] : "";
}

// The choices the card offers for a run, in the order shown. Each is what
// gets sent: { label, action: "exclude" | "include", scope: "run" | "line", danger }.
export function runActions(run) {
  const actions = [];
  const nIncluded = run.nPoints - run.nExcluded;
  if (nIncluded > 0) actions.push({ key: "exclude-run", label: `이 구간 제외 (${nIncluded.toLocaleString()}점)`, action: "exclude", scope: "run", danger: true });
  if (run.kind === "line" && run.nLinePoints > run.nPoints) {
    actions.push({ key: "exclude-line", label: `측선 #${run.lineId} 전체 제외 (${run.nLinePoints.toLocaleString()}점)`, action: "exclude", scope: "line", danger: true });
  }
  if (run.nExcluded > 0) actions.push({ key: "include-run", label: `이 구간 다시 포함 (${run.nExcluded.toLocaleString()}점)`, action: "include", scope: "run", danger: false });
  return actions;
}
