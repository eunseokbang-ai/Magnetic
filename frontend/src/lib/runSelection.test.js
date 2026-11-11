import { describe, expect, it } from "vitest";
import { clock, runActions, runTitle, summariseRun, transitReason } from "./runSelection";

// two passes of line 3 (a line resumed in a second flight), a ferry leg, a turn
const pt = (id, run, line, extra = {}) => ({
  point_id: id, run_id: run, line_id: line, lat: 34 + id * 1e-5, lon: 126, x: 0, y: id * 10, excluded: line < 0,
  is_ramp: false, timestamp: `2026-11-10 01:00:${String(id).padStart(2, "0")}`, ...extra,
});
const points = [
  pt(0, 0, -1), pt(1, 0, -1), pt(2, 0, -1),          // ferry leg
  pt(3, 1, 3), pt(4, 1, 3), pt(5, 1, 3), pt(6, 1, 3), // first pass of line 3
  pt(7, 2, -1), pt(8, 2, -1),                         // turn
  pt(9, 3, 3), pt(10, 3, 3),                          // line 3 again
  pt(11, 4, -1, { is_ramp: true }),
];
const transitRuns = [{ run_id: 0, verdict: "transit", why: ["fast", "off_direction"], where: "start", speed_mps: 10, line_speed_mps: 5, azimuth_deviation_deg: -9.7 }];

describe("summariseRun", () => {
  it("is the clicked pass only, not the whole line", () => {
    const run = summariseRun(points, 1, transitRuns);
    expect(run.kind).toBe("line");
    expect(run.pointIds).toEqual([3, 4, 5, 6]);
    expect(run.lengthM).toBe(30);
    expect(run.nLinePoints).toBe(6);
    expect([clock(run.timeStart), clock(run.timeEnd)]).toEqual(["01:00:03", "01:00:06"]);
    expect(runTitle(run)).toBe("측선 #3 구간");
  });

  it("tells a ferry leg, a ramp and a turn apart", () => {
    expect(summariseRun(points, 0, transitRuns).kind).toBe("transit");
    expect(summariseRun(points, 2, transitRuns).kind).toBe("other");
    expect(summariseRun(points, 4, transitRuns).kind).toBe("ramp");
    expect(summariseRun(points, 99, transitRuns)).toBeNull();
    expect(summariseRun(points, null, transitRuns)).toBeNull();
  });
});

describe("runActions", () => {
  it("offers this pass or the whole line for a pass of a line flown twice", () => {
    const actions = runActions(summariseRun(points, 1, transitRuns));
    expect(actions.map((a) => [a.action, a.scope])).toEqual([["exclude", "run"], ["exclude", "line"]]);
  });

  it("offers only to put back what is already out", () => {
    const actions = runActions(summariseRun(points, 0, transitRuns));
    expect(actions.map((a) => [a.action, a.scope])).toEqual([["include", "run"]]);
  });

  it("offers both ways for a run that is partly out", () => {
    const partly = points.map((p) => (p.point_id === 4 ? { ...p, excluded: true } : p));
    const actions = runActions(summariseRun(partly, 1, transitRuns));
    expect(actions.map((a) => a.key)).toEqual(["exclude-run", "exclude-line", "include-run"]);
  });
});

describe("transitReason", () => {
  it("says how fast and how far off", () => {
    const text = transitReason(transitRuns[0]);
    expect(text).toContain("2.00배");
    expect(text).toContain("9.7°");
  });
});
