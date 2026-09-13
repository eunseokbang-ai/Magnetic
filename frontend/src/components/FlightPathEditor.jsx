import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Dedicated flight-path editor: plots every point in local projected (x, y)
// meters - not lat/lon on the web map - so straight survey/tie lines look
// straight and rectangle selections line up cleanly with them, mirroring
// DroneMagAdv's FlightPathDlg. Kept intentionally simpler than that dialog
// (no tilted-rectangle/parallelogram select, no PCA-angle guide line): pan,
// zoom, and axis-aligned rectangle select cover the actual editing need
// (excluding/restoring a burst of points) without the extra interaction
// surface.

const COLOR_SURVEY = "#a9631f";
const COLOR_TIE = "#f59e0b";
const COLOR_EXCLUDED = "#ab9a78";
const PADDING_FRACTION = 0.92;

function pointColor(p) {
  if (p.excluded) return COLOR_EXCLUDED;
  if (p.tie_line_id != null && p.tie_line_id >= 0) return COLOR_TIE;
  return COLOR_SURVEY;
}

function fitView(points, width, height) {
  const finite = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (finite.length === 0) return { viewX0: 0, viewY0: 0, scale: 1 };
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of finite) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const scale = Math.min(width / spanX, height / spanY) * PADDING_FRACTION;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  return {
    viewX0: centerX - width / (2 * scale),
    viewY0: centerY - height / (2 * scale),
    scale,
  };
}

const modeButtonStyle = (active) => ({
  padding: "6px 10px",
  fontSize: 12,
  borderRadius: 6,
  border: active ? "1px solid #a9631f" : "1px solid #ddd0b2",
  background: active ? "#faf0e2" : "white",
  color: active ? "#a9631f" : "#4a3d28",
  cursor: "pointer",
});

export default function FlightPathEditor({ points, loading, onApply, onClose }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 800, height: 560 });
  const [view, setView] = useState(null); // { viewX0, viewY0, scale }
  const [mode, setMode] = useState("pan"); // "pan" | "cut" | "restore"
  const dragRef = useRef(null); // { startPx: {x,y}, lastPx: {x,y}, kind }
  const [selectionRectPx, setSelectionRectPx] = useState(null);

  const validPoints = useMemo(() => (points || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)), [points]);

  // fit-to-data once we know the canvas size and have points
  useEffect(() => {
    if (view || validPoints.length === 0 || size.width === 0) return;
    setView(fitView(validPoints, size.width, size.height));
  }, [validPoints, size, view]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const dataToCanvas = useCallback(
    (x, y) => {
      if (!view) return { cx: 0, cy: 0 };
      return { cx: (x - view.viewX0) * view.scale, cy: size.height - (y - view.viewY0) * view.scale };
    },
    [view, size.height]
  );

  const canvasToData = useCallback(
    (cx, cy) => {
      if (!view) return { x: 0, y: 0 };
      return { x: view.viewX0 + cx / view.scale, y: view.viewY0 + (size.height - cy) / view.scale };
    },
    [view, size.height]
  );

  // redraw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !view) return;
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#2c2418";
    ctx.fillRect(0, 0, size.width, size.height);

    for (const p of validPoints) {
      const { cx, cy } = dataToCanvas(p.x, p.y);
      if (cx < -5 || cx > size.width + 5 || cy < -5 || cy > size.height + 5) continue;
      ctx.fillStyle = pointColor(p);
      ctx.fillRect(cx - 1, cy - 1, 2, 2);
    }

    if (selectionRectPx) {
      const { x0, y0, x1, y1 } = selectionRectPx;
      ctx.strokeStyle = mode === "restore" ? "#16a34a" : "#dc2626";
      ctx.fillStyle = mode === "restore" ? "rgba(22,163,74,0.15)" : "rgba(220,38,38,0.15)";
      const rx = Math.min(x0, x1);
      const ry = Math.min(y0, y1);
      const rw = Math.abs(x1 - x0);
      const rh = Math.abs(y1 - y0);
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
    }
  }, [validPoints, view, size, dataToCanvas, selectionRectPx, mode]);

  const handleWheel = (e) => {
    e.preventDefault();
    if (!view) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const before = canvasToData(mx, my);
    const factor = e.deltaY < 0 ? 1.25 : 0.8;
    const newScale = Math.min(Math.max(view.scale * factor, 1e-4), 500);
    const newViewX0 = before.x - mx / newScale;
    const newViewY0 = before.y - (size.height - my) / newScale;
    setView({ viewX0: newViewX0, viewY0: newViewY0, scale: newScale });
  };

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const px = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    dragRef.current = { startPx: px, lastPx: px, kind: mode };
    if (mode !== "pan") setSelectionRectPx({ x0: px.x, y0: px.y, x1: px.x, y1: px.y });
  };

  const handleMouseMove = (e) => {
    if (!dragRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const px = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    if (dragRef.current.kind === "pan") {
      const dx = px.x - dragRef.current.lastPx.x;
      const dy = px.y - dragRef.current.lastPx.y;
      setView((v) => (v ? { ...v, viewX0: v.viewX0 - dx / v.scale, viewY0: v.viewY0 + dy / v.scale } : v));
    } else {
      setSelectionRectPx((r) => (r ? { ...r, x1: px.x, y1: px.y } : r));
    }
    dragRef.current.lastPx = px;
  };

  const finishDrag = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag || drag.kind === "pan") return;
    setSelectionRectPx((r) => {
      if (!r) return null;
      const cx0 = Math.min(r.x0, r.x1);
      const cx1 = Math.max(r.x0, r.x1);
      const cy0 = Math.min(r.y0, r.y1);
      const cy1 = Math.max(r.y0, r.y1);
      if (cx1 - cx0 < 3 && cy1 - cy0 < 3) return null; // treat as a click, not a drag - ignore
      const a = canvasToData(cx0, cy0);
      const b = canvasToData(cx1, cy1);
      const xMin = Math.min(a.x, b.x);
      const xMax = Math.max(a.x, b.x);
      const yMin = Math.min(a.y, b.y);
      const yMax = Math.max(a.y, b.y);
      const action = drag.kind === "restore" ? "include" : "exclude";
      const idsToApply = validPoints
        .filter((p) => p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax && !!p.excluded !== (action === "exclude"))
        .map((p) => p.point_id);
      if (idsToApply.length > 0) onApply(idsToApply, action);
      return null;
    });
  };

  const nExcluded = validPoints.filter((p) => p.excluded).length;
  const nTie = validPoints.filter((p) => !p.excluded && p.tie_line_id != null && p.tie_line_id >= 0).length;
  const nSurvey = validPoints.length - nExcluded - nTie;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        zIndex: 2000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onMouseUp={finishDrag}
    >
      <div
        style={{
          background: "white",
          borderRadius: 10,
          padding: 12,
          width: "90vw",
          height: "88vh",
          maxWidth: 1400,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          boxShadow: "0 8px 30px rgba(0,0,0,0.3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <h3 style={{ margin: 0, fontSize: 14, flex: "0 0 auto" }}>비행 경로 편집기 (투영 좌표 평면)</h3>
          <div style={{ flex: 1 }} />
          <button style={modeButtonStyle(mode === "pan")} onClick={() => setMode("pan")}>
            보기/이동
          </button>
          <button style={modeButtonStyle(mode === "cut")} onClick={() => setMode("cut")}>
            제거(Cut) — 드래그 영역
          </button>
          <button style={modeButtonStyle(mode === "restore")} onClick={() => setMode("restore")}>
            복원(Restore) — 드래그 영역
          </button>
          <button style={modeButtonStyle(false)} onClick={() => setView(fitView(validPoints, size.width, size.height))}>
            전체 보기
          </button>
          <button
            style={{ ...modeButtonStyle(false), border: "1px solid #dc2626", color: "#dc2626" }}
            onClick={onClose}
          >
            닫기
          </button>
        </div>

        <div style={{ display: "flex", gap: 14, fontSize: 11, color: "#6b5c42" }}>
          <span>
            <span style={{ display: "inline-block", width: 10, height: 10, background: COLOR_SURVEY, marginRight: 4, verticalAlign: "middle" }} />
            Survey ({nSurvey})
          </span>
          <span>
            <span style={{ display: "inline-block", width: 10, height: 10, background: COLOR_TIE, marginRight: 4, verticalAlign: "middle" }} />
            Tie ({nTie})
          </span>
          <span>
            <span style={{ display: "inline-block", width: 10, height: 10, background: COLOR_EXCLUDED, marginRight: 4, verticalAlign: "middle" }} />
            제외됨 ({nExcluded})
          </span>
          <span style={{ marginLeft: "auto", color: "#ab9a78" }}>
            마우스 휠: 확대/축소 · {mode === "pan" ? "드래그: 화면 이동" : "드래그: 영역 선택 후 자동 적용"}
          </span>
        </div>

        <div ref={containerRef} style={{ flex: 1, minHeight: 0, position: "relative" }}>
          <canvas
            ref={canvasRef}
            style={{ width: "100%", height: "100%", cursor: mode === "pan" ? "grab" : "crosshair", borderRadius: 6 }}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={finishDrag}
            onMouseLeave={() => {
              dragRef.current = null;
              setSelectionRectPx(null);
            }}
          />
          {validPoints.length === 0 && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#ab9a78",
                fontSize: 13,
                background: "#2c2418",
                borderRadius: 6,
                pointerEvents: "none",
              }}
            >
              {loading ? "포인트를 불러오는 중입니다..." : "표시할 포인트가 없습니다 (자료 처리를 먼저 실행하세요)."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
