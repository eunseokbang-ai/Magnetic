import { useEffect, useMemo, useRef } from "react";
import { MapContainer, TileLayer, LayersControl, ImageOverlay, ScaleControl, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet-draw";
import { makeColorScale } from "../colormap";
import { minMax } from "../arrayUtils";
import pointCanvasLayer from "../leafletPointCanvasLayer";

// Direct-to-canvas rendering (see leafletPointCanvasLayer.js) instead of one
// L.circleMarker per point - at 100k+ points, per-marker object creation and
// event-listener registration is the actual bottleneck, and it was rebuilt
// from scratch on every color-range/value-field tweak. The layer is created
// once and only re-painted via setData() afterwards.
function PointLayer({ points, vmin, vmax, cmapName, onHover }) {
  const map = useMap();
  const layerRef = useRef(null);
  const onHoverRef = useRef(onHover);
  onHoverRef.current = onHover;

  useEffect(() => {
    const layer = pointCanvasLayer([], { onHover: (p) => onHoverRef.current && onHoverRef.current(p) });
    layer.addTo(map);
    layerRef.current = layer;
    return () => {
      layer.remove();
      layerRef.current = null;
    };
  }, [map]);

  useEffect(() => {
    if (!layerRef.current) return;
    const colorScale = makeColorScale(cmapName, vmin, vmax);
    layerRef.current.setData(points || [], colorScale);
  }, [points, vmin, vmax, cmapName]);

  return null;
}

function FitBounds({ points }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (fitted.current || !points || points.length === 0) return;
    const lats = points.map((p) => p.lat);
    const lons = points.map((p) => p.lon);
    const [latMin, latMax] = minMax(lats);
    const [lonMin, lonMax] = minMax(lons);
    const bounds = [
      [latMin, lonMin],
      [latMax, lonMax],
    ];
    map.fitBounds(bounds, { padding: [20, 20] });
    fitted.current = true;
  }, [points, map]);
  return null;
}

// Line count is normally a few dozen at most, so plain Leaflet markers (with
// no per-point performance concerns) are fine here - unlike PointLayer this
// doesn't need the canvas approach.
function LineLabels({ lines, visible }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!visible || !lines) return;
    for (const l of lines) {
      if (l.centroid_lat == null || l.centroid_lon == null) continue;
      const icon = L.divIcon({
        className: "line-number-label",
        html: `<div style="background:rgba(37,99,235,0.9);color:white;font-size:11px;font-weight:600;padding:1px 5px;border-radius:4px;white-space:nowrap;transform:translate(-50%,-50%);">#${l.line_id}</div>`,
        iconSize: [0, 0],
      });
      L.marker([l.centroid_lat, l.centroid_lon], { icon, interactive: false }).addTo(group);
    }
  }, [lines, visible]);

  return null;
}

function DrawControl({ enabled, shapeType = "polygon", onShapeDrawn, repeatMode = false }) {
  const map = useMap();
  const controlRef = useRef(null);
  const groupRef = useRef(null);

  useEffect(() => {
    if (!groupRef.current) {
      groupRef.current = new L.FeatureGroup();
      map.addLayer(groupRef.current);
    }
    const handleCreated = (e) => {
      const layer = e.layer;
      // Polygon/rectangle getLatLngs() nests one ring array deep; polyline
      // returns a flat array of LatLngs directly - Array.isArray on the
      // first element (an array vs. a LatLng object) tells them apart.
      const rawLatLngs = layer.getLatLngs();
      const latlngs = Array.isArray(rawLatLngs[0]) ? rawLatLngs[0] : rawLatLngs;
      const coords = latlngs.map((ll) => [ll.lat, ll.lng]);
      onShapeDrawn && onShapeDrawn(coords);
      groupRef.current.clearLayers();
    };
    map.on(L.Draw.Event.CREATED, handleCreated);
    return () => map.off(L.Draw.Event.CREATED, handleCreated);
  }, [map, onShapeDrawn]);

  useEffect(() => {
    if (controlRef.current) {
      map.removeControl(controlRef.current);
      controlRef.current = null;
    }
    if (!enabled) return undefined;

    const isPolyline = shapeType === "polyline";
    controlRef.current = new L.Control.Draw({
      draw: {
        polygon: isPolyline ? false : { allowIntersection: false, showArea: false, repeatMode },
        // leaflet-draw's readableArea() throws ("type is not defined") on
        // Leaflet 1.9.x when the rectangle tooltip tries to show area, so
        // this stays off - see https://github.com/Leaflet/Leaflet.draw/issues/1026
        rectangle: isPolyline ? false : { showArea: false },
        polyline: isPolyline ? { showLength: false, repeatMode } : false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: false,
    });
    map.addControl(controlRef.current);

    return () => {
      if (controlRef.current) {
        map.removeControl(controlRef.current);
        controlRef.current = null;
      }
    };
  }, [enabled, shapeType, repeatMode, map]);

  return null;
}

// Persistent ruler results: each finished distance line or area polygon
// (see App.jsx's measurements state, built in geoMeasure.js) stays drawn on
// the map with its computed length/area labeled at its midpoint/centroid,
// so several anomaly-zone measurements can sit on screen together for
// comparison rather than disappearing after each draw.
function MeasureLayer({ measurements }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    for (const m of measurements || []) {
      let anchor;
      if (m.type === "distance") {
        const line = L.polyline(m.latlngs, { color: "#dc2626", weight: 3 });
        group.addLayer(line);
        anchor = m.latlngs[Math.floor(m.latlngs.length / 2)];
      } else {
        const poly = L.polygon(m.latlngs, { color: "#7c3aed", weight: 2, fillColor: "#7c3aed", fillOpacity: 0.12 });
        group.addLayer(poly);
        anchor = poly.getBounds().getCenter();
      }
      const color = m.type === "distance" ? "#dc2626" : "#7c3aed";
      const icon = L.divIcon({
        className: "measure-value-label",
        html:
          `<div style="background:${color};color:white;font-size:11px;font-weight:700;padding:2px 6px;` +
          `border-radius:4px;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.4);transform:translate(-50%,-50%);">` +
          `${m.label}: ${m.valueText}</div>`,
        iconSize: [0, 0],
      });
      group.addLayer(L.marker(anchor, { icon, interactive: false }));
    }
  }, [measurements]);

  return null;
}

// Contour paths come pre-computed from the backend (see processing/contours.py)
// as plain lat/lon polylines; drawn directly with the Leaflet canvas renderer
// rather than one more matplotlib raster so they stay crisp at any zoom and
// can be toggled without waiting for a full grid re-render.
function ContourLayer({ contours }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    const features = contours?.features || [];
    if (features.length === 0) return;
    const levels = contours.levels || [];
    const majorEvery = Math.max(1, Math.floor(levels.length / 5) || 1);
    const levelIndex = new Map(levels.map((lv, i) => [lv, i]));
    for (const feat of features) {
      const idx = levelIndex.get(feat.level) ?? 0;
      const isMajor = idx % majorEvery === 0;
      const line = L.polyline(feat.path, {
        color: isMajor ? "#374151" : "#9ca3af",
        weight: isMajor ? 1.6 : 0.8,
        opacity: 0.85,
      });
      line.bindTooltip(`${feat.level.toFixed(1)} nT`, { sticky: true });
      group.addLayer(line);
    }
  }, [contours]);

  return null;
}

// Leaflet maps here never rotate (no bearing/heading control), so a north
// arrow is purely a static "up = north" indicator, not a computed rotation.
function NorthArrow() {
  const map = useMap();
  useEffect(() => {
    const control = L.control({ position: "bottomleft" });
    control.onAdd = () => {
      const div = L.DomUtil.create("div");
      div.style.background = "rgba(255,255,255,0.9)";
      div.style.padding = "3px 8px";
      div.style.borderRadius = "4px";
      div.style.boxShadow = "0 1px 4px rgba(0,0,0,0.3)";
      div.style.textAlign = "center";
      div.style.color = "#111827";
      div.style.userSelect = "none";
      div.innerHTML = '<div style="font-size:15px;line-height:1;">▲</div><div style="font-size:10px;font-weight:600;">N</div>';
      return div;
    };
    control.addTo(map);
    return () => control.remove();
  }, [map]);
  return null;
}

// Euler deconvolution solutions are sparse (tens to a few hundred points),
// so plain circleMarkers (not the canvas point layer) are fine here.
function EulerLayer({ solutions }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    const points = solutions || [];
    if (points.length === 0) return;
    const depths = points.map((s) => s.depth_m);
    const dmin = Math.min(...depths);
    const dmax = Math.max(...depths);
    const colorScale = makeColorScale("viridis", dmin, dmax);
    for (const s of points) {
      const marker = L.circleMarker([s.lat, s.lon], {
        radius: 5,
        color: "#111827",
        weight: 1,
        fillColor: colorScale(s.depth_m),
        fillOpacity: 0.85,
      });
      marker.bindTooltip(`깊이: ${s.depth_m.toFixed(1)} m<br/>불확실도: ±${s.uncertainty_m.toFixed(1)} m`, { sticky: true });
      group.addLayer(marker);
    }
  }, [solutions]);

  return null;
}

// Multi-scale edge detection ("worming") results: THDR ridge points at
// several upward-continuation heights, colored by height so a ridge that
// persists across many heights (a steep, laterally-continuous contact)
// visually stands apart from one that only shows up near the surface
// (a shallow, localised source) - see processing/multiscale_edges.py.
function MultiscaleEdgeLayer({ points }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    const list = points || [];
    if (list.length === 0) return;
    const heights = list.map((p) => p.height_m);
    const hmin = Math.min(...heights);
    const hmax = Math.max(...heights);
    const colorScale = makeColorScale("plasma", hmin, hmax);
    for (const p of list) {
      const marker = L.circleMarker([p.lat, p.lon], {
        radius: 3,
        color: "#111827",
        weight: 0.5,
        fillColor: colorScale(p.height_m),
        fillOpacity: 0.85,
      });
      marker.bindTooltip(`상향연속 고도: ${p.height_m.toFixed(0)} m<br/>THDR: ${p.thdr_value.toFixed(2)}`, { sticky: true });
      group.addLayer(marker);
    }
  }, [points]);

  return null;
}

function TargetDetectionLayer({ targets }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    const list = targets || [];
    if (list.length === 0) return;
    list.forEach((t, i) => {
      // radius scales with log(moment) so a huge vehicle-scale target
      // doesn't visually swallow small candidates on the same map
      const radius = 6 + 3 * Math.log10(Math.max(t.moment_am2, 0.01) + 1);
      const marker = L.circleMarker([t.lat, t.lon], {
        radius,
        color: "#111827",
        weight: 2,
        dashArray: "3,2",
        fillColor: "#dc2626",
        fillOpacity: 0.55 * t.fit_quality + 0.15,
      });
      marker.bindTooltip(
        `표적 후보 #${i + 1}<br/>심도: ${t.depth_m.toFixed(2)} m<br/>쌍극자모멘트: ${t.moment_am2.toFixed(2)} A·m²` +
          `<br/>크기등급: ${t.size_class}<br/>첨두이상: ${t.peak_anomaly_nt.toFixed(1)} nT<br/>적합도: ${t.fit_quality.toFixed(2)}`,
        { sticky: true }
      );
      group.addLayer(marker);
    });
  }, [targets]);

  return null;
}

// Structure-distortion auto-scan results (store.py::scan_structure_distortion):
// buffered OpenStreetMap building/road regions as filled polygons, plus the
// signal-only compact-anomaly candidates as points colored by whether they
// landed inside one of those regions - a solid highlighted outline marks
// whichever ones are currently checked in StructureDistortionPanel, so the
// selection being about to be smoothed is visible on the map before applying.
function StructureCandidateLayer({ result, selectedPolygonIndices, selectedAnomalyIndices }) {
  const map = useMap();
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!result) return;

    (result.structure_polygons || []).forEach((ring, i) => {
      const selected = selectedPolygonIndices?.has(i);
      const poly = L.polygon(ring, {
        color: selected ? "#0f766e" : "#5eead4",
        weight: selected ? 3 : 1.5,
        fillColor: "#0f766e",
        fillOpacity: selected ? 0.25 : 0.1,
        dashArray: selected ? null : "4,3",
      });
      poly.bindTooltip(`구조물 영역 #${i + 1}`, { sticky: true });
      group.addLayer(poly);
    });

    (result.anomalies || []).forEach((a, i) => {
      const selected = selectedAnomalyIndices?.has(i);
      const color = a.matched_structure ? "#0f766e" : "#b45309";
      const marker = L.circleMarker([a.lat, a.lon], {
        radius: selected ? 8 : 6,
        color: selected ? "#111827" : color,
        weight: selected ? 2 : 1,
        fillColor: color,
        fillOpacity: 0.7,
      });
      marker.bindTooltip(
        `${a.matched_structure ? "구조물 매칭" : "미매칭 (직접 확인 필요)"}<br/>` +
          `첨두이상: ${a.peak_anomaly_nt.toFixed(1)} nT<br/>크기: ${a.footprint_m.toFixed(1)} m<br/>적합도: ${a.fit_quality.toFixed(2)}`,
        { sticky: true }
      );
      group.addLayer(marker);
    });
  }, [result, selectedPolygonIndices, selectedAnomalyIndices]);

  return null;
}

// What Project.sample_overlay_value's "label" field (the base value field
// or the active transform key) actually reads out as: a short Korean name
// to disambiguate what's pinned, and the physical unit of that quantity -
// most transforms are still nT, but the derivative-based ones are not
// (nT/m, nT/m²) and tilt/theta are angles (°), so a bare "nT" suffix on
// every pin would misrepresent them. Mirrors the transform button labels
// in WorkflowSteps.jsx.
const OVERLAY_VALUE_INFO = {
  anomaly: { name: "자력 이상", unit: "nT" },
  tmi: { name: "TMI", unit: "nT" },
  rtp: { name: "RTP", unit: "nT" },
  rte: { name: "RTE", unit: "nT" },
  "1vd": { name: "1VD", unit: "nT/m" },
  "2vd": { name: "2VD", unit: "nT/m²" },
  as: { name: "AS", unit: "nT/m" },
  thdr: { name: "THDR", unit: "nT/m" },
  tilt: { name: "틸트각", unit: "°" },
  theta: { name: "세타맵", unit: "°" },
  dx: { name: "dX", unit: "nT/m" },
  dy: { name: "dY", unit: "nT/m" },
  dxx: { name: "dXX", unit: "nT/m²" },
  dyy: { name: "dYY", unit: "nT/m²" },
  dxy: { name: "dXY", unit: "nT/m²" },
  dxz: { name: "dXZ", unit: "nT/m²" },
  dyz: { name: "dYZ", unit: "nT/m²" },
  upward_continuation: { name: "상방연속", unit: "nT" },
  detrend: { name: "추세면제거", unit: "nT" },
  microlevel: { name: "마이크로레벨링", unit: "nT" },
};

// Click-to-inspect: while active, every map click samples the currently
// displayed grid/derivative overlay at that exact point (bilinear
// interpolation - see Project.sample_overlay_value) and pins the value
// right there, so several points can be read and compared at once without
// leaving the map. Points accumulate until the toggle button is switched
// off, which clears them (see App.jsx's inspectPoints state).
function InspectLayer({ active, points, onPointClick }) {
  const map = useMapEvents({
    click: (e) => {
      if (!active) return;
      onPointClick && onPointClick(e.latlng.lat, e.latlng.lng);
    },
  });
  const groupRef = useRef(null);

  useEffect(() => {
    const group = L.layerGroup().addTo(map);
    groupRef.current = group;
    return () => group.remove();
  }, [map]);

  useEffect(() => {
    const container = map.getContainer();
    container.style.cursor = active ? "crosshair" : "";
    return () => {
      container.style.cursor = "";
    };
  }, [active, map]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    group.clearLayers();
    for (const p of points || []) {
      const hasValue = p.in_bounds && p.value_nt != null;
      const info = OVERLAY_VALUE_INFO[p.label] || { name: p.label || "", unit: "nT" };
      const text = hasValue ? `${info.name}: ${p.value_nt.toFixed(2)} ${info.unit}` : `${info.name}: 자료 없음`;
      const icon = L.divIcon({
        className: "inspect-value-label",
        html:
          `<div style="display:flex;flex-direction:column;align-items:center;transform:translate(-50%,-100%);">` +
          `<div style="background:${hasValue ? "#111827" : "#6b7280"};color:white;font-size:11px;font-weight:700;` +
          `padding:2px 6px;border-radius:4px;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.4);">${text}</div>` +
          `<div style="width:0;height:0;border-left:4px solid transparent;border-right:4px solid transparent;` +
          `border-top:5px solid ${hasValue ? "#111827" : "#6b7280"};"></div>` +
          `</div>`,
        iconSize: [0, 0],
      });
      group.addLayer(L.marker([p.lat, p.lon], { icon, interactive: false }));
      group.addLayer(
        L.circleMarker([p.lat, p.lon], { radius: 3, color: "#111827", weight: 1, fillColor: "#f59e0b", fillOpacity: 1 })
      );
    }
  }, [points]);

  return null;
}

// Outlines the optional user-drawn display-boundary polygon (see App.jsx's
// boundaryPolygon state / api.js::setDisplayBoundary) so the user can see
// exactly what area grid interpolation/extrapolation is currently clipped
// to, on top of the automatic convex-hull cap the backend always applies.
function BoundaryLayer({ polygon }) {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.remove();
      layerRef.current = null;
    }
    if (!polygon || polygon.length < 3) return undefined;
    const layer = L.polygon(polygon, {
      color: "#ea580c",
      weight: 2,
      dashArray: "6 4",
      fill: false,
      interactive: false,
    }).addTo(map);
    layerRef.current = layer;
    return () => {
      layer.remove();
      layerRef.current = null;
    };
  }, [polygon, map]);

  return null;
}

// Reports the current visible map bounds up to the parent (e.g. so the
// offline tile download panel can offer "use the area I'm currently
// looking at" instead of making the user type lat/lon by hand).
function BoundsWatcher({ onBoundsChange }) {
  const map = useMapEvents({
    moveend: () => onBoundsChange && onBoundsChange(map.getBounds()),
  });
  useEffect(() => {
    if (onBoundsChange) onBoundsChange(map.getBounds());
  }, [map, onBoundsChange]);
  return null;
}

export default function MapView({
  points,
  colorRange,
  cmapName,
  overlay,
  gridOpacity,
  showPointsOverGrid,
  overlayLayers,
  lines,
  showLineLabels,
  onHoverPoint,
  drawMode,
  drawShapeType,
  onShapeDrawn,
  eulerSolutions,
  detectedTargets,
  multiscaleEdgePoints,
  onBoundsChange,
  inspectMode,
  inspectPoints,
  onInspectClick,
  measureMode,
  measurements,
  onMeasureShapeDrawn,
  structureScanResult,
  selectedStructurePolygonIndices,
  selectedStructureAnomalyIndices,
  boundaryMode,
  boundaryPolygon,
  onBoundaryDrawn,
}) {
  const center = useMemo(() => [46.5, 106.27], []);
  const pointsVisible = !overlay || showPointsOverGrid;

  return (
    <MapContainer center={center} zoom={13} style={{ height: "100%", width: "100%" }} preferCanvas>
      {onBoundsChange && <BoundsWatcher onBoundsChange={onBoundsChange} />}
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="OpenStreetMap">
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Esri World Imagery (위성)">
          <TileLayer
            attribution="Tiles &copy; Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Google 위성 (비공식 타일, ToS 주의)">
          <TileLayer attribution="&copy; Google" url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}" />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="OpenStreetMap (오프라인 캐시)">
          <TileLayer
            attribution="&copy; OpenStreetMap contributors (오프라인 캐시)"
            url="/api/tiles/osm/{z}/{x}/{y}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Esri 위성 (오프라인 캐시)">
          <TileLayer attribution="Tiles &copy; Esri (오프라인 캐시)" url="/api/tiles/esri/{z}/{x}/{y}" />
        </LayersControl.BaseLayer>
      </LayersControl>

      <FitBounds points={points} />

      {/* user-uploaded reference layers (e.g. GeoTIFF geology maps, or a
          pre-tiled local folder for large orthophotos), bottom to top */}
      {(overlayLayers || [])
        .filter((l) => l.visible)
        .map((l) =>
          l.type === "tiles" ? (
            <TileLayer
              key={l.id}
              url={`/api/local-tiles/${l.tile_layer_id}/{z}/{x}/{y}`}
              bounds={l.bounds}
              minZoom={l.min_zoom}
              maxNativeZoom={l.max_zoom}
              opacity={l.opacity}
              pane="overlayPane"
            />
          ) : (
            <ImageOverlay key={l.id} url={l.image_data_url} bounds={l.bounds} opacity={l.opacity} />
          )
        )}

      {overlay && <ImageOverlay url={overlay.image_data_url} bounds={overlay.bounds} opacity={gridOpacity} />}
      {overlay?.contours && <ContourLayer contours={overlay.contours} />}

      <PointLayer
        points={pointsVisible ? points : []}
        vmin={colorRange.vmin}
        vmax={colorRange.vmax}
        cmapName={cmapName}
        onHover={onHoverPoint}
      />

      <LineLabels lines={lines} visible={showLineLabels} />

      {eulerSolutions && <EulerLayer solutions={eulerSolutions} />}
      {detectedTargets && <TargetDetectionLayer targets={detectedTargets} />}
      {multiscaleEdgePoints && <MultiscaleEdgeLayer points={multiscaleEdgePoints} />}
      {structureScanResult && (
        <StructureCandidateLayer
          result={structureScanResult}
          selectedPolygonIndices={selectedStructurePolygonIndices}
          selectedAnomalyIndices={selectedStructureAnomalyIndices}
        />
      )}

      <DrawControl enabled={drawMode} shapeType={drawShapeType} onShapeDrawn={onShapeDrawn} />
      <DrawControl
        enabled={!!measureMode}
        shapeType={measureMode === "distance" ? "polyline" : "polygon"}
        onShapeDrawn={onMeasureShapeDrawn}
        repeatMode
      />
      <DrawControl enabled={!!boundaryMode} shapeType="polygon" onShapeDrawn={onBoundaryDrawn} />
      <BoundaryLayer polygon={boundaryPolygon} />
      <InspectLayer active={inspectMode} points={inspectPoints} onPointClick={onInspectClick} />
      <MeasureLayer measurements={measurements} />
      <ScaleControl position="bottomright" imperial={false} />
      <NorthArrow />
    </MapContainer>
  );
}
