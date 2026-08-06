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

function DrawControl({ enabled, shapeType = "polygon", onShapeDrawn }) {
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
        polygon: isPolyline ? false : { allowIntersection: false, showArea: false },
        // leaflet-draw's readableArea() throws ("type is not defined") on
        // Leaflet 1.9.x when the rectangle tooltip tries to show area, so
        // this stays off - see https://github.com/Leaflet/Leaflet.draw/issues/1026
        rectangle: isPolyline ? false : { showArea: false },
        polyline: isPolyline ? { showLength: false } : false,
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
  }, [enabled, shapeType, map]);

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

      <DrawControl enabled={drawMode} shapeType={drawShapeType} onShapeDrawn={onShapeDrawn} />
      <ScaleControl position="bottomright" imperial={false} />
      <NorthArrow />
    </MapContainer>
  );
}
