import { useEffect, useMemo, useRef } from "react";
import { MapContainer, TileLayer, LayersControl, ImageOverlay, useMap } from "react-leaflet";
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

function DrawControl({ enabled, onShapeDrawn }) {
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
      const latlngs = layer.getLatLngs()[0] || layer.getLatLngs();
      const coords = latlngs.map((ll) => [ll.lat, ll.lng]);
      onShapeDrawn && onShapeDrawn(coords);
      groupRef.current.clearLayers();
    };
    map.on(L.Draw.Event.CREATED, handleCreated);
    return () => map.off(L.Draw.Event.CREATED, handleCreated);
  }, [map, onShapeDrawn]);

  useEffect(() => {
    if (enabled && !controlRef.current) {
      controlRef.current = new L.Control.Draw({
        draw: {
          polygon: { allowIntersection: false, showArea: false },
          // leaflet-draw's readableArea() throws ("type is not defined") on
          // Leaflet 1.9.x when the rectangle tooltip tries to show area, so
          // this stays off - see https://github.com/Leaflet/Leaflet.draw/issues/1026
          rectangle: { showArea: false },
          circle: false,
          circlemarker: false,
          marker: false,
          polyline: false,
        },
        edit: false,
      });
      map.addControl(controlRef.current);
    } else if (!enabled && controlRef.current) {
      map.removeControl(controlRef.current);
      controlRef.current = null;
    }
    return () => {
      if (controlRef.current) {
        map.removeControl(controlRef.current);
        controlRef.current = null;
      }
    };
  }, [enabled, map]);

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
  onShapeDrawn,
}) {
  const center = useMemo(() => [46.5, 106.27], []);
  const pointsVisible = !overlay || showPointsOverGrid;

  return (
    <MapContainer center={center} zoom={13} style={{ height: "100%", width: "100%" }} preferCanvas>
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
      </LayersControl>

      <FitBounds points={points} />

      {/* user-uploaded reference layers (e.g. GeoTIFF geology maps), bottom to top */}
      {(overlayLayers || [])
        .filter((l) => l.visible)
        .map((l) => (
          <ImageOverlay key={l.id} url={l.image_data_url} bounds={l.bounds} opacity={l.opacity} />
        ))}

      {overlay && <ImageOverlay url={overlay.image_data_url} bounds={overlay.bounds} opacity={gridOpacity} />}

      <PointLayer
        points={pointsVisible ? points : []}
        vmin={colorRange.vmin}
        vmax={colorRange.vmax}
        cmapName={cmapName}
        onHover={onHoverPoint}
      />

      <LineLabels lines={lines} visible={showLineLabels} />

      <DrawControl enabled={drawMode} onShapeDrawn={onShapeDrawn} />
    </MapContainer>
  );
}
