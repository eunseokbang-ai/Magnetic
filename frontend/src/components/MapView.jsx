import { useEffect, useMemo, useRef } from "react";
import { MapContainer, TileLayer, LayersControl, ImageOverlay, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet-draw";
import { makeColorScale } from "../colormap";

function PointLayer({ points, valueField, vmin, vmax, cmapKind, onHover }) {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    if (!layerRef.current) {
      layerRef.current = L.layerGroup().addTo(map);
    }
    const layer = layerRef.current;
    layer.clearLayers();
    if (!points || points.length === 0) return;

    const colorScale = makeColorScale(cmapKind, vmin, vmax);
    const renderer = L.canvas({ padding: 0.5 });
    for (const p of points) {
      const excluded = p.excluded;
      const color = excluded ? "#9ca3af" : colorScale(p.value);
      const marker = L.circleMarker([p.lat, p.lon], {
        renderer,
        radius: excluded ? 1.5 : 2.5,
        color,
        fillColor: color,
        fillOpacity: excluded ? 0.35 : 0.9,
        weight: 0,
        opacity: excluded ? 0.35 : 0.9,
      });
      marker.on("mouseover", () => onHover && onHover(p));
      layer.addLayer(marker);
    }
  }, [points, valueField, vmin, vmax, cmapKind, map, onHover]);

  useEffect(() => () => {
    if (layerRef.current) {
      layerRef.current.remove();
      layerRef.current = null;
    }
  }, [map]);

  return null;
}

function FitBounds({ points }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (fitted.current || !points || points.length === 0) return;
    const lats = points.map((p) => p.lat);
    const lons = points.map((p) => p.lon);
    const bounds = [
      [Math.min(...lats), Math.min(...lons)],
      [Math.max(...lats), Math.max(...lons)],
    ];
    map.fitBounds(bounds, { padding: [20, 20] });
    fitted.current = true;
  }, [points, map]);
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
  valueField,
  colorRange,
  cmapKind,
  overlay,
  onHoverPoint,
  drawMode,
  onShapeDrawn,
}) {
  const center = useMemo(() => [46.5, 106.27], []);

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
      {overlay ? (
        <ImageOverlay url={overlay.image_data_url} bounds={overlay.bounds} opacity={0.85} />
      ) : (
        <PointLayer
          points={points}
          valueField={valueField}
          vmin={colorRange.vmin}
          vmax={colorRange.vmax}
          cmapKind={cmapKind}
          onHover={onHoverPoint}
        />
      )}
      <DrawControl enabled={drawMode} onShapeDrawn={onShapeDrawn} />
    </MapContainer>
  );
}
