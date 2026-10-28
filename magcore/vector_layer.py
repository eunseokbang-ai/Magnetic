"""Vector reference layers: drill holes, sample sites, survey outlines
and the like, read from the formats field people actually have - a
Google Earth KML/KMZ, a GeoJSON, or a CSV with latitude/longitude
columns - into one plain structure the map can draw:

    {"type": "vector", "name": ..., "format": "kml" | "geojson" | "csv",
     "features": [{"name": str, "geometry": "Point" | "LineString" | "Polygon",
                   "coordinates": [[lat, lon], ...], "properties": {...}}, ...],
     "bounds": [[south, west], [north, east]],
     "n_points": int, "n_lines": int, "n_polygons": int}

Coordinates come out as [lat, lon] pairs (Leaflet's order). Only WGS84
input is accepted: a KML is WGS84 by definition, a GeoJSON is WGS84 by
RFC 7946, and a CSV must carry lat/lon columns - projected eastings and
northings cannot be told from a frame without a CRS, so they are refused
with a message rather than silently misplaced.

KML details that matter: a Placemark may hold a MultiGeometry (Google
Earth Pro writes several points under one placemark - the Andong drill
holes arrived that way), coordinates are "lon,lat[,alt]" triples
separated by whitespace, and the tags are in the KML namespace, so the
parser matches on the local tag name only. A Polygon's outer ring is kept;
inner rings (holes) are dropped. Nested Folders/Documents are walked.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET

VECTOR_EXTENSIONS = (".kml", ".kmz", ".geojson", ".json", ".csv", ".txt")
MAX_FEATURES = 20000


class VectorLayerError(ValueError):
    pass


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_coords(text: str) -> list[list[float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        lon, lat = float(parts[0]), float(parts[1])
        pts.append([lat, lon])
    return pts


def _kml_geometries(elem) -> list[tuple[str, list[list[float]]]]:
    out = []
    for g in elem.iter():
        t = _local(g.tag)
        if t == "Point":
            c = next((x for x in g.iter() if _local(x.tag) == "coordinates"), None)
            if c is not None and c.text:
                pts = _parse_coords(c.text)
                if pts:
                    out.append(("Point", pts[:1]))
        elif t == "LineString":
            c = next((x for x in g.iter() if _local(x.tag) == "coordinates"), None)
            if c is not None and c.text:
                pts = _parse_coords(c.text)
                if len(pts) >= 2:
                    out.append(("LineString", pts))
        elif t == "Polygon":
            outer = next((x for x in g.iter() if _local(x.tag) == "outerBoundaryIs"), None)
            c = next((x for x in (outer if outer is not None else g).iter() if _local(x.tag) == "coordinates"), None)
            if c is not None and c.text:
                pts = _parse_coords(c.text)
                if len(pts) >= 3:
                    out.append(("Polygon", pts))
    return out


def _from_kml(text: str) -> list[dict]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise VectorLayerError(f"KML을 읽지 못했습니다: {exc}") from exc
    features = []
    for pm in root.iter():
        if _local(pm.tag) != "Placemark":
            continue
        name_el = next((x for x in pm if _local(x.tag) == "name"), None)
        name = (name_el.text or "").strip() if name_el is not None else ""
        desc_el = next((x for x in pm if _local(x.tag) == "description"), None)
        props = {"description": (desc_el.text or "").strip()} if desc_el is not None and desc_el.text else {}
        for ext in pm.iter():
            if _local(ext.tag) == "Data" and ext.get("name"):
                v = next((x for x in ext if _local(x.tag) == "value"), None)
                props[ext.get("name")] = (v.text or "").strip() if v is not None else ""
            elif _local(ext.tag) == "SimpleData" and ext.get("name"):
                props[ext.get("name")] = (ext.text or "").strip()
        geoms = _kml_geometries(pm)
        for i, (gtype, pts) in enumerate(geoms):
            label = name if len(geoms) == 1 or not name else f"{name} #{i + 1}"
            features.append({"name": label, "geometry": gtype, "coordinates": pts, "properties": props})
    return features


def _from_geojson(text: str) -> list[dict]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VectorLayerError(f"GeoJSON을 읽지 못했습니다: {exc}") from exc
    feats = []
    if obj.get("type") == "FeatureCollection":
        items = obj.get("features") or []
    elif obj.get("type") == "Feature":
        items = [obj]
    else:
        items = [{"type": "Feature", "geometry": obj, "properties": {}}]
    for k, f in enumerate(items):
        geom = f.get("geometry") or {}
        props = f.get("properties") or {}
        name = str(props.get("name") or props.get("Name") or props.get("id") or f.get("id") or f"feature {k + 1}")
        gtype, coords = geom.get("type"), geom.get("coordinates")
        parts: list[tuple[str, list]] = []
        if gtype == "Point":
            parts = [("Point", [coords])]
        elif gtype == "MultiPoint":
            parts = [("Point", [c]) for c in coords]
        elif gtype == "LineString":
            parts = [("LineString", coords)]
        elif gtype == "MultiLineString":
            parts = [("LineString", c) for c in coords]
        elif gtype == "Polygon":
            parts = [("Polygon", coords[0])]
        elif gtype == "MultiPolygon":
            parts = [("Polygon", p[0]) for p in coords]
        for i, (t, pts) in enumerate(parts):
            label = name if len(parts) == 1 else f"{name} #{i + 1}"
            feats.append({"name": label, "geometry": t, "coordinates": [[float(c[1]), float(c[0])] for c in pts],
                          "properties": {k2: v for k2, v in props.items() if isinstance(v, (str, int, float, bool)) or v is None}})
    return feats


_LAT_NAMES = ("lat", "latitude", "위도", "y_wgs84")
_LON_NAMES = ("lon", "lng", "long", "longitude", "경도", "x_wgs84")
_NAME_NAMES = ("name", "id", "label", "hole", "station", "이름", "지점", "공번", "시추공")


def _from_csv(text: str) -> list[dict]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise VectorLayerError("CSV에 자료 행이 없습니다.")
    header = [h.strip().lstrip("﻿") for h in rows[0]]
    low = [h.lower() for h in header]
    lat_i = next((i for i, h in enumerate(low) if h in _LAT_NAMES), None)
    lon_i = next((i for i, h in enumerate(low) if h in _LON_NAMES), None)
    if lat_i is None or lon_i is None:
        raise VectorLayerError(f"CSV에 위도/경도 열이 없습니다 (lat/latitude/위도, lon/longitude/경도). 헤더: {header}")
    name_i = next((i for i, h in enumerate(low) if h in _NAME_NAMES), None)
    feats = []
    for k, r in enumerate(rows[1:]):
        if len(r) <= max(lat_i, lon_i):
            continue
        try:
            lat, lon = float(r[lat_i]), float(r[lon_i])
        except ValueError:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            raise VectorLayerError(f"{k + 2}행의 좌표({lat}, {lon})가 위경도 범위를 벗어납니다. 투영 좌표(동거/북거)는 받지 않으니 WGS84 위경도로 바꿔 주세요.")
        props = {header[i]: r[i].strip() for i in range(min(len(header), len(r))) if i not in (lat_i, lon_i) and header[i]}
        name = r[name_i].strip() if name_i is not None and name_i < len(r) and r[name_i].strip() else f"{k + 1}"
        feats.append({"name": name, "geometry": "Point", "coordinates": [[lat, lon]], "properties": props})
    return feats


def detect_vector_format(filename: str, data: bytes) -> str | None:
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if data[:2] == b"PK" or ext == ".kmz":
        return "kmz"
    head = data[:512].lstrip().lower()
    if ext == ".kml" or b"<kml" in head or (head.startswith(b"<?xml") and b"placemark" in data[:4096].lower()):
        return "kml"
    if ext in (".geojson", ".json") or head.startswith(b"{"):
        return "geojson"
    if ext in (".csv", ".txt"):
        return "csv"
    return None


def load_vector_layer(data: bytes, filename: str = "layer") -> dict:
    fmt = detect_vector_format(filename, data)
    if fmt is None:
        raise VectorLayerError(f"지원하지 않는 벡터 형식입니다: {filename} (KML, KMZ, GeoJSON, CSV)")
    if fmt == "kmz":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not kml_names:
                    raise VectorLayerError("KMZ 안에 KML 파일이 없습니다.")
                text = z.read(kml_names[0]).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as exc:
            raise VectorLayerError("KMZ 파일이 손상되었습니다.") from exc
        features, fmt = _from_kml(text), "kml"
    else:
        text = data.decode("utf-8-sig", errors="replace")
        features = _from_kml(text) if fmt == "kml" else _from_geojson(text) if fmt == "geojson" else _from_csv(text)
    if not features:
        raise VectorLayerError("파일에서 점·선·면을 찾지 못했습니다.")
    if len(features) > MAX_FEATURES:
        raise VectorLayerError(f"피처가 너무 많습니다 ({len(features)} > {MAX_FEATURES}).")
    lats = [p[0] for f in features for p in f["coordinates"]]
    lons = [p[1] for f in features for p in f["coordinates"]]
    name = re.sub(r"\.[A-Za-z0-9]+$", "", filename.split("/")[-1].split("\\")[-1]) or "layer"
    return {
        "type": "vector",
        "name": name,
        "format": fmt,
        "features": features,
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
        "n_points": sum(f["geometry"] == "Point" for f in features),
        "n_lines": sum(f["geometry"] == "LineString" for f in features),
        "n_polygons": sum(f["geometry"] == "Polygon" for f in features),
    }
