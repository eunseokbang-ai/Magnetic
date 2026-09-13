"""Auto-detect ground structures (buildings, roads) likely to be causing
localized, non-geological magnetic distortion in a survey - rebar,
buried pipes, guardrails, parked vehicles and the like - so the existing
manual "draw a polygon over the structure in the orthophoto and smooth
it" workflow (see processing/manual_smooth.py, store.py::set_manual_smoothing)
does not have to be done one building at a time by hand.

Two independent, complementary signals feed this:
  - fetch_osm_structures(): pulls building/road footprints for the survey
    area from OpenStreetMap's public Overpass API - a location prior with
    no imagery classification needed, since footprints already exist as
    structured vector data almost everywhere with any development at all.
  - dipole-fit compact-anomaly detection (processing/dipole_fit.py, reused
    as-is from the near-surface target-detection feature) - a signal
    prior that also catches distortion sources OSM has no footprint for
    (a buried fence, an unmapped shed, farm machinery) and lets a
    structure-shaped footprint with no real anomaly under it be skipped
    instead of blindly smoothed.

See store.py::scan_structure_distortion for how the two are combined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# A survey block bigger than this is almost certainly a bounding-box bug
# (e.g. degrees mixed up with meters) rather than a real drone flight
# footprint - failing fast beats sending an enormous, slow query to a
# shared public service.
_MAX_BBOX_AREA_KM2 = 200.0


class OsmFetchError(RuntimeError):
    pass


@dataclass
class OsmStructures:
    # Each entry is a list of [lat, lon] pairs (a closed ring for
    # buildings, an open path for roads) - the same shape
    # ManualSmoothRequest.polygon already expects.
    buildings: list[list[list[float]]]
    roads: list[list[list[float]]]


def fetch_osm_structures(
    south: float, west: float, north: float, east: float, timeout_seconds: float = 30.0
) -> OsmStructures:
    """Query OpenStreetMap's public Overpass API for every building
    footprint and road centerline inside the given WGS84 bounding box.

    This makes a live outbound HTTPS request to a third-party public
    service - some networks (locked-down corporate/sandboxed egress
    policies in particular) block it entirely, in which case this raises
    OsmFetchError. There is no manual-upload fallback for this one (unlike
    the INTERMAGNET base-station substitute) since building/road
    footprints are not something a user would have on hand as a file -
    if this is unreachable, use the manual "지도에서 왜곡 영역 그려 스무딩"
    tool directly instead."""
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise OsmFetchError("잘못된 조사 영역 범위입니다.")
    # Rough equirectangular area estimate - only used as a sanity cap, not
    # for anything geometrically precise.
    area_km2 = (north - south) * 111.0 * (east - west) * 111.0 * np.cos(np.radians((north + south) / 2))
    if area_km2 > _MAX_BBOX_AREA_KM2:
        raise OsmFetchError(
            f"조사 영역이 너무 넓습니다 (약 {area_km2:.0f}km², 최대 {_MAX_BBOX_AREA_KM2:.0f}km²) - "
            "드론 자료가 올바르게 업로드되었는지 확인하세요."
        )

    bbox = f"{south},{west},{north},{east}"
    query = f"""
[out:json][timeout:{int(timeout_seconds) - 5}];
(
  way["building"]({bbox});
  way["highway"]({bbox});
);
out geom;
"""
    try:
        resp = requests.post(_OVERPASS_URL, data={"data": query}, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise OsmFetchError(
            f"OpenStreetMap 건물/도로 자료 요청에 실패했습니다 (네트워크 접근이 막혀 있을 수 있습니다): {exc}"
        ) from exc

    if resp.status_code != 200:
        raise OsmFetchError(
            f"OpenStreetMap Overpass 서버가 오류를 반환했습니다 (HTTP {resp.status_code}). "
            "요청 영역이 너무 넓거나 서버가 일시적으로 과부하 상태일 수 있습니다 - 잠시 후 다시 시도하세요."
        )

    try:
        payload = resp.json()
        elements = payload["elements"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OsmFetchError("OpenStreetMap 응답을 해석하지 못했습니다.") from exc

    buildings: list[list[list[float]]] = []
    roads: list[list[list[float]]] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geometry = el.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        latlon_ring = [[pt["lat"], pt["lon"]] for pt in geometry]
        tags = el.get("tags") or {}
        if "building" in tags:
            buildings.append(latlon_ring)
        elif "highway" in tags:
            roads.append(latlon_ring)

    return OsmStructures(buildings=buildings, roads=roads)


def buffer_structures_to_polygons(
    structures: OsmStructures,
    utm_epsg: int,
    building_buffer_m: float,
    road_buffer_m: float,
) -> list[list[list[float]]]:
    """Buffer every building footprint and road centerline by its own
    radius in local projected meters (an induced-magnetic source's field
    falls off with distance, so how far its distortion actually reaches
    depends on the structure, not on OSM's WGS84 coordinates), union
    everything that overlaps into single polygons, and project back to
    lat/lon - ready to feed straight into ManualSmoothRequest(mode=
    "polygons", polygons=...).

    Buildings need at least 4 points (3 corners + closing point) to form
    a polygon; malformed/degenerate OSM ways are skipped rather than
    raising, since a handful of bad geometries in a large query shouldn't
    fail the whole scan."""
    to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_latlon = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    shapes = []
    for ring in structures.buildings:
        if len(ring) < 4:
            continue
        xs, ys = to_local.transform([p[1] for p in ring], [p[0] for p in ring])
        try:
            poly = Polygon(zip(xs, ys))
            if not poly.is_valid:
                poly = poly.buffer(0)
            shapes.append(poly.buffer(building_buffer_m))
        except Exception:
            continue
    for path in structures.roads:
        if len(path) < 2:
            continue
        xs, ys = to_local.transform([p[1] for p in path], [p[0] for p in path])
        try:
            shapes.append(LineString(zip(xs, ys)).buffer(road_buffer_m))
        except Exception:
            continue

    if not shapes:
        return []

    merged = unary_union(shapes)
    polygons = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

    result = []
    for poly in polygons:
        if poly.is_empty:
            continue
        # Holes (a courtyard fully enclosed by buffered buildings) aren't
        # representable in the plain [[lat,lon],...] ring format the
        # existing manual-smoothing polygon mode expects, and would only
        # matter for exotic donut-shaped building clusters anyway - the
        # exterior ring alone is the right level of detail here.
        xs, ys = poly.exterior.coords.xy
        lons, lats = to_latlon.transform(list(xs), list(ys))
        result.append([[lat, lon] for lat, lon in zip(lats, lons)])
    return result
