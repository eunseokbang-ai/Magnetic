"""Which rock is each reading over, and what does the field do there.

The national 1:50,000 geology map is published as vector polygons (KIGAM
GeoServer WFS, no key, EPSG:4326), each carrying the unit symbol (Kj),
the lithology name (화산력 응회암), the age and the sheet it came from. Laid
under the anomaly map that is a picture to look at; joined to the
readings it is a table: for every unit the survey crossed, how many
readings, what the anomaly averages, how much it varies, and how strong
the analytic signal is - which is what an interpretation report
actually states about the geology.

Two things are deliberate:

- The join is done on the *readings*, not on the grid. The grid between
  lines is the interpolator's invention; the readings are what was
  measured over that rock.
- The polygons are fetched for the survey's own bounding box only, with
  the browser-shaped User-Agent the service insists on (see
  tile_cache.py for the measurement), and kept on disk, so the second
  run of a survey and the run in the field both work without asking the
  service again.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape
from shapely.strtree import STRtree

from .. import paths
from .tile_cache import _BROWSER_USER_AGENT

WFS_URL = "https://data.kigam.re.kr/geoserver/wfs"

# Lithology polygon layers, by scale. The 1:50k sheet is the detailed
# mapping but does not cover every island; 1:250k and 1:1M are complete.
LITHOLOGY_LAYERS = {
    "50k": "Geology_map:l_50k_geology_litho_view_latest",
    "250k": "Geology_map:l_250k_geology_litho",
    "1m": "Geology_map:L_1M_geology_litho_2019v_latest",
}
# Attribute names differ a little between scales; these are tried in order.
_SYMBOL_KEYS = ("lithoidx", "litho_idx", "symbol", "sym", "code")
_NAME_KEYS = ("lithoname", "litho_name", "name", "litho")
_AGE_KEYS = ("age", "era", "period")
_SHEET_KEYS = ("mapname", "map_name", "sheet")
_MAX_FEATURES = 5000


class GeologyServiceError(RuntimeError):
    pass


def _cache_path(layer: str, bbox: tuple[float, float, float, float]):
    key = hashlib.sha1(f"{layer}|{','.join(f'{v:.4f}' for v in bbox)}".encode()).hexdigest()[:16]
    directory = paths.data_dir() / "geology_wfs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.geojson"


def fetch_lithology(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    scale: str = "50k", timeout: float = 120.0, use_cache: bool = True,
) -> dict:
    """GeoJSON FeatureCollection of the lithology polygons intersecting
    the box. WFS 2.0 with a plain EPSG:4326 BBOX takes lon,lat order
    here (measured: lat,lon returned nothing, lon,lat returned the
    sheet)."""
    if scale not in LITHOLOGY_LAYERS:
        raise GeologyServiceError(f"지원하지 않는 축척입니다: {scale}")
    layer = LITHOLOGY_LAYERS[scale]
    bbox = (min_lon, min_lat, max_lon, max_lat)
    cache = _cache_path(layer, bbox)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    params = {
        "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
        "TYPENAMES": layer, "OUTPUTFORMAT": "application/json", "SRSNAME": "EPSG:4326",
        "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat},EPSG:4326",
        "COUNT": str(_MAX_FEATURES),
    }
    try:
        resp = requests.get(WFS_URL + "?" + urlencode(params), timeout=timeout,
                            headers={"User-Agent": _BROWSER_USER_AGENT})
    except requests.RequestException as exc:
        raise GeologyServiceError(f"KIGAM 지질도 서비스에 연결할 수 없습니다: {exc}") from exc
    if resp.status_code != 200:
        raise GeologyServiceError(f"KIGAM 지질도 서비스 응답 오류 (HTTP {resp.status_code})")
    try:
        data = resp.json()
    except ValueError as exc:
        raise GeologyServiceError("KIGAM 지질도 서비스가 GeoJSON이 아닌 응답을 보냈습니다.") from exc
    if data.get("type") != "FeatureCollection":
        raise GeologyServiceError("KIGAM 지질도 서비스 응답 형식이 예상과 다릅니다.")
    if use_cache:
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _first(props: dict, keys) -> str | None:
    for k in keys:
        v = props.get(k)
        if v not in (None, ""):
            return str(v)
    return None


@dataclass
class UnitStats:
    symbol: str
    name: str
    age: str | None
    n_points: int
    n_lines: int
    mean_nt: float
    std_nt: float
    median_nt: float
    p10_nt: float
    p90_nt: float
    mean_signal: float | None       # analytic signal, if a grid was given
    area_share_pct: float           # of the survey's readings, by count

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name, "age": self.age,
            "n_points": self.n_points, "n_lines": self.n_lines,
            "mean_nt": round(self.mean_nt, 2), "std_nt": round(self.std_nt, 2),
            "median_nt": round(self.median_nt, 2),
            "p10_nt": round(self.p10_nt, 2), "p90_nt": round(self.p90_nt, 2),
            "mean_signal": None if self.mean_signal is None else round(self.mean_signal, 4),
            "share_pct": round(self.area_share_pct, 1),
        }


def assign_units(lon: np.ndarray, lat: np.ndarray, collection: dict) -> tuple[np.ndarray, list[dict]]:
    """Index of the polygon each point falls in (-1 outside every one),
    and the polygons' attributes in that index order. Point-in-polygon
    through an STRtree, so a million readings against a few hundred
    polygons is one query, not a million."""
    features = collection.get("features") or []
    geoms, attrs = [], []
    for f in features:
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        if g.is_empty:
            continue
        props = f.get("properties") or {}
        geoms.append(g)
        attrs.append({
            "symbol": _first(props, _SYMBOL_KEYS) or "?",
            "name": _first(props, _NAME_KEYS) or "(이름 없음)",
            "age": _first(props, _AGE_KEYS),
            "sheet": _first(props, _SHEET_KEYS),
        })
    unit = np.full(len(lon), -1, dtype=int)
    if not geoms or len(lon) == 0:
        return unit, attrs
    from shapely import points as _points

    tree = STRtree(geoms)
    pts = _points(np.column_stack([lon, lat]))
    hit_pt, hit_geom = tree.query(pts, predicate="within")
    # a point on a shared boundary may be inside two polygons; keep the
    # first, which is as good a rule as any and reproducible
    seen = set()
    for pi, gi in zip(hit_pt, hit_geom):
        if pi not in seen:
            unit[pi] = gi
            seen.add(pi)
    return unit, attrs


def summarize_by_unit(
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray, line_id: np.ndarray,
    collection: dict, signal: np.ndarray | None = None,
) -> tuple[list[UnitStats], dict]:
    """Per-unit statistics of `values` (the anomaly) and optionally of the
    analytic signal sampled at the same points. Units are merged by
    symbol, because one unit is usually several polygons."""
    ok = np.isfinite(values)
    unit, attrs = assign_units(lon[ok], lat[ok], collection)
    values_ok, lines_ok = values[ok], line_id[ok]
    signal_ok = signal[ok] if signal is not None else None

    by_symbol: dict[str, list[int]] = {}
    for gi, a in enumerate(attrs):
        by_symbol.setdefault(a["symbol"], []).append(gi)

    n_total = int(ok.sum())
    out: list[UnitStats] = []
    for symbol, gids in by_symbol.items():
        mask = np.isin(unit, gids)
        n = int(mask.sum())
        if n == 0:
            continue
        v = values_ok[mask]
        sig = signal_ok[mask] if signal_ok is not None else None
        sig = sig[np.isfinite(sig)] if sig is not None else None
        a = attrs[gids[0]]
        out.append(UnitStats(
            symbol=symbol, name=a["name"], age=a["age"],
            n_points=n, n_lines=int(len(np.unique(lines_ok[mask]))),
            mean_nt=float(v.mean()), std_nt=float(v.std()), median_nt=float(np.median(v)),
            p10_nt=float(np.percentile(v, 10)), p90_nt=float(np.percentile(v, 90)),
            mean_signal=float(sig.mean()) if sig is not None and sig.size else None,
            area_share_pct=100.0 * n / max(n_total, 1),
        ))
    out.sort(key=lambda u: -u.n_points)
    n_outside = int((unit < 0).sum())
    report = {
        "n_points": n_total,
        "n_outside": n_outside,
        "n_polygons": len(attrs),
        "n_units": len(out),
        "sheets": sorted({a["sheet"] for a in attrs if a["sheet"]}),
    }
    return out, report


def units_geojson(collection: dict) -> dict:
    """The polygons for the map, trimmed to what the map needs."""
    features = []
    for f in collection.get("features") or []:
        props = f.get("properties") or {}
        features.append({
            "type": "Feature",
            "geometry": f.get("geometry"),
            "properties": {
                "symbol": _first(props, _SYMBOL_KEYS) or "?",
                "name": _first(props, _NAME_KEYS) or "",
                "age": _first(props, _AGE_KEYS),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def stats_to_csv(units: list[UnitStats]) -> bytes:
    frame = pd.DataFrame([u.to_dict() for u in units])
    frame = frame.rename(columns={
        "symbol": "기호", "name": "암상", "age": "시대", "n_points": "측점수", "n_lines": "측선수",
        "mean_nt": "평균_nT", "std_nt": "표준편차_nT", "median_nt": "중앙값_nT",
        "p10_nt": "P10_nT", "p90_nt": "P90_nT", "mean_signal": "해석신호평균", "share_pct": "측점비율_%",
    })
    return frame.to_csv(index=False).encode("utf-8-sig")
