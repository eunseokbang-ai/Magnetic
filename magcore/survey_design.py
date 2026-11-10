"""Survey design: how far apart to fly the lines for the map that is
wanted, whether the target will show at that height, and a flight plan
with tie lines and a repeated line to go with it.

Two limits of a finished survey cannot be processed away, only designed
away: lines too far apart to resolve the field (line_resolution.py
measures it afterwards - 42 % of the signal unresolved on the Andong
block at 100 m), and drift during a flight that nothing in the data can
separate from a regional gradient (no tie lines, no repeated ground).
This module is the before-the-flight side of both.

**Line spacing.** The field of sources a distance h below the sensor is
the source distribution upward-continued by h, so its power spectrum
falls as exp(-2kh) (k in rad/m). Lines dx apart sample the across-line
direction with Nyquist wavenumber pi/dx; everything above it folds back
as false long-wavelength signal. With a product that weights the
spectrum by k^(2n) - n = 0 the field itself, 1 a first derivative
(gradients, analytic signal, tilt), 2 a second derivative - the aliased
share of the power is

    F_n = integral_{pi/dx}^inf k^(2n) exp(-2kh) dk / integral_0^inf (same)
        = Q(2n + 1, 2*pi*h/dx)            (regularised upper incomplete gamma)

Setting F_n to a tolerance and solving for dx gives the widest
acceptable spacing. At the usual 5 %: dx = 2.10 h for the field, 1.00 h
for first derivatives, 0.69 h for second derivatives - the 2h / h / h/2
rules of Reid (1980, Geophysics 45, "Aeromagnetic survey design"), which
is this same argument. h is sensor to source: flying height above ground
plus depth of the source below ground. The spectrum assumed is that of
an uncorrelated layer of sources at one depth, the standard
(Spector-Grant) model; shallower sources than the design depth alias
more than the figure says, which is why the design depth should be the
shallowest source of interest.

**Detectability.** A compact body far enough away is a dipole. For a
sphere of radius R and susceptibility contrast chi in an inducing field
F, the moment is (4/3) pi R^3 chi F / mu0 and the peak anomaly over it

    dB = (mu0 / 4 pi) * 2m / h^3 = (2/3) * chi * F * (R / h)^3

(exact on the axis at the magnetic pole; the peak-to-peak value at
mid-latitudes is of the same size). Against a noise level sigma this
gives a signal-to-noise ratio, and the greatest sensor-to-source distance
at which the body still reaches a wanted ratio. It is an order-of-
magnitude screen, not a forward model: remanence, shape and
demagnetisation move the real figure by a factor of a few either way.

**Flight plan.** Parallel lines across the survey polygon at the chosen
spacing and azimuth, each extended past the boundary by a lead-in so the
turns (which the processing discards) happen outside the area; tie lines
at right angles; and the first line flown again as the last leg, which is
what repeat_passes.py and the repeatability QC need. Lines are ordered
back and forth and split into sorties by endurance. The plan is
geometry at constant map position: terrain following, obstacle
clearance and airspace are the pilot's flight-planning software's job,
and the KML is meant to be imported there.
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

import numpy as np
from pyproj import Transformer
from scipy.special import gammaincc, gammainccinv
from shapely.geometry import LineString, MultiLineString, Polygon

PRODUCTS = {
    # derivative order n of the product, and what it is called on screen
    "field": (0, "자력 이상도 (TMI·RTP)"),
    "first_derivative": (1, "1차 미분 산출물 (구배·해석신호·틸트)"),
    "second_derivative": (2, "2차 미분 산출물 (2차 수직미분·오일러)"),
}
DEFAULT_ALIAS_TOLERANCE = 0.05
# Tie lines every this many survey lines: the usual 10:1 of airborne
# practice. Fewer than MIN_TIE_LINES crossings per line cannot separate a
# drift from a tilt, so small blocks get at least that many.
TIE_LINE_RATIO = 10.0
MIN_TIE_LINES = 3
MU0_OVER_4PI_NT = 100.0          # (mu0 / 4 pi) in nT m / A


class SurveyDesignError(ValueError):
    pass


def aliased_fraction(spacing_m: float, source_distance_m: float, derivative_order: int = 0) -> float:
    """Share of the product's power that lines `spacing_m` apart fold back
    as false signal - F_n in the module docstring."""
    if spacing_m <= 0 or source_distance_m <= 0:
        raise SurveyDesignError("측선 간격과 소스까지의 거리는 0보다 커야 합니다.")
    return float(gammaincc(2 * derivative_order + 1, 2.0 * math.pi * source_distance_m / spacing_m))


def max_line_spacing(source_distance_m: float, derivative_order: int = 0, tolerance: float = DEFAULT_ALIAS_TOLERANCE) -> float:
    """The widest spacing that keeps the aliased share within `tolerance`."""
    if not 0.0 < tolerance < 1.0:
        raise SurveyDesignError("허용 에일리어싱 비율은 0과 1 사이여야 합니다.")
    return float(2.0 * math.pi * source_distance_m / gammainccinv(2 * derivative_order + 1, tolerance))


def sphere_anomaly_nt(radius_m: float, susceptibility_si: float, field_nt: float, distance_m: float) -> float:
    """Peak anomaly of a uniformly magnetised sphere at `distance_m` from
    its centre - (2/3) chi F (R/h)^3, see the module docstring."""
    return (2.0 / 3.0) * susceptibility_si * field_nt * (radius_m / distance_m) ** 3


def recommend(
    target_depth_m: float,
    flight_height_m: float,
    product: str = "first_derivative",
    alias_tolerance: float = DEFAULT_ALIAS_TOLERANCE,
    speed_mps: float = 5.0,
    sample_rate_hz: float = 10.0,
    noise_nt: float = 0.5,
    target_radius_m: float | None = None,
    susceptibility_si: float | None = None,
    field_nt: float = 50000.0,
    min_snr: float = 3.0,
) -> dict:
    """Line spacing, tie spacing and sampling for a target `target_depth_m`
    below ground flown `flight_height_m` above it; with a target size and
    susceptibility contrast, whether it will show."""
    if product not in PRODUCTS:
        raise SurveyDesignError(f"알 수 없는 산출물입니다: {product}")
    if target_depth_m < 0 or flight_height_m <= 0:
        raise SurveyDesignError("목표 심도는 0 이상, 비행 고도는 0보다 커야 합니다.")
    h = float(target_depth_m + flight_height_m)
    order, label = PRODUCTS[product]
    spacing = max_line_spacing(h, order, alias_tolerance)
    out = {
        "source_distance_m": round(h, 2),
        "product": product,
        "product_label": label,
        "alias_tolerance": alias_tolerance,
        "line_spacing_m": round(spacing, 1),
        "spacing_over_distance": round(spacing / h, 2),
        # the same height with each product, so the cost of asking for more is visible
        "spacing_by_product": {key: round(max_line_spacing(h, n, alias_tolerance), 1) for key, (n, _) in PRODUCTS.items()},
        "tie_line_spacing_m": round(spacing * TIE_LINE_RATIO, 0),
        # anomaly of a compact source is about as wide (FWHM) as it is far away
        "anomaly_width_m": round(h, 1),
        "along_line_sample_m": round(speed_mps / sample_rate_hz, 2),
        "samples_per_anomaly_width": round(h / (speed_mps / sample_rate_hz), 1),
        "warnings": [],
    }
    if out["samples_per_anomaly_width"] < 5:
        out["warnings"].append(
            f"측선 방향 샘플 간격({out['along_line_sample_m']} m)이 이상 폭({out['anomaly_width_m']} m)에 비해 성깁니다 - 속도를 낮추거나 샘플링을 올리세요."
        )
    if target_radius_m and susceptibility_si:
        amplitude = sphere_anomaly_nt(target_radius_m, susceptibility_si, field_nt, h)
        reach = target_radius_m * ((2.0 / 3.0) * susceptibility_si * field_nt / (min_snr * noise_nt)) ** (1.0 / 3.0)
        out["detectability"] = {
            "peak_anomaly_nt": round(amplitude, 2),
            "noise_nt": noise_nt,
            "snr": round(amplitude / noise_nt, 1),
            "min_snr": min_snr,
            "detectable": bool(amplitude >= min_snr * noise_nt),
            # sensor-to-source distance at which the anomaly falls to min_snr x noise
            "max_source_distance_m": round(reach, 1),
            "max_flight_height_m": round(reach - target_depth_m, 1),
        }
        if not out["detectability"]["detectable"]:
            out["warnings"].append(
                f"이 고도에서 예상 이상({amplitude:.2f} nT)이 잡음의 {min_snr:g}배({min_snr * noise_nt:.2f} nT)에 못 미칩니다 - "
                + (f"비행 고도를 {reach - target_depth_m:.0f} m 이하로 낮춰야 합니다." if reach > target_depth_m else "표적이 너무 깊거나 작아 고도를 낮춰도 어렵습니다.")
            )
    return out


# ------------------------------------------------------------------ flight plan
def _utm_transformers(lat: float, lon: float):
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return (Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True),
            Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True), epsg)


def _parallel_lines(poly: Polygon, azimuth_deg: float, spacing: float, lead_in: float) -> list[list[tuple[float, float]]]:
    """Segments of the lines at `azimuth_deg` (clockwise from north) that
    cross `poly`, `spacing` apart, each extended by `lead_in` at both
    ends. One entry per crossing (a concave polygon cuts a line in two)."""
    a = math.radians(azimuth_deg)
    along = np.array([math.sin(a), math.cos(a)])         # unit vector of the flight direction (east, north)
    across = np.array([math.cos(a), -math.sin(a)])
    cx, cy = poly.centroid.x, poly.centroid.y
    pts = np.array(poly.exterior.coords) - [cx, cy]
    t = pts @ across
    s = pts @ along
    reach = float(np.abs(s).max()) + spacing
    # As many lines as the width holds spacings, centred: the outermost
    # lines then sit equally far inside both edges, at most 0.55 of a
    # spacing (so the edge is covered about as well as the gap between two
    # lines). The 0.1 slack keeps an area drawn as 1,000 m wide, which
    # measures 1,003 m across lines once the map grid's small rotation
    # against true north is in, from gaining a line for those 3 m.
    extent = float(t.max() - t.min())
    n = max(1, int(math.ceil(extent / spacing - 0.1)))
    start = (t.min() + t.max()) / 2.0 - (n - 1) * spacing / 2.0
    out = []
    for k in range(n):
        offset = start + k * spacing
        p0 = np.array([cx, cy]) + across * offset - along * reach
        p1 = np.array([cx, cy]) + across * offset + along * reach
        cut = poly.intersection(LineString([p0, p1]))
        parts = list(cut.geoms) if isinstance(cut, MultiLineString) else ([cut] if isinstance(cut, LineString) and not cut.is_empty else [])
        for part in parts:
            if part.length < 1.0:
                continue
            q = np.array(part.coords)
            # order along the flight direction, then extend
            if (q[-1] - q[0]) @ along < 0:
                q = q[::-1]
            out.append([tuple(q[0] - along * lead_in), tuple(q[-1] + along * lead_in)])
    return out


def plan_flight(
    polygon_latlon: list[list[float]],
    line_spacing_m: float,
    azimuth_deg: float | None = None,
    tie_spacing_m: float | None = None,
    lead_in_m: float = 30.0,
    speed_mps: float = 5.0,
    endurance_min: float = 25.0,
    turn_time_s: float = 20.0,
    repeat_line: bool = True,
) -> dict:
    """Survey lines, tie lines and a repeated line over a polygon of
    [lat, lon] vertices. `azimuth_deg` None = along the polygon's longer
    side (fewest turns); `tie_spacing_m` None = TIE_LINE_RATIO lines."""
    if len(polygon_latlon) < 3:
        raise SurveyDesignError("조사 구역은 꼭짓점이 3개 이상이어야 합니다.")
    if line_spacing_m <= 0:
        raise SurveyDesignError("측선 간격은 0보다 커야 합니다.")
    lat0 = float(np.mean([p[0] for p in polygon_latlon]))
    lon0 = float(np.mean([p[1] for p in polygon_latlon]))
    fwd, inv, epsg = _utm_transformers(lat0, lon0)
    xy = [fwd.transform(p[1], p[0]) for p in polygon_latlon]
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < line_spacing_m ** 2:
        raise SurveyDesignError("조사 구역이 측선 간격에 비해 너무 작습니다.")
    if isinstance(poly, Polygon) is False:                 # buffer(0) of a bow-tie gives several pieces
        poly = max(poly.geoms, key=lambda g: g.area)

    if azimuth_deg is None:
        box = np.array(poly.minimum_rotated_rectangle.exterior.coords)
        edges = box[1:] - box[:-1]
        longest = edges[np.argmax(np.hypot(edges[:, 0], edges[:, 1]))]
        azimuth_deg = math.degrees(math.atan2(longest[0], longest[1])) % 180.0
    azimuth_deg = float(azimuth_deg) % 180.0

    survey = _parallel_lines(poly, azimuth_deg, line_spacing_m, lead_in_m)
    if not survey:
        raise SurveyDesignError("이 간격과 방향으로는 구역을 지나는 측선이 없습니다.")
    n_lines = len(survey)
    if n_lines > 2000:
        raise SurveyDesignError(f"측선이 {n_lines}개입니다 - 구역이 너무 넓거나 간격이 너무 좁습니다.")

    # tie lines: perpendicular, at least MIN_TIE_LINES of them
    a = math.radians(azimuth_deg)
    along = np.array([math.sin(a), math.cos(a)])
    extent_along = float(np.ptp(np.array(poly.exterior.coords) @ along))
    tie_spacing = float(tie_spacing_m) if tie_spacing_m else line_spacing_m * TIE_LINE_RATIO
    if math.ceil(extent_along / tie_spacing - 0.02) < MIN_TIE_LINES:
        tie_spacing = extent_along / MIN_TIE_LINES          # a small block still gets its three
    ties = _parallel_lines(poly, (azimuth_deg + 90.0) % 180.0, tie_spacing, lead_in_m) if tie_spacing > 0 else []

    # back and forth, so each line starts where the last one ended
    legs = []
    for i, seg in enumerate(survey):
        legs.append({"kind": "survey", "name": f"L{i + 1:03d}", "xy": seg if i % 2 == 0 else seg[::-1]})
    for i, seg in enumerate(ties):
        legs.append({"kind": "tie", "name": f"T{i + 1:02d}", "xy": seg if i % 2 == 0 else seg[::-1]})
    if repeat_line:
        # the first line again, last: the same ground at the start and the end of the survey
        legs.append({"kind": "repeat", "name": "R001 (L001 재비행)", "xy": list(survey[0])})

    # sorties by endurance: a leg is never split, the ferry between legs counts
    budget = endurance_min * 60.0
    sortie, used, previous_end = 1, 0.0, None
    for leg in legs:
        length = math.dist(leg["xy"][0], leg["xy"][1])
        ferry = math.dist(previous_end, leg["xy"][0]) if previous_end is not None else 0.0
        cost = (length + ferry) / speed_mps + turn_time_s
        if used > 0 and used + cost > budget:
            sortie, used = sortie + 1, 0.0
            cost = length / speed_mps + turn_time_s
        used += cost
        leg.update(length_m=round(length, 1), sortie=sortie)
        previous_end = leg["xy"][1]
        if length / speed_mps + turn_time_s > budget:
            leg["exceeds_endurance"] = True

    def to_latlon(seg):
        return [[round(lat, 7), round(lon, 7)] for lon, lat in (inv.transform(x, y) for x, y in seg)]

    features = [{"name": "조사 구역", "geometry": "Polygon", "coordinates": to_latlon(list(poly.exterior.coords)), "properties": {"kind": "area"}}]
    for leg in legs:
        features.append({"name": leg["name"], "geometry": "LineString", "coordinates": to_latlon(leg["xy"]),
                         "properties": {"kind": leg["kind"], "sortie": leg["sortie"], "length_m": leg["length_m"]}})
    total = {kind: round(sum(l["length_m"] for l in legs if l["kind"] == kind), 0) for kind in ("survey", "tie", "repeat")}
    total_m = sum(total.values())
    lats = [p[0] for f in features for p in f["coordinates"]]
    lons = [p[1] for f in features for p in f["coordinates"]]
    warnings = []
    if any(l.get("exceeds_endurance") for l in legs):
        warnings.append("한 번의 비행 시간으로 끝낼 수 없는 측선이 있습니다 - 구역을 나누거나 속도를 올리세요.")
    if len(ties) < MIN_TIE_LINES:
        warnings.append(f"타이라인이 {len(ties)}개뿐입니다 - 비행 중 드리프트와 광역 경사를 가르려면 {MIN_TIE_LINES}개 이상이 필요합니다.")
    return {
        "utm_epsg": epsg,
        "azimuth_deg": round(azimuth_deg, 1),
        "line_spacing_m": round(float(line_spacing_m), 2),
        "tie_spacing_m": round(tie_spacing, 1),
        "n_survey_lines": n_lines,
        "n_tie_lines": len(ties),
        "has_repeat_line": bool(repeat_line),
        "length_m": {**total, "total": round(total_m, 0)},
        "area_km2": round(poly.area / 1e6, 4),
        "n_sorties": sortie,
        "flight_time_min": round(sum(l["length_m"] for l in legs) / speed_mps / 60.0 + len(legs) * turn_time_s / 60.0, 1),
        "warnings": warnings,
        # the reference-layer shape (magcore.vector_layer), so the screen can draw the plan as it draws any vector file
        "layer": {
            "type": "vector", "name": f"비행 계획 ({round(float(line_spacing_m))} m)", "format": "plan", "features": features,
            "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
            "n_points": 0, "n_lines": len(legs), "n_polygons": 1,
        },
    }


_KML_STYLES = {"survey": ("ff1f63a9", 2), "tie": ("ff2f8f2f", 2), "repeat": ("ff2020d0", 3), "area": ("ff404040", 1)}
_KML_FOLDERS = (("survey", "측선"), ("tie", "타이라인"), ("repeat", "재비행 측선"))


def plan_to_kml(plan: dict, name: str = "DroneMag Studio 비행 계획") -> bytes:
    """The plan as KML: a folder each for survey lines, tie lines and the
    repeated line (in flying order), and the area outline."""
    features = plan["layer"]["features"]
    coords = lambda f: " ".join(f"{lon},{lat},0" for lat, lon in f["coordinates"])
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">', "<Document>", f"<name>{escape(name)}</name>",
           f"<description>{escape(f'측선 {plan['n_survey_lines']}개 · 간격 {plan['line_spacing_m']} m · 방위 {plan['azimuth_deg']}° · 타이라인 {plan['n_tie_lines']}개 · 총 {plan['length_m']['total'] / 1000:.1f} km · 비행 {plan['n_sorties']}회')}</description>"]
    for kind, (color, width) in _KML_STYLES.items():
        out.append(f'<Style id="{kind}"><LineStyle><color>{color}</color><width>{width}</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>')
    for kind, label in _KML_FOLDERS:
        legs = [f for f in features if f["properties"].get("kind") == kind]
        if not legs:
            continue
        out.append(f"<Folder><name>{escape(label)}</name>")
        for f in legs:
            p = f["properties"]
            out.append(f"<Placemark><name>{escape(f['name'])}</name><description>{escape(f'비행 {p['sortie']}회차 · {p['length_m']} m')}</description>"
                       f"<styleUrl>#{kind}</styleUrl><LineString><tessellate>1</tessellate><coordinates>{coords(f)}</coordinates></LineString></Placemark>")
        out.append("</Folder>")
    area = next(f for f in features if f["properties"].get("kind") == "area")
    out.append(f"<Placemark><name>조사 구역</name><styleUrl>#area</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{coords(area)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    out += ["</Document>", "</kml>"]
    return "\n".join(out).encode("utf-8")
