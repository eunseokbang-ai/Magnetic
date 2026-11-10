"""Survey design: line spacing from the aliasing of the field's own
spectrum, a detectability screen, and a flight plan with tie lines and a
repeated line (magcore/survey_design.py)."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import Geod

import app  # noqa: F401 - puts the repository root (magcore) on sys.path
from app.main import app as fastapi_app
from magcore.survey_design import (
    SurveyDesignError, aliased_fraction, max_line_spacing, plan_flight, plan_to_kml, recommend, sphere_anomaly_nt,
)
from magcore.vector_layer import load_vector_layer


# ------------------------------------------------------------- line spacing
def test_the_spacing_rules_come_out_of_the_spectrum():
    # Reid (1980): 2h for the field, h for first derivatives, h/2 for second derivatives at ~5 % aliasing
    assert aliased_fraction(200.0, 100.0, 0) == pytest.approx(math.exp(-math.pi), rel=1e-9)        # 4.3 %
    assert aliased_fraction(100.0, 100.0, 1) == pytest.approx(0.050, abs=0.002)
    assert aliased_fraction(50.0, 100.0, 2) == pytest.approx(0.005, abs=0.001)
    assert [round(max_line_spacing(100.0, n), 1) for n in (0, 1, 2)] == [209.7, 99.8, 68.6]
    # consistent both ways, and monotonic: wider lines or a more demanding product alias more
    for n in (0, 1, 2):
        assert aliased_fraction(max_line_spacing(80.0, n, 0.05), 80.0, n) == pytest.approx(0.05, rel=1e-6)
    assert aliased_fraction(100.0, 50.0, 0) > aliased_fraction(50.0, 50.0, 0)
    assert aliased_fraction(100.0, 50.0, 2) > aliased_fraction(100.0, 50.0, 1) > aliased_fraction(100.0, 50.0, 0)


def test_the_formula_agrees_with_aliasing_measured_on_a_synthetic_field():
    """Sample a field with spectrum exp(-kh) (power exp(-2kh)) at spacing dx
    and count the power that lies above the Nyquist wavenumber."""
    h, n, step = 40.0, 2 ** 16, 0.5
    k = 2 * np.pi * np.fft.rfftfreq(n, step)
    power = np.exp(-2 * k * h)
    for dx in (40.0, 80.0, 160.0):
        measured = power[k > np.pi / dx].sum() / power[1:].sum()
        assert aliased_fraction(dx, h, 0) == pytest.approx(measured, rel=0.03)
        measured_gradient = (k ** 2 * power)[k > np.pi / dx].sum() / (k ** 2 * power).sum()
        assert aliased_fraction(dx, h, 1) == pytest.approx(measured_gradient, rel=0.03)


def test_recommendation_for_a_target_and_what_each_product_costs():
    r = recommend(target_depth_m=20.0, flight_height_m=30.0, product="first_derivative")
    assert r["source_distance_m"] == 50.0 and r["line_spacing_m"] == pytest.approx(49.9, abs=0.1)
    assert r["spacing_by_product"] == {"field": 104.9, "first_derivative": 49.9, "second_derivative": 34.3}
    assert r["tie_line_spacing_m"] == pytest.approx(499, abs=1) and r["anomaly_width_m"] == 50.0
    assert r["along_line_sample_m"] == 0.5 and r["warnings"] == [] and "detectability" not in r
    # the Andong block after the fact: sources ~38 m from the sensor, flown at 100 m spacing
    assert aliased_fraction(100.0, 38.2, 1) > 0.5                                    # most of the gradient power aliased
    assert max_line_spacing(38.2, 2) == pytest.approx(26.2, abs=0.1)
    with pytest.raises(SurveyDesignError):
        recommend(10.0, 0.0)
    with pytest.raises(SurveyDesignError):
        recommend(10.0, 30.0, product="nope")


def test_detectability_screen():
    # a 5 m sphere of magnetite-rich rock (chi 0.1 SI) 20 m down, flown 30 m up, in a 50,000 nT field
    assert sphere_anomaly_nt(5.0, 0.1, 50000.0, 50.0) == pytest.approx((2 / 3) * 0.1 * 50000 * 1e-3)   # 3.33 nT
    r = recommend(20.0, 30.0, target_radius_m=5.0, susceptibility_si=0.1, noise_nt=0.5)
    d = r["detectability"]
    assert d["peak_anomaly_nt"] == pytest.approx(3.33, abs=0.01) and d["snr"] == pytest.approx(6.7, abs=0.1) and d["detectable"] is True
    # the anomaly falls as 1/h^3: at the stated limit it is exactly min_snr x noise
    assert sphere_anomaly_nt(5.0, 0.1, 50000.0, d["max_source_distance_m"]) == pytest.approx(1.5, rel=0.01)
    assert d["max_flight_height_m"] == pytest.approx(d["max_source_distance_m"] - 20.0)
    # the same body from 80 m up is below the noise criterion, and the result says how low to fly
    high = recommend(20.0, 80.0, target_radius_m=5.0, susceptibility_si=0.1, noise_nt=0.5)
    assert high["detectability"]["detectable"] is False and any("낮춰야" in w for w in high["warnings"])
    # coarse along-line sampling is flagged
    assert any("샘플 간격" in w for w in recommend(0.0, 3.0, speed_mps=10.0, sample_rate_hz=5.0)["warnings"])


# ------------------------------------------------------------- flight plan
GEOD = Geod(ellps="WGS84")


def _offset(lat, lon, east_m, north_m):
    """[lat, lon] of the point east_m / north_m from (lat, lon), on the ellipsoid."""
    lon1, lat1, _ = GEOD.fwd(lon, lat, 90.0, east_m)
    lon2, lat2, _ = GEOD.fwd(lon1, lat1, 0.0, north_m)
    return [lat2, lon2]


def _rectangle(width_m=1000.0, height_m=600.0, lat0=36.5, lon0=128.5):
    return [_offset(lat0, lon0, x, y) for x, y in ((0, 0), (width_m, 0), (width_m, height_m), (0, height_m))]


def _legs(plan, kind):
    return [f for f in plan["layer"]["features"] if f["properties"].get("kind") == kind]


def _metres(a, b):
    return GEOD.inv(a[1], a[0], b[1], b[0])[2]


def test_lines_cover_the_area_at_the_spacing_with_lead_ins_ties_and_a_repeat():
    plan = plan_flight(_rectangle(), line_spacing_m=50.0, lead_in_m=30.0)
    assert plan["azimuth_deg"] == pytest.approx(90.0, abs=1.5)                       # along the longer (east-west) side
    survey = _legs(plan, "survey")
    assert plan["n_survey_lines"] == len(survey) == 12                               # 600 m / 50 m
    # each line is the 1,000 m crossing plus a 30 m lead-in at both ends
    assert all(_metres(*f["coordinates"]) == pytest.approx(1060.0, abs=2.0) for f in survey)
    # neighbouring lines are 50 m apart and flown in opposite directions
    lats = [np.mean([p[0] for p in f["coordinates"]]) for f in survey]
    gaps = [_metres([lats[i], 128.5], [lats[i + 1], 128.5]) for i in range(len(lats) - 1)]
    assert gaps == pytest.approx([50.0] * 11, abs=0.5)
    assert survey[0]["coordinates"][0][1] < survey[0]["coordinates"][1][1] and survey[1]["coordinates"][0][1] > survey[1]["coordinates"][1][1]
    # the outermost lines sit the same distance inside both edges
    south, north = 36.5, _offset(36.5, 128.5, 0.0, 600.0)[0]
    margins = sorted([_metres([min(lats), 128.5], [south, 128.5]), _metres([max(lats), 128.5], [north, 128.5])])
    assert margins == pytest.approx([25.0, 25.0], abs=1.0)                           # half a spacing inside each edge
    # tie lines at right angles: at least three even though 10 x spacing would give two
    ties = _legs(plan, "tie")
    assert plan["n_tie_lines"] == len(ties) >= 3
    assert all(abs(f["coordinates"][0][1] - f["coordinates"][1][1]) < 1e-4 for f in ties)   # north-south
    # the first line is flown again as the last leg
    (repeat,) = _legs(plan, "repeat")
    assert repeat["coordinates"] == survey[0]["coordinates"] and plan["layer"]["features"][-1] is repeat
    assert plan["length_m"]["total"] == pytest.approx(plan["length_m"]["survey"] + plan["length_m"]["tie"] + plan["length_m"]["repeat"])
    assert plan["area_km2"] == pytest.approx(0.6, abs=0.01) and plan["warnings"] == []


def test_sorties_follow_endurance_and_an_explicit_azimuth_is_respected():
    plan = plan_flight(_rectangle(), line_spacing_m=50.0, azimuth_deg=0.0, speed_mps=5.0, endurance_min=10.0, repeat_line=False)
    survey = _legs(plan, "survey")
    assert plan["azimuth_deg"] == 0.0 and len(survey) == 20 and not _legs(plan, "repeat")     # 1,000 m / 50 m, north-south
    assert all(abs(f["coordinates"][0][1] - f["coordinates"][1][1]) < 1e-4 for f in survey)
    sorties = [f["properties"]["sortie"] for f in plan["layer"]["features"][1:]]
    assert sorties == sorted(sorties) and plan["n_sorties"] == max(sorties) > 1
    # no sortie flies longer than the endurance (legs + turns; ferries between legs are short here)
    for s in set(sorties):
        seconds = sum(f["properties"]["length_m"] / 5.0 + 20.0 for f in plan["layer"]["features"][1:] if f["properties"]["sortie"] == s)
        assert seconds <= 10.0 * 60.0 + 60.0
    # a line that cannot be flown on one battery is reported, not hidden
    long_plan = plan_flight(_rectangle(width_m=6000.0), line_spacing_m=100.0, endurance_min=10.0)
    assert any("한 번의 비행" in w for w in long_plan["warnings"])


def test_a_concave_area_cuts_lines_instead_of_flying_through_the_gap():
    m = lambda x, y: _offset(36.5, 128.5, x, y)
    u_shape = [m(0, 0), m(900, 0), m(900, 600), m(600, 600), m(600, 200), m(300, 200), m(300, 600), m(0, 600)]
    plan = plan_flight(u_shape, line_spacing_m=100.0, azimuth_deg=90.0, lead_in_m=0.0, repeat_line=False)
    survey = _legs(plan, "survey")
    lengths = sorted(round(_metres(*f["coordinates"]), -1) for f in survey)
    assert lengths.count(900.0) == 2 and lengths.count(300.0) == 8            # two full crossings, four rows cut into two arms each


def test_bad_areas_are_refused():
    with pytest.raises(SurveyDesignError, match="꼭짓점"):
        plan_flight([[36.5, 128.5], [36.6, 128.6]], 50.0)
    with pytest.raises(SurveyDesignError, match="너무 작습니다"):
        plan_flight(_rectangle(10.0, 10.0), 50.0)
    with pytest.raises(SurveyDesignError, match="너무 넓거나"):
        plan_flight(_rectangle(50000.0, 50000.0), 5.0)


def test_the_kml_reads_back_as_the_same_lines():
    plan = plan_flight(_rectangle(), line_spacing_m=100.0)
    kml = plan_to_kml(plan)
    layer = load_vector_layer(kml, "plan.kml")                                   # the app's own KML reader
    assert layer["n_lines"] == plan["n_survey_lines"] + plan["n_tie_lines"] + 1 and layer["n_polygons"] == 1
    names = [f["name"] for f in layer["features"]]
    assert names[0] == "L001" and "T01" in names and any(n.startswith("R001") for n in names)
    first = next(f for f in layer["features"] if f["name"] == "L001")
    np.testing.assert_allclose(first["coordinates"], plan["layer"]["features"][1]["coordinates"], atol=1e-6)
    text = kml.decode("utf-8")
    assert "<name>타이라인</name>" in text and "<name>재비행 측선</name>" in text


# ------------------------------------------------------------- the API
def test_endpoints_recommend_plan_and_export():
    client = TestClient(fastapi_app)
    r = client.post("/api/survey-design/recommend", json={"target_depth_m": 20, "flight_height_m": 30, "product": "field"})
    assert r.status_code == 200 and r.json()["line_spacing_m"] == pytest.approx(104.9, abs=0.1)
    body = {"polygon": _rectangle(), "line_spacing_m": 50.0}
    r = client.post("/api/survey-design/plan", json=body)
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["n_survey_lines"] == 12 and plan["layer"]["type"] == "vector" and plan["layer"]["n_lines"] == 12 + plan["n_tie_lines"] + 1
    r = client.post("/api/survey-design/plan/kml", json=body)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/vnd.google-earth.kml+xml")
    assert load_vector_layer(r.content, "plan.kml")["n_lines"] == plan["layer"]["n_lines"]
    r = client.post("/api/survey-design/plan", json={"polygon": _rectangle(10.0, 10.0), "line_spacing_m": 50.0})
    assert r.status_code == 400 and "너무 작습니다" in r.json()["detail"]
