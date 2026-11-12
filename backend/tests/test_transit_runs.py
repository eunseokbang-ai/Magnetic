"""The ferry legs between the takeoff point and the survey block.

A flight is flown to its first line and home from its last, and where
such a leg points roughly along the lines it used to be kept as a survey
line - a diagonal across the block. detect_lines now takes those out
(magcore.lines._exclude_transit_runs). Checked here on flights built leg
by leg, where what is a line and what is not is known, and on the survey
the rule was measured on (GaSaDo 2026-11; skipped where the field data is
absent - point MAGNETIC_GASADO_DIR at it to run elsewhere).

Also the stretch a click on the map selects (project_common.point_run_ids)
and the list the line editor shows (transit_summary).
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.project_common import point_run_ids, transit_summary
from magcore.lines import LineDetectionParams, detect_lines

LAT0, LON0 = 34.48, 126.07
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))
DT = 0.1            # 10 Hz, as the magnetometer records
CRUISE = 5.0        # m/s along the lines
SPACING = 50.0
LENGTH = 600.0


def fly(start, legs):
    """Positions every DT seconds along `legs` from `start`: each leg is
    (x, y, speed) to fly straight there, or ("hover", seconds)."""
    xs, ys = [start[0]], [start[1]]
    for leg in legs:
        if leg[0] == "hover":
            n = int(round(leg[1] / DT))
            xs += [xs[-1]] * n
            ys += [ys[-1]] * n
            continue
        x1, y1, speed = leg
        x0, y0 = xs[-1], ys[-1]
        n = max(1, int(round(np.hypot(x1 - x0, y1 - y0) / (speed * DT))))
        f = np.arange(1, n + 1) / n
        xs += list(x0 + (x1 - x0) * f)
        ys += list(y0 + (y1 - y0) * f)
    return np.array(xs), np.array(ys)


def lines_legs(n_lines, x0=0.0, speed=CRUISE, speeds=None):
    """North-south lines SPACING apart, flown alternately up and down with
    a square turn between them, starting at (x0, 0) heading north."""
    legs = []
    for k in range(n_lines):
        x = x0 + k * SPACING
        v = (speeds or {}).get(k, speed)
        top = k % 2 == 0
        legs.append((x, LENGTH if top else 0.0, v))
        if k < n_lines - 1:
            legs.append((x + SPACING, LENGTH if top else 0.0, speed))
    return legs


def frame(x, y, source=0, t0="2026-11-10T01:00:00"):
    t = pd.Timestamp(t0) + pd.to_timedelta(np.arange(len(x)) * DT, unit="s")
    return pd.DataFrame({
        "timestamp": t, "lat": LAT0 + y / M_PER_DEG_LAT, "lon": LON0 + x / M_PER_DEG_LON,
        "source_file_index": source, "sx": x, "sy": y,
    })


def on_a_line(df):
    """True where the built position is on one of the survey lines."""
    return (np.abs(df["sx"] - np.round(df["sx"] / SPACING) * SPACING) < 0.5) & df["sy"].between(-0.5, LENGTH + 0.5)


def survey_with_ferry_legs(ferry_out_speed=10.0, ferry_home_speed=12.0):
    # up from the takeoff point, a hand-flown leg 10 deg off the lines to
    # the first line's start, a pause to turn onto it, four lines, and a
    # leg home that happens to run parallel to them
    x, y = fly((-60.0, -350.0), [
        (0.0, 0.0, ferry_out_speed), ("hover", 3.0),
        *lines_legs(4), ("hover", 2.0),
        (152.0, -400.0, ferry_home_speed), ("hover", 2.0),
    ])
    return frame(x, y)


def test_ferry_legs_at_both_ends_are_taken_out_and_the_lines_left_whole():
    df = survey_with_ferry_legs()
    out = detect_lines(df)
    kept = out["line_id"] >= 0

    assert out.loc[kept, "line_id"].nunique() == 4
    assert on_a_line(out[kept]).all(), "something off the lines was kept"
    # each line is still there end to end, but for the turn buffer
    for k in range(4):
        y_kept = out.loc[kept & (np.abs(out["sx"] - k * SPACING) < 0.5), "sy"]
        assert y_kept.min() < 15 and y_kept.max() > LENGTH - 15

    runs = out.attrs["transit_runs"]
    assert [(r["verdict"], r["where"]) for r in runs] == [("transit", "start"), ("transit", "end")]
    out_leg, home_leg = runs
    assert out_leg["why"] == ["fast", "off_direction"] and abs(abs(out_leg["azimuth_deviation_deg"]) - 9.7) < 0.5
    assert home_leg["why"] == ["fast"] and abs(home_leg["azimuth_deviation_deg"]) < 1.0
    assert out_leg["line_speed_mps"] == pytest.approx(CRUISE, rel=0.02)

    # the legs keep their own name: not folded into the hidden, protected ramps
    ferry = ~on_a_line(out) & (out["sy"] < -20)
    named = out.loc[ferry, "exclusion_reason"]
    assert (named == "transit").mean() > 0.9
    assert set(named.unique()) <= {"transit", "takeoff_ramp", "landing_ramp"}


def test_without_the_step_the_ferry_legs_are_survey_lines_again():
    df = survey_with_ferry_legs()
    out = detect_lines(df, LineDetectionParams(remove_transit=False))
    kept = out["line_id"] >= 0
    # ... and worse than lines of their own: each is joined to the line it
    # leads to or from, a line with a tail across the block
    assert out.loc[kept, "line_id"].nunique() == 4
    assert (~on_a_line(out[kept])).sum() > 500
    assert out.attrs["transit_runs"] == []


def test_a_leg_at_line_speed_goes_by_its_direction_alone():
    out = detect_lines(survey_with_ferry_legs(ferry_out_speed=CRUISE, ferry_home_speed=CRUISE))
    runs = out.attrs["transit_runs"]
    # the leg out is 10 deg off; the one home is parallel and no faster
    # than a line, and nothing tells it from one - it stays, as the tail
    # of the line it left from, for the operator to click away
    assert [(r["verdict"], r["where"], r["why"]) for r in runs] == [("transit", "start", ["off_direction"])]
    kept = out["line_id"] >= 0
    assert out.loc[kept, "line_id"].nunique() == 4
    assert not (kept & (out["sy"] < -20) & (out["sx"] < 100)).any() and (kept & (out["sy"] < -20)).sum() > 500


def test_lines_flown_slowly_over_steep_ground_are_not_what_fast_is_measured_against():
    # terrain following: two of the lines crawl at a quarter of the mission
    # speed. Against the flight's *average* line speed the others would be
    # "fast"; against its cruise speed nothing is but the ferry legs.
    x, y = fly((-60.0, -350.0), [
        (0.0, 0.0, 10.0), ("hover", 3.0),
        *lines_legs(6, speeds={0: 1.2, 1: 1.2, 2: 1.2, 3: 1.2}), ("hover", 2.0),
        (252.0, -400.0, 12.0),
    ])
    out = detect_lines(frame(x, y))
    kept = out["line_id"] >= 0
    assert out.loc[kept, "line_id"].nunique() == 6 and on_a_line(out[kept]).all()
    assert [r["verdict"] for r in out.attrs["transit_runs"]] == ["transit", "transit"]


def test_nothing_between_two_lines_is_removed_only_reported():
    # the third of seven lines flown at twice the speed: odd, and in the
    # middle of the flight, so it is not a leg to or from anywhere
    x, y = fly((0.0, 0.0), [("hover", 2.0), *lines_legs(7, speeds={2: 10.0}), ("hover", 2.0)])
    out = detect_lines(frame(x, y))
    kept = out["line_id"] >= 0
    assert out.loc[kept, "line_id"].nunique() == 7
    (run,) = out.attrs["transit_runs"]
    assert (run["verdict"], run["where"], run["why"]) == ("suspect", "middle", ["fast"])
    # ... under the id the line has in the result
    fast_line = int(out.loc[kept & (np.abs(out["sx"] - 2 * SPACING) < 0.5), "line_id"].iloc[0])
    assert run["line_id"] == fast_line


def test_a_flight_flown_another_way_is_a_block_not_a_ferry_leg():
    x0, y0 = fly((0.0, 0.0), [("hover", 2.0), *lines_legs(6), ("hover", 2.0)])
    # a second flight, three lines 10 deg off the first block's
    a = np.radians(10.0)
    legs = []
    for k in range(3):
        bx = 400.0 + k * SPACING
        ends = [(bx, 0.0), (bx + LENGTH * np.sin(a), LENGTH * np.cos(a))]
        if k % 2:
            ends.reverse()
        legs += [(*ends[0], CRUISE), (*ends[1], CRUISE)]
    x1, y1 = fly(legs[0][:2], [("hover", 2.0), *legs[1:], ("hover", 2.0)])
    df = pd.concat([frame(x0, y0, 0), frame(x1, y1, 1, t0="2026-11-10T03:00:00")], ignore_index=True)

    out = detect_lines(df)
    kept = out["line_id"] >= 0
    assert out.loc[kept & (out["source_file_index"] == 1), "line_id"].nunique() == 3
    assert (out["exclusion_reason"] == "transit").sum() == 0
    runs = out.attrs["transit_runs"]
    assert len(runs) == 3
    assert all(r["verdict"] == "suspect" and r["where"] == "flight" and "whole_flight" in r["why"] for r in runs)


def test_a_click_selects_one_pass_one_leg_one_turn():
    out = detect_lines(survey_with_ferry_legs())
    out["point_id"] = np.arange(len(out))
    run = point_run_ids(out)

    # numbered in the order flown, every stretch one piece of the record
    assert (np.diff(run) >= 0).all() and run[0] == 0
    for rid in np.unique(run):
        part = out[run == rid]
        assert part["line_id"].nunique() == 1
    # a line is one stretch; a ferry leg is one stretch of its own
    for lid in sorted(out.loc[out["line_id"] >= 0, "line_id"].unique()):
        assert len(np.unique(run[(out["line_id"] == lid).to_numpy()])) == 1
    transit = (out["exclusion_reason"] == "transit").to_numpy()
    assert len(np.unique(run[transit])) == 2
    assert not np.isin(run[~transit], np.unique(run[transit])).any()

    runs = [{**r, "first_point_id": r["first_index"], "last_point_id": r["last_index"]} for r in out.attrs["transit_runs"]]
    summary = transit_summary(runs, out, True, ["flight1.csv"])
    assert (summary["n_removed"], summary["n_suspect"]) == (2, 0)
    assert summary["n_points_removed"] == int(transit.sum())
    assert summary["length_removed_m"] == pytest.approx(355 + 400, abs=50)   # less the turn buffers and the lag at each end
    for item in summary["runs"]:
        assert item["source_file"] == "flight1.csv"
        assert (run == item["run_id"]).sum() == item["n_points"] and transit[run == item["run_id"]].all()
    assert transit_summary([], out, False)["enabled"] is False


# ---------------------------------------------------------------------------
# The survey the rule was measured on.
# ---------------------------------------------------------------------------
GASADO = pathlib.Path(os.environ.get("MAGNETIC_GASADO_DIR") or pathlib.Path(__file__).resolve().parents[2] / "GaSaDo")
needs_gasado = pytest.mark.skipif(not list(GASADO.glob("SRVY0-ACQU*.csv")), reason=f"GaSaDo field data not present ({GASADO})")


@pytest.fixture(scope="module")
def gasado():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    # the 24 flights the rule was measured on (2026-11-03..11); the survey went on after them
    number = lambda f: int(f.name.split("ACQU")[1].split("-")[0])  # noqa: E731
    files = sorted((f for f in GASADO.glob("SRVY0-ACQU*.csv") if 85 <= number(f) <= 108), key=number)
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files])
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}})
    assert r.status_code == 200, r.text
    return {"client": client, "pid": pid, "process": r.json()}


@needs_gasado
def test_gasado_ferry_legs(gasado):
    p = gasado["process"]
    # 55 "lines" before, eleven of them ferry legs or made of them
    assert p["n_lines"] == 44
    t = p["transit"]
    assert (t["enabled"], t["n_removed"], t["n_points_removed"], t["n_suspect"]) == (True, 36, 5907, 0)
    assert t["length_removed_m"] == pytest.approx(6067.5, abs=1.0)
    ratios = [r["speed_mps"] / r["line_speed_mps"] for r in t["runs"]]
    assert min(ratios) > 1.2
    # what is left runs north-south: every line within a degree and a half
    # of the others, where the legs were up to 20 deg off
    from app.store import store

    df = store.get(gasado["pid"]).processed
    kept = df[df["line_id"] >= 0]
    for _lid, g in kept.groupby("line_id"):
        (slope, _), resid, *_ = np.polyfit(g["y"], g["x"], 1, full=True)
        assert abs(np.degrees(np.arctan(slope)) - 1.70) < 1.5


@needs_gasado
def test_gasado_click_a_leg_and_put_it_back(gasado):
    client, pid = gasado["client"], gasado["pid"]
    leg = next(r for r in gasado["process"]["transit"]["runs"] if r["source_file"] == "SRVY0-ACQU96-10Hz.csv" and r["where"] == "start")
    points = client.get(f"/api/projects/{pid}/points", params={"value": "anomaly"}).json()
    points = points["points"] if isinstance(points, dict) else points
    selected = [q for q in points if q["run_id"] == leg["run_id"]]
    assert len(selected) == leg["n_points"] == 244
    assert all(q["excluded"] and not q["is_ramp"] and q["line_id"] == -1 for q in selected)

    before = gasado["process"]["n_kept"]
    ids = [q["point_id"] for q in selected]
    r = client.post(f"/api/projects/{pid}/manual-exclude", json={"mode": "point_ids", "action": "include", "point_ids": ids})
    assert r.status_code == 200 and (r.json()["n_kept"], r.json()["n_manual_included"]) == (before + 244, 244)
    # taking it out again is back to what detection decided: no override left
    r = client.post(f"/api/projects/{pid}/manual-exclude", json={"mode": "point_ids", "action": "exclude", "point_ids": ids})
    assert (r.json()["n_kept"], r.json()["n_manual_included"], r.json()["n_manual_excluded"]) == (before, 0, 0)
