"""Synthetic validation of the tie-line detection + crossover leveling
algorithm: build a small parallel-line + perpendicular-tie-line survey by
hand, inject known per-line level errors, and confirm the leveling
recovers offsets close to the injected ones and reduces crossover RMS."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.processing.crossover_leveling import compute_crossover_leveling
from app.processing.lines import LineDetectionParams, detect_lines, detect_tie_lines

RNG = np.random.default_rng(0)


def _make_synthetic_survey():
    """5 N-S survey lines at x = 0, 50, 100, 150, 200 m, y in [0, 400], plus
    1 E-W tie line at y=200, x in [-20, 220]. A smooth background field
    (function of x, y only - the "real" geology, identical for both line
    families) plus a per-survey-line constant level error is injected."""
    injected_shift = {0: 5.0, 1: -3.0, 2: 8.0, 3: 0.0, 4: -6.0}

    def geology(x, y):
        return 50.0 + 0.02 * x + 0.01 * y  # smooth, so crossovers should agree exactly pre-noise

    rows = []
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    t = t0
    line_id_counter = 0
    turn_gap = pd.Timedelta(seconds=60)  # bigger than max_gap_seconds, so each line is its own contiguous run
    for i, x0 in enumerate([0.0, 50.0, 100.0, 150.0, 200.0]):
        t += turn_gap
        for y in np.arange(0.0, 400.0, 2.0):
            val = geology(x0, y) + injected_shift[i]
            rows.append({"x": x0, "y": y, "value": val, "line_id": line_id_counter, "t": t})
            t += pd.Timedelta(seconds=1)
        line_id_counter += 1

    tie_rows = []
    t += turn_gap
    for x in np.arange(-20.0, 220.0, 2.0):
        val = geology(x, 200.0)  # tie line NOT shifted - it's the reference
        tie_rows.append({"x": x, "y": 200.0, "value": val, "t": t})
        t += pd.Timedelta(seconds=1)

    survey_df = pd.DataFrame(rows)
    tie_df = pd.DataFrame(tie_rows)
    return survey_df, tie_df, injected_shift


def test_crossover_leveling_recovers_injected_shifts():
    survey_df, tie_df, injected_shift = _make_synthetic_survey()

    df = pd.concat([survey_df[["x", "y", "value", "t"]], tie_df[["x", "y", "value", "t"]]], ignore_index=True)
    df = df.sort_values("t").reset_index(drop=True)
    df["timestamp"] = df["t"]
    # fabricate lat/lon so detect_lines' UTM projection round-trips close enough
    # to the original local x, y (use a local equirectangular-ish approximation
    # centered far from poles/dateline so meters-per-degree is ~constant).
    lat0, lon0 = 37.5, 127.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    df["lat"] = lat0 + df["y"] / m_per_deg_lat
    df["lon"] = lon0 + df["x"] / m_per_deg_lon

    params = LineDetectionParams(
        heading_lag_seconds=1.0, heading_tolerance_deg=20.0, min_speed_mps=0.1, min_line_length_m=50.0, turn_buffer_m=2.0, max_gap_seconds=5.0
    )
    detected = detect_lines(df, params)
    dominant_azimuth = detected.attrs["dominant_azimuth_deg"]
    assert detected.loc[detected["line_id"] >= 0, "line_id"].nunique() == 5, "expected 5 detected survey lines"

    tie_line_id = detect_tie_lines(detected, dominant_azimuth, params, tie_tolerance_deg=20.0)
    assert (tie_line_id >= 0).sum() > 50, "expected the tie line to be detected"

    # iterative=False: this test's synthetic setup treats the single tie
    # line as a perfect, unshifted reference (only survey lines carry an
    # injected error) - exactly the assumption the non-iterative single
    # pass makes, so it's the right mode to check exact shift recovery
    # against. The default iterative network mode is covered separately
    # in test_dronemagadv_features.py, where it fairly redistributes
    # residual between survey and tie lines instead of trusting the tie
    # line completely.
    result = compute_crossover_leveling(detected, tie_line_id, "value", max_crossover_distance_m=5.0, iterative=False)
    assert result.applied, result.reason
    assert result.n_tie_lines == 1
    assert result.n_survey_lines_corrected == 5

    # recovered shift should be the negative of the injected shift (leveling
    # brings the survey line up/down to match the fixed tie-line reference).
    # match detected line_id -> injected index by x-coordinate ordering (both sorted ascending)
    centroid_x = detected[detected["line_id"] >= 0].groupby("line_id")["x"].mean().sort_values()
    ordered_detected_ids = centroid_x.index.tolist()
    for injected_idx, detected_lid in enumerate(ordered_detected_ids):
        expected = -injected_shift[injected_idx]
        got = result.line_shifts.get(detected_lid)
        assert got is not None, f"line {detected_lid} got no crossover shift"
        assert abs(got - expected) < 0.5, f"line {detected_lid}: expected shift ~{expected}, got {got}"

    assert result.rms_after_nt < result.rms_before_nt
    assert result.rms_after_nt < 0.5
    print("recovered shifts:", result.line_shifts, "rms before/after:", result.rms_before_nt, result.rms_after_nt)


def _make_synthetic_survey_with_linear_error():
    """Same 5 N-S survey lines as above, but with 2 E-W tie lines (y=100,
    y=300, both fixed/unshifted references) and a per-line error that
    varies *linearly* with y (centered on the survey's own y-midpoint, so
    the true intercept at each line's own centroid is ~0) instead of a
    flat per-line constant - the case leveling_order=1 exists for."""
    slope_per_line = {0: 0.05, 1: -0.03, 2: 0.08, 3: 0.0, 4: -0.06}  # nT/m

    def geology(x, y):
        return 50.0 + 0.02 * x + 0.01 * y

    rows = []
    t = pd.Timestamp("2026-01-01T00:00:00")
    line_id_counter = 0
    turn_gap = pd.Timedelta(seconds=60)
    for i, x0 in enumerate([0.0, 50.0, 100.0, 150.0, 200.0]):
        t += turn_gap
        for y in np.arange(0.0, 400.0, 2.0):
            val = geology(x0, y) + slope_per_line[i] * (y - 200.0)
            rows.append({"x": x0, "y": y, "value": val, "line_id": line_id_counter, "t": t})
            t += pd.Timedelta(seconds=1)
        line_id_counter += 1

    tie_rows = []
    for y_tie in (100.0, 300.0):
        t += turn_gap
        for x in np.arange(-20.0, 220.0, 2.0):
            val = geology(x, y_tie)  # tie lines NOT shifted - fixed reference
            tie_rows.append({"x": x, "y": y_tie, "value": val, "t": t})
            t += pd.Timedelta(seconds=1)

    survey_df = pd.DataFrame(rows)
    tie_df = pd.DataFrame(tie_rows)
    return survey_df, tie_df, slope_per_line


def test_polynomial_leveling_recovers_linear_error():
    survey_df, tie_df, slope_per_line = _make_synthetic_survey_with_linear_error()

    df = pd.concat([survey_df[["x", "y", "value", "t"]], tie_df[["x", "y", "value", "t"]]], ignore_index=True)
    df = df.sort_values("t").reset_index(drop=True)
    df["timestamp"] = df["t"]
    lat0, lon0 = 37.5, 127.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    df["lat"] = lat0 + df["y"] / m_per_deg_lat
    df["lon"] = lon0 + df["x"] / m_per_deg_lon

    params = LineDetectionParams(
        heading_lag_seconds=1.0, heading_tolerance_deg=20.0, min_speed_mps=0.1, min_line_length_m=50.0, turn_buffer_m=2.0, max_gap_seconds=5.0
    )
    detected = detect_lines(df, params)
    dominant_azimuth = detected.attrs["dominant_azimuth_deg"]
    assert detected.loc[detected["line_id"] >= 0, "line_id"].nunique() == 5

    tie_line_id = detect_tie_lines(detected, dominant_azimuth, params, tie_tolerance_deg=20.0)
    assert tie_line_id[tie_line_id >= 0].nunique() == 2, "expected both tie lines to be detected"

    result = compute_crossover_leveling(
        detected, tie_line_id, "value", max_crossover_distance_m=5.0, leveling_order=1
    )
    assert result.applied, result.reason
    assert result.line_shifts_linear is not None

    centroid_x = detected[detected["line_id"] >= 0].groupby("line_id")["x"].mean().sort_values()
    ordered_detected_ids = centroid_x.index.tolist()

    before_values = detected["value"].to_numpy(copy=True)
    from app.processing.crossover_leveling import apply_crossover_leveling

    leveled_values = apply_crossover_leveling(detected, "value", result)

    for injected_idx, detected_lid in enumerate(ordered_detected_ids):
        true_slope = slope_per_line[injected_idx]
        _intercept, recovered_slope, *_rest = result.line_shifts_linear[detected_lid]
        # recovered slope has opposite sign of the injected one (levelling
        # subtracts the error back out)
        assert abs(-recovered_slope - true_slope) < 0.02, (
            f"line {detected_lid}: expected recovered slope ~{-true_slope}, got {recovered_slope}"
        )

    # the levelled data should be much flatter (less y-dependent residual
    # per line) than before - check overall variance reduction as a coarse
    # end-to-end sanity check that apply_crossover_leveling's per-point
    # linear correction actually matches what was fit.
    line0_mask = (detected["line_id"] == ordered_detected_ids[2]).to_numpy()  # slope 0.08, largest
    resid_before = before_values[line0_mask] - float(np.mean(before_values[line0_mask]))
    resid_after = leveled_values[line0_mask] - float(np.mean(leveled_values[line0_mask]))
    assert np.std(resid_after) < np.std(resid_before) * 0.3


def test_iterative_leveling_anchors_on_zero_mean_survey_shift_not_pooled():
    """Regression test: the iterative network adjustment's per-pass anchor
    used to subtract the mean of survey+tie shifts POOLED together, even
    though only the survey shifts are ever added back to the real data
    (apply_crossover_leveling never touches tie-line points). That let a
    net DC offset - however the pass happened to split it between the two
    line families - leak into the leveled data as a side effect of an
    anchor that's only supposed to fix leveling's one true degree of
    freedom (a constant added to every line, survey and tie alike,
    doesn't change any crossover difference).

    Two survey lines cross one tie line, with crossover differences of
    +10 and 0 (the same numbers used to illustrate the bug) - the fix
    should converge the two survey shifts to +5/-5 (mean exactly 0), not
    the old code's ~+6.67/-3.33 (mean ~+1.67, which was quietly shifting
    the whole leveled survey up by that amount)."""
    df = pd.DataFrame({
        "x": [-5.0, 5.0, 15.0, 25.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        "y": [0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 5.0, -5.0, 0.0, 5.0],
        "line_id": [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
        "value": [0.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0],
    })
    tie_line_id = pd.Series([0, 0, 0, 0, -1, -1, -1, -1, -1, -1])

    result = compute_crossover_leveling(df, tie_line_id, "value", max_crossover_distance_m=15.0, iterative=True)
    assert result.applied, result.reason
    assert set(result.line_shifts) == {0, 1}

    shifts = np.array(list(result.line_shifts.values()))
    assert abs(float(np.mean(shifts))) < 1e-6, f"survey shifts should average to zero, got {result.line_shifts}"
    assert result.line_shifts[0] == pytest.approx(5.0, abs=0.01)
    assert result.line_shifts[1] == pytest.approx(-5.0, abs=0.01)


if __name__ == "__main__":
    test_crossover_leveling_recovers_injected_shifts()
    test_polynomial_leveling_recovers_linear_error()
    test_iterative_leveling_anchors_on_zero_mean_survey_shift_not_pooled()
    print("ALL CHECKS PASSED")
