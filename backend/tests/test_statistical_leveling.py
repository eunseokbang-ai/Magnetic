"""Statistical leveling and the local-plane heading correction - the two
corrections aimed at visible striping.

The synthetic surveys below are built the way the real problem looks: a
smooth geological field that every line samples honestly, plus a level
error added per line. A correction works if it takes the striping out
without taking the geology with it, so most tests here assert on both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.processing.leveling import apply_heading_correction, compute_heading_correction
from app.processing.microlevel import apply_microleveling
from app.processing.statistical_leveling import (
    apply_statistical_leveling,
    compute_statistical_leveling,
)

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"

SPACING = 50.0
LINE_LENGTH = 600.0
N_LINES = 12
# Lines run north (azimuth 90 in this codebase's atan2(dy, dx) convention),
# so x is the across-line axis and y the along-line one.
AZIMUTH = 90.0


def _geology(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth field with real structure at scales well above the line
    spacing - what a correction must leave alone."""
    return (
        30.0 * np.sin(2 * np.pi * x / 900.0)
        + 20.0 * np.cos(2 * np.pi * y / 700.0)
        + 0.02 * x
    )


def _survey(line_errors: np.ndarray, alternate_direction: bool = True, n_per_line: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(len(line_errors)):
        x = np.full(n_per_line, i * SPACING)
        y = np.linspace(0.0, LINE_LENGTH, n_per_line)
        # Reverse every other line so the flight directions alternate the
        # way a real boustrophedon survey does.
        if alternate_direction and i % 2 == 1:
            y = y[::-1]
        rows.append(
            pd.DataFrame({
                "line_id": i,
                "x": x,
                "y": y,
                "anomaly": _geology(x, y) + line_errors[i],
            })
        )
    return pd.concat(rows, ignore_index=True)


def _striping_metric(df: pd.DataFrame, value_col: str = "anomaly") -> float:
    """Mean |second difference| of the per-line residual from the true
    geology, across lines - a direct measure of how much each line
    disagrees with its neighbours beyond a smooth trend."""
    per_line = []
    for lid in sorted(df["line_id"].unique()):
        sub = df[df["line_id"] == lid]
        residual = sub[value_col].to_numpy() - _geology(sub["x"].to_numpy(), sub["y"].to_numpy())
        per_line.append(float(np.mean(residual)))
    lv = np.array(per_line)
    return float(np.mean(np.abs(lv[2:] - 2 * lv[1:-1] + lv[:-2])))


# ---------------------------------------------------------------------------
# statistical leveling
# ---------------------------------------------------------------------------


def test_random_per_line_offsets_are_largely_removed():
    rng = np.random.default_rng(3)
    errors = rng.normal(0.0, 6.0, N_LINES)
    df = _survey(errors)

    before = _striping_metric(df)
    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING)
    df["leveled"] = apply_statistical_leveling(df, "anomaly", result)
    after = _striping_metric(df, "leveled")

    assert result.applied, result.reason
    assert after < 0.2 * before, f"striping {before:.2f} -> {after:.2f} nT is not enough of a reduction"
    # ...and the reported diagnostic must agree with reality, since that
    # number is what the user judges the correction by.
    assert result.roughness_after_nt < result.roughness_before_nt


def test_alternating_offsets_are_removed_where_a_single_heading_offset_cannot_be_the_answer():
    """Even a pure A/B alternation is handled - and unlike the heading
    correction, without being told the flight directions."""
    errors = np.where(np.arange(N_LINES) % 2 == 0, 4.0, -4.0)
    df = _survey(errors)

    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING)
    df["leveled"] = apply_statistical_leveling(df, "anomaly", result)

    assert _striping_metric(df, "leveled") < 0.25 * _striping_metric(df)


def test_smooth_geology_survives_when_there_is_no_leveling_error_to_find():
    """The correction must be nearly a no-op on clean data - a leveling
    step that always "finds" something would just be eating signal."""
    df = _survey(np.zeros(N_LINES))

    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING)
    df["leveled"] = apply_statistical_leveling(df, "anomaly", result)

    assert result.applied
    assert result.max_shift_nt < 1.0, f"shifted clean data by up to {result.max_shift_nt:.2f} nT"
    assert np.allclose(df["leveled"], df["anomaly"], atol=1.0)


def test_correction_preserves_the_survey_mean_level():
    """Leveling redistributes level between lines; it must not move the
    whole survey, or every absolute value downstream shifts with it."""
    rng = np.random.default_rng(11)
    df = _survey(rng.normal(0.0, 5.0, N_LINES))

    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING)
    shifts = np.array(list(result.line_shifts.values()))

    assert abs(float(np.mean(shifts))) < 1e-9


def test_a_too_wide_trend_window_invents_corrections_and_is_caught():
    """The knob's dangerous end. Past the span the geology stays quadratic
    over, the trend stops following the field and its own misfit is what
    gets subtracted - on data with no leveling error at all. The
    self-check has to catch that, because this correction runs by
    default."""
    df = _survey(np.zeros(30))  # no leveling error whatsoever

    ok = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=9)
    too_wide = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=27)

    assert ok.max_shift_nt < 2.0, "the default-ish window must barely touch clean data"
    assert too_wide.max_shift_nt > 10.0, "fixture no longer reproduces the failure this guards"
    assert not any("추세 창" in w for w in ok.warnings)
    assert any("추세 창" in w and "배로 커졌습니다" in w for w in too_wide.warnings)


def test_the_window_self_check_stays_quiet_when_the_error_is_real():
    """The same check must not cry wolf: where there is genuine per-line
    error, a wider window finds much the same correction, so widening is
    not evidence of misfit."""
    df = _survey(np.random.default_rng(3).normal(0.0, 6.0, 30))

    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=15)

    assert result.applied
    assert not any("배로 커졌습니다" in w for w in result.warnings)


def test_trend_window_below_the_supported_minimum_is_rejected():
    """A window too narrow for the local quadratic silently leaves most of
    the striping in place, so it is refused rather than accepted."""
    df = _survey(np.zeros(N_LINES))
    with pytest.raises(ValueError, match="이상"):
        compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=5)


def test_max_shift_clamp_limits_the_correction_and_says_so():
    errors = np.zeros(N_LINES)
    df = _survey(errors)
    df.loc[df["line_id"] == 5, "anomaly"] += 40.0

    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, max_shift_nt=5.0)

    assert max(abs(v) for v in result.line_shifts.values()) <= 5.0 + 1e-9
    assert any("상한" in w for w in result.warnings)


def test_order_1_removes_drift_along_a_single_line():
    """A line whose level ramps from one end to the other is drift, not a
    DC offset; order=0 can only take out its average."""
    df = _survey(np.zeros(N_LINES))
    target = df["line_id"] == 5
    # +/-8 nT ramp along the line.
    df.loc[target, "anomaly"] += (df.loc[target, "y"] / LINE_LENGTH - 0.5) * 16.0

    const = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, order=0)
    linear = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, order=1, n_segments=4)
    df["c"] = apply_statistical_leveling(df, "anomaly", const)
    df["l"] = apply_statistical_leveling(df, "anomaly", linear)

    def residual_range(col):
        sub = df[target]
        r = sub[col].to_numpy() - _geology(sub["x"].to_numpy(), sub["y"].to_numpy())
        return float(np.ptp(r))

    assert linear.line_shifts_linear is not None
    assert residual_range("l") < 0.6 * residual_range("c")


def test_too_few_lines_is_declined_rather_than_guessed_at():
    df = _survey(np.zeros(3))
    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING)

    assert not result.applied
    assert "측선" in result.reason
    assert apply_statistical_leveling(df, "anomaly", result).tolist() == df["anomaly"].tolist()


def test_non_overlapping_lines_are_reported_not_silently_chained():
    """Two blocks flown over different ground share no along-line extent,
    so their relative level is unmeasurable - the user has to be told."""
    df = _survey(np.zeros(N_LINES))
    # Push the second half of the lines far along-track, away from the first.
    df.loc[df["line_id"] >= 6, "y"] += 5 * LINE_LENGTH

    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING)

    assert result.applied
    assert result.n_pairs < N_LINES - 1
    assert any("겹치지" in w for w in result.warnings)


def test_invalid_parameters_are_rejected():
    df = _survey(np.zeros(N_LINES))
    with pytest.raises(ValueError):
        compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, order=2)


# ---------------------------------------------------------------------------
# local-plane heading correction
# ---------------------------------------------------------------------------


def test_local_plane_recovers_a_heading_offset_the_nearest_pair_method_misjudges():
    """The point of the local-plane fit: the cross-line geological
    gradient is modelled and removed, instead of leaking into the
    estimate as it does when two points a full line spacing apart are
    differenced."""
    true_offset = 6.0
    errors = np.where(np.arange(N_LINES) % 2 == 0, true_offset / 2.0, -true_offset / 2.0)
    df = _survey(errors)

    local = compute_heading_correction(df, "anomaly", AZIMUTH, SPACING, method="local_plane")
    nearest = compute_heading_correction(df, "anomaly", AZIMUTH, SPACING, method="nearest_pair")

    assert local.applied, local.reason
    # Group A is the even (forward-flown) lines, which carry +offset/2.
    assert local.offset_nt == pytest.approx(true_offset, abs=0.5)
    assert abs(local.offset_nt - true_offset) < abs(nearest.offset_nt - true_offset)


def test_local_plane_heading_correction_actually_flattens_the_alternation():
    true_offset = 6.0
    errors = np.where(np.arange(N_LINES) % 2 == 0, true_offset / 2.0, -true_offset / 2.0)
    df = _survey(errors)

    result = compute_heading_correction(df, "anomaly", AZIMUTH, SPACING, method="local_plane")
    df["leveled"] = apply_heading_correction(df, "anomaly", result)

    assert _striping_metric(df, "leveled") < 0.2 * _striping_metric(df)


def test_local_plane_reports_a_spread_that_flags_a_non_heading_error():
    """Per-line random error is not a heading effect. The estimate will
    come out near zero, but silence would be misleading - the spread has
    to say the model doesn't fit."""
    rng = np.random.default_rng(7)
    df = _survey(rng.normal(0.0, 8.0, N_LINES))

    result = compute_heading_correction(df, "anomaly", AZIMUTH, SPACING, method="local_plane")

    assert result.applied
    assert result.offset_spread_nt > abs(result.offset_nt)
    assert any("통계적 레벨링" in w for w in result.warnings)


def test_local_plane_declines_when_every_line_was_flown_the_same_way():
    df = _survey(np.zeros(N_LINES), alternate_direction=False)
    result = compute_heading_correction(df, "anomaly", AZIMUTH, SPACING, method="local_plane")

    assert not result.applied
    assert "반대 방향" in result.reason


def test_unknown_heading_method_is_rejected():
    df = _survey(np.zeros(N_LINES))
    with pytest.raises(ValueError):
        compute_heading_correction(df, "anomaly", AZIMUTH, SPACING, method="magic")


# ---------------------------------------------------------------------------
# micro-leveling modes
# ---------------------------------------------------------------------------


def _corrugated_grid(offsets: np.ndarray, cell: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """A gridded field with smooth geology plus one constant offset per
    line-spacing-wide column, i.e. exactly the corrugation pattern
    per-line level errors produce. Returns (clean, corrugated)."""
    ny, nx = 120, 120
    xs = np.arange(nx) * cell
    ys = np.arange(ny) * cell
    gx, gy = np.meshgrid(xs, ys)
    clean = 30.0 * np.sin(2 * np.pi * gx / 900.0) + 20.0 * np.cos(2 * np.pi * gy / 700.0)
    col = np.clip((gx / SPACING).astype(int), 0, len(offsets) - 1)
    return clean, clean + offsets[col]


def test_decorrugation_beats_the_narrow_notch_on_alternating_stripes():
    """The failure the mode exists to fix: A/B alternation puts its
    energy at two line spacings, outside a notch centred on one."""
    offsets = np.where(np.arange(40) % 2 == 0, 5.0, -5.0)
    clean, corrugated = _corrugated_grid(offsets)

    kwargs = dict(cell_size_m=10.0, line_azimuth_deg=AZIMUTH, line_spacing_m=SPACING, strength=1.0)
    notch = apply_microleveling(corrugated, mode="notch", **kwargs)
    deco = apply_microleveling(corrugated, mode="decorrugation", **kwargs)

    def err(a):  # interior only - FFT filters wrap at the edges
        return float(np.sqrt(np.mean((a - clean)[20:-20, 20:-20] ** 2)))

    assert err(deco) < err(notch)
    assert err(deco) < 0.5 * err(corrugated)


def test_decorrugation_leaves_long_wavelength_geology_alone():
    clean, _corrugated = _corrugated_grid(np.zeros(40))
    out = apply_microleveling(
        clean, cell_size_m=10.0, line_azimuth_deg=AZIMUTH, line_spacing_m=SPACING,
        strength=1.0, mode="decorrugation",
    )
    interior = (slice(20, -20), slice(20, -20))
    assert float(np.sqrt(np.mean((out - clean)[interior] ** 2))) < 0.1 * float(np.std(clean[interior]))


def test_unknown_microlevel_mode_is_rejected():
    with pytest.raises(ValueError):
        apply_microleveling(np.zeros((32, 32)), 10.0, AZIMUTH, SPACING, mode="magic")


# ---------------------------------------------------------------------------
# through the API, on the real sample survey
# ---------------------------------------------------------------------------


def _processed(**overrides):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    for endpoint, path in (("drone", DRONE_CSV), ("base", BASE_CSV)):
        with open(path, "rb") as f:
            client.post(f"/api/projects/{pid}/upload/{endpoint}", files={"files": ("f.csv", f, "text/csv")})
    body = {"line_params": {}, "diurnal_params": {}, "heading_correction": {}}
    body.update(overrides)
    r = client.post(f"/api/projects/{pid}/process", json=body)
    assert r.status_code == 200, r.text
    return client, pid, r.json()


def test_statistical_leveling_is_on_by_default():
    _client, _pid, summary = _processed()
    assert summary["statistical_leveling"] is not None
    # The sample survey has only 4 lines, so the default-on step declines
    # here - see the no-op test below.
    assert "비활성화" not in (summary["statistical_leveling"]["reason"] or "")


def test_default_on_leveling_is_a_true_no_op_on_a_survey_too_small_for_it():
    """Being on by default only stays safe if declining really means
    "changed nothing" - not "applied something small"."""
    on = _processed()[2]
    off = _processed(statistical_leveling={"enabled": False})[2]

    assert on["statistical_leveling"]["applied"] is False
    assert "측선" in on["statistical_leveling"]["reason"]
    for key in ("mean", "std", "min", "max"):
        assert on["anomaly_stats"][key] == pytest.approx(off["anomaly_stats"][key], rel=1e-12)


def test_striping_is_measured_even_when_the_correction_declines():
    """The severity number is what makes two systems comparable, so it has
    to be there on surveys too small to correct."""
    _client, _pid, summary = _processed()
    st = summary["striping"]

    assert st["available"], st.get("reason")
    assert st["line_level_jitter_nt"] > 0
    assert st["stripe_ratio_pct"] is not None


def test_heading_correction_defaults_to_the_local_plane_method():
    _client, _pid, summary = _processed(heading_correction={"enabled": True})
    info = summary["heading_correction"]

    assert info["applied"], info["reason"]
    assert info["method"] == "local_plane"
    # The spread over neighbourhoods is what tells the user whether the
    # single number means anything, so it has to be reported.
    assert info["offset_spread_nt"] is not None


def test_saved_project_replays_the_leveling_settings_it_was_processed_with():
    client, pid, summary = _processed(statistical_leveling={"enabled": True, "trend_window_lines": 11})
    blob = client.get(f"/api/projects/{pid}/save").content

    new_pid = client.post("/api/projects").json()["project_id"]
    restored = client.post(
        f"/api/projects/{new_pid}/load", files={"file": ("p.zip", blob, "application/zip")}
    ).json()["process_summary"]

    assert restored["statistical_leveling"] == summary["statistical_leveling"]
    assert restored["statistical_leveling"]["trend_window_lines"] == 11
    assert restored["striping"] == summary["striping"]


def test_a_survey_with_too_few_lines_for_the_window_is_declined_not_applied():
    """With fewer lines than the trend window, the window covers the whole
    survey and there is no scale separation left to make. Since this runs
    by default, the data must be left alone rather than corrected by a
    number nobody can check - and the reason has to say how to proceed."""
    df = _survey(np.zeros(8))
    result = compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=9)

    assert not result.applied
    assert "11개 이상 필요" in result.reason and "9개부터" in result.reason
    assert np.array_equal(apply_statistical_leveling(df, "anomaly", result), df["anomaly"].to_numpy())


def test_lowering_the_trend_window_brings_a_smaller_survey_into_range():
    """The escape hatch the decline message points at has to actually work."""
    df = _survey(np.zeros(9))

    assert not compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=9).applied
    assert compute_statistical_leveling(df, "anomaly", AZIMUTH, SPACING, trend_window_lines=7).applied


def test_the_direction_step_is_measured_even_with_the_correction_off():
    """The number people actually want when they see striping is "do the
    two flight directions read differently, and by how much". Getting it
    must not require turning a correction on and comparing two maps.

    heading_correction is on by default now (see models.py
    HeadingCorrectionParams), so "off" has to be requested explicitly here -
    the thing under test is the measured-but-not-applied state itself, not
    whatever the current default happens to be."""
    _client, _pid, summary = _processed(heading_correction={"enabled": False})
    info = summary["heading_correction"]

    assert info["measured"] is True, info["reason"]
    assert info["applied"] is False
    assert info["offset_nt"] is not None


def test_measuring_the_direction_step_does_not_change_the_data():
    off = _processed(heading_correction={"enabled": False})[2]
    on = _processed(heading_correction={"enabled": True})[2]

    assert on["heading_correction"]["applied"] is True
    assert off["heading_correction"]["offset_nt"] == pytest.approx(on["heading_correction"]["offset_nt"])
    # Same measurement, but only the enabled run may move the field.
    assert off["anomaly_stats"]["std"] != pytest.approx(on["anomaly_stats"]["std"], rel=1e-9)


def test_per_line_shifts_are_only_listed_when_they_were_really_applied():
    """The line list shows each line's applied shift; with the correction
    measured but off, reporting one would describe data that was never
    written."""
    off = _processed(heading_correction={"enabled": False})[2]
    on = _processed(heading_correction={"enabled": True})[2]

    assert all(line["heading_shift_nt"] in (None, 0.0) for line in off["lines"])
    assert any(line["heading_shift_nt"] for line in on["lines"])
    # The A/B grouping is still worth showing either way.
    assert {line["heading_group"] for line in off["lines"]} == {"A", "B"}
