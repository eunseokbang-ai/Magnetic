"""Normalised nth-difference noise QC channels, the standard airborne-
magnetics noise metric (Denisov et al., 2006) reiterated throughout the
Near-Surface Geophysics Inter-Society Committee's UAV magnetics guidelines
(2022) as the main post-flight-path-cleaning QC check: neighbouring
samples of a potential field cannot vary randomly, so a large nth
difference between adjacent samples has to come from instrument/manoeuvre
noise rather than geology.

The 4th difference is the classic measure; the guidelines also recommend
the 8th difference alongside it specifically for drone surveys, since
their lower flying height means more genuine short-wavelength geological
signal survives into the 4th difference than for conventional (higher,
faster) airborne surveys - going to the 8th difference suppresses more of
that geological content while still keeping short-period instrument/
manoeuvre noise clearly visible.

Both are "normalised" by dividing the raw nth-difference by the L2 norm of
its binomial coefficients (sqrt(70) for n=4, sqrt(12870) for n=8), so that
for pure white noise of standard deviation sigma the normalised difference
itself has standard deviation sigma - letting it be read directly as a
noise-level estimate in nT, not just a relative indicator.

Guidelines note the raw threshold has to scale with sampling rate (a 1 kHz
system shows roughly 10x the 4th-difference noise level of a 10 Hz system
recording the same real noise) - reference_threshold_nt below reflects
that scaling but is informational only, not a hard pass/fail gate, since
the "right" threshold also depends on sensor/platform quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# sqrt of the sum-of-squares of the nth central-difference binomial
# coefficients ([1,-4,6,-4,1] for n=4, sum of squares 70; the n=8 row sums
# to 12870) - see module docstring.
_NORM_4TH = float(np.sqrt(70.0))
_NORM_8TH = float(np.sqrt(12870.0))

# Guideline-cited reference for a conventional 10 Hz airborne system: 0.1
# nT normalised 4th difference. Scaled by sqrt(fs/10) per the guideline's
# empirical "1 kHz has ~10x the 4th-difference noise of 10 Hz" note.
_REFERENCE_4TH_DIFF_NT_AT_10HZ = 0.1


def _nth_difference_rms(values: np.ndarray, n: int) -> tuple[float, float]:
    """RMS and max-abs of the normalised nth difference of a single line's
    (already time-ordered) values. Returns (nan, nan) if too short."""
    diff = values.astype(float)
    for _ in range(n):
        diff = np.diff(diff)
    if diff.size == 0:
        return float("nan"), float("nan")
    norm = _NORM_4TH if n == 4 else _NORM_8TH
    normalised = diff / norm
    finite = normalised[np.isfinite(normalised)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(finite**2))), float(np.max(np.abs(finite)))


def compute_difference_qc(df: pd.DataFrame, value_col: str, line_id_col: str = "line_id") -> dict:
    """Per-line and overall normalised 4th/8th difference noise QC,
    computed on `value_col` (a production magnetics channel, e.g. the
    low-pass filtered but not yet diurnal/IGRF/leveling-corrected data -
    matching where the guidelines place this check in the workflow, right
    after flight-path cleaning) restricted to points already assigned to a
    flight line (line_id >= 0).

    A no-op (available=False) when there isn't enough data to compute
    anything meaningful."""
    lines_out = []
    for lid, group in df[df[line_id_col] >= 0].groupby(line_id_col):
        ordered = group.sort_values("timestamp")
        values = ordered[value_col].to_numpy(dtype=float)
        if len(values) < 10:
            continue
        rms4, max4 = _nth_difference_rms(values, 4)
        rms8, max8 = _nth_difference_rms(values, 8)
        if not np.isfinite(rms4):
            continue
        lines_out.append(
            {
                "line_id": int(lid),
                "n_points": len(values),
                "rms_4th_diff_nt": rms4,
                "max_4th_diff_nt": max4,
                "rms_8th_diff_nt": rms8,
                "max_8th_diff_nt": max8,
            }
        )

    if not lines_out:
        return {"available": False}

    dt = df["timestamp"].diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    sample_rate_hz = float(1.0 / dt.median()) if not dt.empty else None
    reference_threshold_nt = (
        _REFERENCE_4TH_DIFF_NT_AT_10HZ * float(np.sqrt(sample_rate_hz / 10.0))
        if sample_rate_hz and sample_rate_hz > 0
        else None
    )

    all_rms4 = np.array([l["rms_4th_diff_nt"] for l in lines_out])
    median_rms4 = float(np.median(all_rms4))
    # Relative outlier flag rather than a fixed absolute threshold: the
    # "right" absolute noise level varies hugely by sensor/sampling rate
    # (see reference_threshold_nt above, which is only a rough guide), but
    # a line running several times noisier than the survey's own other
    # lines is a reliable indicator something specific to that line (wind
    # gust, manoeuvre, obstacle) went wrong, regardless of the absolute
    # scale.
    for l in lines_out:
        l["flagged"] = bool(median_rms4 > 0 and l["rms_4th_diff_nt"] > 3.0 * median_rms4)

    overall_rms4 = float(np.sqrt(np.mean(all_rms4**2)))
    overall_rms8 = float(np.sqrt(np.mean(np.array([l["rms_8th_diff_nt"] for l in lines_out]) ** 2)))

    return {
        "available": True,
        "sample_rate_hz": sample_rate_hz,
        "reference_threshold_4th_diff_nt": reference_threshold_nt,
        "overall_rms_4th_diff_nt": overall_rms4,
        "overall_rms_8th_diff_nt": overall_rms8,
        "n_lines_flagged": sum(1 for l in lines_out if l["flagged"]),
        "lines": lines_out,
    }
