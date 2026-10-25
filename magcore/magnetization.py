"""Does this anomaly's source carry remanent magnetization?

The question matters at exactly one point in this app's workflow: an
operator is looking at a candidate (processing/anomaly_candidates.py) and
has to decide whether it is a ground structure to remove or geology to
keep. Rebar, guardrails, fence posts, pipes and vehicles are steel, and
steel magnetized by anything other than today's field - welding, cold
working, a lightning strike, the factory - carries a remanent moment that
points wherever it happens to point. Ordinary magnetite-bearing geology
at this scale is usually magnetized along the present field, because it
is induced. So a source whose magnetization is far from the IGRF
direction is evidence, not proof, of something man-made.

How it is measured here: fit the readings around the candidate with a
single dipole whose moment is free in all three components, and with the
same dipole constrained to the induced direction. Two numbers come out of
that pair.

- **The angle** between the fitted moment and the induced direction.
- **How much the freedom bought**: the drop in fit rms from the
  constrained fit to the free one. Without this the angle means nothing -
  a weak anomaly in noisy data will happily report 80 degrees because
  every direction fits equally badly.

Both are needed. Measured on synthetic dipoles 50 m below the sensor,
50 m line spacing, 1 nT noise and geology underneath (see
tests/test_magnetization.py): an induced source comes back 0.1 degrees
from the field direction and freeing the direction buys 0.4% of the rms,
while a source magnetized 109 degrees away is recovered to 109.1 degrees
with the free fit cutting the rms by 96% (28.6 -> 1.0 nT).

Two cases need care, and both are handled:

- **A moment along the field but reversed.** The constrained fit reaches
  that too, by taking a negative amplitude, and fits exactly as well - so
  the improvement is zero and only the angle shows it. A negative
  amplitude is a negative susceptibility, which rocks do not have, so
  that fit is itself the evidence of remanence.
- **An anomaly lost in the noise.** Every direction fits it equally, and
  the free fit picks one: a 0.8 nT source under 3 nT of noise came back
  as "remanent" at 134 degrees before the fitted dipole's own peak was
  required to stand above the residual.

What this is not: a Koenigsberger ratio. That needs the body's volume and
susceptibility, neither of which a total-field survey knows. This is the
direction of the *total* magnetization, which is what the data can say.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .source_removal import _background_columns, _field_direction, _kernel

# Radius of the fitting window, as a multiple of the source depth, floored
# at a multiple of the line spacing so a shallow source still gets several
# lines. Wider than this and the plane background stops describing the
# geology; narrower and the lobe - which is what carries the direction
# information - falls outside.
_WINDOW_DEPTHS = 3.0
_WINDOW_LINE_SPACINGS = 1.5
_MIN_POINTS = 40
_MAX_FIT_POINTS = 4000
# Position and depth are both uncertain (the depth comes from the analytic
# signal's width, the position from its peak), and an error in either
# leaks into the direction. Both are searched over a small grid and the
# best fit kept.
_OFFSET_STEPS = (-0.5, -0.25, 0.0, 0.25, 0.5)
_DEPTH_FACTORS = (0.7, 1.0, 1.4)
# Thresholds for the verdict. Measured in tests/test_magnetization.py:
# induced sources recover within ~10 deg, and freeing the direction on an
# induced source buys ~0-10% of rms, while a genuinely remanent one buys
# half or more.
_INDUCED_ANGLE_DEG = 25.0
_REMANENT_ANGLE_DEG = 40.0
_MIN_IMPROVEMENT = 0.25
# How far the fitted dipole's own field has to stand above what the fit
# could not explain before any direction it reports means anything. A
# 0.8 nT anomaly under 3 nT of noise fits every direction equally, and
# without this guard such a source came back "remanent" on the strength
# of pure noise.
_MIN_MODEL_SNR = 3.0


@dataclass
class MagnetizationEstimate:
    inclination_deg: float          # of the fitted total magnetization
    declination_deg: float
    angle_from_induced_deg: float
    free_fit_rms_nt: float
    induced_fit_rms_nt: float
    improvement: float              # 1 - free/induced, i.e. what the freedom bought
    depth_m: float
    n_points: int
    model_peak_nt: float            # the fitted dipole's own peak in the window
    verdict: str                    # "induced" | "remanent" | "unclear"

    def to_dict(self) -> dict:
        return {
            "inclination_deg": round(self.inclination_deg, 1),
            "declination_deg": round(self.declination_deg, 1),
            "angle_from_induced_deg": round(self.angle_from_induced_deg, 1),
            "free_fit_rms_nt": round(self.free_fit_rms_nt, 2),
            "induced_fit_rms_nt": round(self.induced_fit_rms_nt, 2),
            "improvement": round(self.improvement, 3),
            "depth_m": round(self.depth_m, 1),
            "n_points": self.n_points,
            "model_peak_nt": round(self.model_peak_nt, 1),
            "verdict": self.verdict,
            "label": {
                "induced": "유도자화에 가까움",
                "remanent": "잔류자화 강함",
                "unclear": "판단 보류",
            }[self.verdict],
        }


def _direction_angles(moment: np.ndarray) -> tuple[float, float]:
    """(inclination, declination) in degrees of a moment vector, in the
    same convention as source_removal._field_direction: x east, y north,
    z up, so a downward (positive-inclination) field has negative z."""
    norm = float(np.linalg.norm(moment))
    if norm == 0.0:
        return 0.0, 0.0
    inc = float(np.degrees(np.arcsin(-moment[2] / norm)))
    dec = float(np.degrees(np.arctan2(moment[0], moment[1])))
    return inc, dec


def _fit(A: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, float]:
    sol, *_ = np.linalg.lstsq(A, d, rcond=None)
    return sol, float(np.sqrt(np.mean((d - A @ sol) ** 2)))


def estimate_magnetization(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cx: float,
    cy: float,
    depth_m: float,
    inclination_deg: float,
    declination_deg: float,
    line_spacing_m: float | None = None,
) -> MagnetizationEstimate | None:
    """Direction of the magnetization of whatever sits at (cx, cy) and
    `depth_m` below the survey plane, or None when there are too few
    readings around it to ask.

    `values` are the survey's anomaly values at `x`, `y` - the same
    samples the removal works on, not the grid, because the grid between
    lines is the interpolator's invention and would pull the direction
    toward whatever the interpolator did there.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    depth_m = float(max(depth_m, 1.0))

    radius = max(_WINDOW_DEPTHS * depth_m, _WINDOW_LINE_SPACINGS * (line_spacing_m or 50.0))
    near = np.isfinite(values) & (np.hypot(x - cx, y - cy) <= radius)
    if int(near.sum()) < _MIN_POINTS:
        return None
    wx, wy, wv = x[near], y[near], values[near]
    stride = max(1, int(np.ceil(len(wx) / _MAX_FIT_POINTS)))
    wx, wy, wv = wx[::stride], wy[::stride], wv[::stride]

    f = _field_direction(inclination_deg, declination_deg)
    bg = _background_columns(wx, wy, cx, cy, radius)

    best = None
    for depth in (max(1.0, depth_m * k) for k in _DEPTH_FACTORS):
        for dx in _OFFSET_STEPS:
            for dy in _OFFSET_STEPS:
                source = np.array([[cx + dx * depth, cy + dy * depth, -depth]])
                g = _kernel(wx, wy, source, f)          # (n, 3), one per moment component
                free_sol, free_rms = _fit(np.hstack([g, bg]), wv)
                if best is None or free_rms < best[0]:
                    induced = (g @ f)[:, None]          # the same dipole, moment along F
                    induced_sol, induced_rms = _fit(np.hstack([induced, bg]), wv)
                    best = (free_rms, induced_rms, free_sol[:3], depth, float(induced_sol[0]), g)

    free_rms, induced_rms, moment, depth, induced_strength, g = best
    model_peak = float(np.max(np.abs(g @ moment)))
    inc, dec = _direction_angles(moment)
    unit = moment / max(float(np.linalg.norm(moment)), 1e-30)
    angle = float(np.degrees(np.arccos(np.clip(float(np.dot(unit, f)), -1.0, 1.0))))
    improvement = float(1.0 - free_rms / induced_rms) if induced_rms > 0 else 0.0

    # A moment pointing *against* the field is the one case the
    # improvement cannot see: the constrained fit reaches it too, by
    # taking a negative amplitude, and fits exactly as well. That
    # amplitude is a negative susceptibility, which rocks do not have, so
    # an anti-parallel fit is itself the evidence.
    reversed_fit = induced_strength < 0.0
    if model_peak < _MIN_MODEL_SNR * free_rms:
        # Nothing here stands above the noise; whatever direction the fit
        # settled on is a description of that noise.
        verdict = "unclear"
    elif angle <= _INDUCED_ANGLE_DEG:
        verdict = "induced"
    elif angle >= _REMANENT_ANGLE_DEG and (improvement >= _MIN_IMPROVEMENT or reversed_fit):
        verdict = "remanent"
    else:
        # A large angle that buys nothing is not a measurement of
        # anything - say so rather than calling it either way.
        verdict = "unclear"

    return MagnetizationEstimate(
        inclination_deg=inc,
        declination_deg=dec,
        angle_from_induced_deg=angle,
        free_fit_rms_nt=free_rms,
        induced_fit_rms_nt=induced_rms,
        improvement=improvement,
        depth_m=float(depth),
        n_points=int(len(wv)),
        model_peak_nt=model_peak,
        verdict=verdict,
    )
