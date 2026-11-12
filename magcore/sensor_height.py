"""Correcting the sensor's GPS height against the ground it stood on.

A towed magnetometer's own GPS gives its height to within several metres,
and the error is not noise: it is an offset that wanders slowly. Measured
on the 2026-11 GaSaDo survey (MagArrow, 41 flights, a 0.5 m DSM): with the
sensor lying on the ground before takeoff and after landing, its GPS
height was above the ground by a median of 9.7 m, anywhere from -10 to
+28 m, and it changed by 9.4 m rms between the takeoff and the landing of
one flight while agreeing to about a metre across the minute between one
flight's landing and the next one's takeoff. An inversion places its
sources by the distance from the sensor to the ground, so ten metres at a
flying height of fifty is not a detail (about 40 % in the model's
amplitude, see the manual, 5.6).

Those moments on the ground are the correction. Each flight gives two
"ground checks": a time, and the GPS height minus the terrain model's
elevation there - the error, known to the accuracy of the terrain model.
Between and around the checks the error is estimated as what it most
probably was, given how it is seen to behave:

    a stationary process in time with mean mu, variance sill and
    covariance  sill * exp(-|dt| / tau)   (Ornstein-Uhlenbeck),

predicted by simple kriging from the checks, which also says how much is
left unknown at every moment. On GaSaDo the variogram of the 82 checks
gives mu 9.7 m, sill 84 m2 (9.2 m), tau 27 min, and no nugget to speak
of. Tested by leaving checks out:

    predicting a check from the others,      rms error
    leaving out everything within            nothing   mean only   kriged
      0 min (nearest ~1 min away)            12.8 m    8.4 m       3.9 m
      5-15 min                               12.8 m    8.4 m       7.1 m
      30 min                                 12.8 m    8.4 m       8.7 m

So the correction is worth most where it is anchored: a few metres near
takeoff and landing, about 6 m in the middle of a 23-minute flight (the
kriging's own estimate there, 5.8 m), against 13 m uncorrected. Most of
the gain is the mean. What is left in mid-flight cannot be had from the
ground; a height from the aircraft's own log could do better.

Heights here are orthometric (above mean sea level): the receiver's MSL
altitude against a terrain model in the same sense, which is what DEMs
normally are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The error model used when a survey has too few ground checks to fit its
# own: the GaSaDo values. A different receiver will differ; the summary
# says which was used.
DEFAULT_SILL_M2 = 84.0
DEFAULT_TAU_MIN = 27.0
# Variance of one check about the true error: the terrain model at the
# pad, and the sensor not lying exactly on it.
CHECK_VARIANCE_M2 = 0.5
# Fewer checks than this and the variogram is not fitted.
MIN_CHECKS_TO_FIT = 20
# A recording that starts or ends in the air shows a "check" tens of
# metres off. Anything this far from the median of the checks is not one.
OUTLIER_M = 25.0
# What counts as standing on the ground, and for how long at least.
GROUND_MAX_SPEED_MPS = 0.5
GROUND_MAX_RISE_M = 1.0
GROUND_MIN_SECONDS = 1.0


@dataclass
class GroundCheck:
    flight: int                 # source file index
    kind: str                   # "takeoff" | "landing"
    time_s: float               # seconds, same clock as the samples
    gps_height_m: float         # median MSL altitude while on the ground
    ground_m: float             # terrain model there (NaN when it has none)
    error_m: float              # gps - ground - resting offset
    seconds: float              # how long the sensor was seen standing
    lat: float = float("nan")
    lon: float = float("nan")
    used: bool = True
    excluded_reason: str | None = None

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "GroundCheck":
        d = dict(d)
        for key in ("ground_m", "error_m", "lat", "lon", "gps_height_m"):
            if d.get(key) is None:
                d[key] = float("nan")
        return cls(**d)


@dataclass
class ErrorModel:
    mean_m: float
    sill_m2: float
    tau_min: float
    fitted: bool                # False: defaults, too few checks to fit
    n_checks: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SensorHeightCorrection:
    """The checks a correction rests on and the model between them. Enough
    to recompute the correction for any sample time, without the terrain
    model it was measured against."""
    checks: list[GroundCheck]
    model: ErrorModel
    resting_offset_m: float = 0.0
    stats: dict = field(default_factory=dict)

    def predict(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return predict_error(time_s, self.checks, self.model)

    def to_dict(self) -> dict:
        return {"checks": [c.to_dict() for c in self.checks], "model": self.model.to_dict(),
                "resting_offset_m": self.resting_offset_m, "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict) -> "SensorHeightCorrection":
        return cls(checks=[GroundCheck.from_dict(c) for c in d["checks"]], model=ErrorModel(**d["model"]),
                   resting_offset_m=float(d.get("resting_offset_m", 0.0)), stats=dict(d.get("stats") or {}))


def _standing(alt: np.ndarray, speed: np.ndarray, time_s: np.ndarray, from_start: bool) -> slice | None:
    """The stretch at one end of a flight's record during which the sensor
    had not left (or had come back to) the ground: slow, and within
    GROUND_MAX_RISE_M of where it started (ended)."""
    n = len(alt)
    if n < 5:
        return None
    order = np.arange(n) if from_start else np.arange(n)[::-1]
    ref = np.nanmedian(alt[order[: min(50, n)]])
    if not np.isfinite(ref):
        return None
    moved = (np.abs(alt[order] - ref) > GROUND_MAX_RISE_M) | (np.nan_to_num(speed[order], nan=0.0) > GROUND_MAX_SPEED_MPS) | ~np.isfinite(alt[order])
    k = int(np.argmax(moved)) if moved.any() else n
    if k < 2:
        return None
    idx = order[:k]
    if abs(time_s[idx[-1]] - time_s[idx[0]]) < GROUND_MIN_SECONDS:
        return None
    return slice(int(idx.min()), int(idx.max()) + 1)


def find_ground_checks(
    time_s: np.ndarray, altitude_msl_m: np.ndarray, speed_mps: np.ndarray, flight: np.ndarray,
    ground_m: np.ndarray, lat: np.ndarray | None = None, lon: np.ndarray | None = None,
    resting_offset_m: float = 0.0, exclude_sea_level: bool = True,
) -> list[GroundCheck]:
    """The takeoff and landing checks of every flight.

    ground_m is the terrain model sampled at every sample's position (NaN
    where it has no value). resting_offset_m is how far above the ground
    the sensor's GPS antenna sits when it is "on the ground" (0 if it lies
    on it). A check is kept out of the correction - but still listed, with
    the reason - when the terrain model has no value there, when it says
    sea level (many terrain models mask the coast to 0, and a pad on a
    quay then reads metres wrong; exclude_sea_level), or when it is so far
    from the others that the record must have started or ended in the air.
    """
    time_s = np.asarray(time_s, dtype=float)
    alt = np.asarray(altitude_msl_m, dtype=float)
    speed = np.asarray(speed_mps, dtype=float)
    flight = np.asarray(flight)
    ground_m = np.asarray(ground_m, dtype=float)
    checks: list[GroundCheck] = []
    for f in np.unique(flight):
        rows = np.flatnonzero(flight == f)
        rows = rows[np.argsort(time_s[rows], kind="stable")]
        for kind, from_start in (("takeoff", True), ("landing", False)):
            span = _standing(alt[rows], speed[rows], time_s[rows], from_start)
            if span is None:
                continue
            idx = rows[span]
            ground = ground_m[idx]
            ground_value = float(np.nanmedian(ground)) if np.isfinite(ground).any() else float("nan")
            gps = float(np.nanmedian(alt[idx]))
            check = GroundCheck(
                flight=int(f), kind=kind, time_s=float(np.median(time_s[idx])), gps_height_m=gps, ground_m=ground_value,
                error_m=gps - ground_value - resting_offset_m, seconds=float(time_s[idx[-1]] - time_s[idx[0]]),
                lat=float(np.nanmedian(lat[idx])) if lat is not None else float("nan"),
                lon=float(np.nanmedian(lon[idx])) if lon is not None else float("nan"),
            )
            if not np.isfinite(ground_value):
                check.used, check.excluded_reason = False, "no_terrain"
            elif exclude_sea_level and abs(ground_value) < 0.05:
                check.used, check.excluded_reason = False, "sea_level"
            checks.append(check)
    good = np.array([c.error_m for c in checks if c.used])
    if len(good) >= 4:
        centre = float(np.median(good))
        for c in checks:
            if c.used and abs(c.error_m - centre) > OUTLIER_M:
                c.used, c.excluded_reason = False, "outlier"
    return sorted(checks, key=lambda c: c.time_s)


def fit_error_model(checks: list[GroundCheck]) -> ErrorModel:
    """Mean, variance and correlation time of the error, from the checks
    in use. With MIN_CHECKS_TO_FIT or more the variogram is fitted
    (exponential, binned, weighted by pair count, pairs up to ten hours
    apart); with fewer, the mean is the checks' own and the rest are the
    defaults."""
    used = [c for c in checks if c.used]
    errors = np.array([c.error_m for c in used])
    times = np.array([c.time_s for c in used])
    n = len(used)
    if n == 0:
        return ErrorModel(mean_m=0.0, sill_m2=DEFAULT_SILL_M2, tau_min=DEFAULT_TAU_MIN, fitted=False, n_checks=0)
    mean = float(np.median(errors)) if n >= 3 else float(np.mean(errors))
    if n < MIN_CHECKS_TO_FIT:
        return ErrorModel(mean_m=mean, sill_m2=DEFAULT_SILL_M2, tau_min=DEFAULT_TAU_MIN, fitted=False, n_checks=n)

    i, j = np.triu_indices(n, 1)
    lag = np.abs(times[i] - times[j]) / 60.0
    semi = 0.5 * (errors[i] - errors[j]) ** 2
    edges = np.array([0, 2, 4, 6, 9, 13, 18, 24, 30, 40, 55, 75, 100, 140, 200, 300, 450, 600], dtype=float)
    h, g, w = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (lag >= lo) & (lag < hi)
        if k.sum() >= 3:
            h.append(lag[k].mean()); g.append(semi[k].mean()); w.append(k.sum())
    if len(h) < 4 or min(h) > 30:
        # nothing at short lags to hang a correlation time on
        return ErrorModel(mean_m=mean, sill_m2=float(max(np.var(errors), 1.0)), tau_min=DEFAULT_TAU_MIN, fitted=False, n_checks=n)
    h, g, w = np.array(h), np.array(g), np.sqrt(np.array(w, dtype=float))
    best = None
    for tau in np.geomspace(3.0, 240.0, 60):
        shape = 1.0 - np.exp(-h / tau)
        sill = float(np.sum(w * w * shape * g) / np.sum(w * w * shape * shape))     # least squares for the one linear parameter
        cost = float(np.sum((w * (sill * shape - g)) ** 2))
        if best is None or cost < best[0]:
            best = (cost, sill, float(tau))
    return ErrorModel(mean_m=mean, sill_m2=float(max(best[1], 1.0)), tau_min=best[2], fitted=True, n_checks=n)


def predict_error(time_s: np.ndarray, checks: list[GroundCheck], model: ErrorModel) -> tuple[np.ndarray, np.ndarray]:
    """The error at each time and its standard deviation: simple kriging
    from the checks in use. With none, the model's mean and full spread.

    Evaluated on a one-second grid and interpolated to the samples - the
    error moves over minutes, and a survey has hundreds of thousands of
    samples."""
    time_s = np.asarray(time_s, dtype=float)
    used = [c for c in checks if c.used]
    total_sd = float(np.sqrt(model.sill_m2))
    if len(time_s) == 0:
        return np.zeros(0), np.zeros(0)
    if not used:
        return np.full(len(time_s), model.mean_m), np.full(len(time_s), total_sd)
    ta = np.array([c.time_s for c in used])
    ea = np.array([c.error_m for c in used])
    tau_s = model.tau_min * 60.0
    C = model.sill_m2 * np.exp(-np.abs(ta[:, None] - ta[None, :]) / tau_s) + CHECK_VARIANCE_M2 * np.eye(len(ta))
    Cinv_e = np.linalg.solve(C, ea - model.mean_m)
    Cinv = np.linalg.inv(C)

    finite = np.isfinite(time_s)
    lo, hi = float(np.min(time_s[finite])), float(np.max(time_s[finite]))
    mean = np.full(len(time_s), model.mean_m)
    sd = np.full(len(time_s), total_sd)
    # only where a check is near enough to say anything (6 tau: under 0.3 % of the covariance left)
    reach = 6.0 * tau_s
    grid = np.arange(np.floor(lo), np.ceil(hi) + 1.0, 1.0)
    near = np.zeros(len(grid), dtype=bool)
    for t in ta:
        a, b = np.searchsorted(grid, [t - reach, t + reach])
        near[a:b] = True
    g_mean = np.full(len(grid), model.mean_m)
    g_sd = np.full(len(grid), total_sd)
    tg = grid[near]
    for start in range(0, len(tg), 20000):
        chunk = tg[start:start + 20000]
        c = model.sill_m2 * np.exp(-np.abs(chunk[:, None] - ta[None, :]) / tau_s)
        m = model.mean_m + c @ Cinv_e
        v = model.sill_m2 - np.einsum("ij,jk,ik->i", c, Cinv, c)
        where = np.flatnonzero(near)[start:start + 20000]
        g_mean[where] = m
        g_sd[where] = np.sqrt(np.maximum(v, 0.0))
    mean[finite] = np.interp(time_s[finite], grid, g_mean)
    sd[finite] = np.interp(time_s[finite], grid, g_sd)
    return mean, sd


def leave_out_validation(checks: list[GroundCheck], model: ErrorModel, exclude_within_min: float = 5.0) -> dict | None:
    """How well the correction predicts a check it was not given: each
    check in turn is predicted from the others, leaving out everything
    within exclude_within_min of it (so that a landing is not predicted
    from the takeoff a minute later). Three rms errors: of doing nothing,
    of removing the mean alone, of the kriging."""
    used = [c for c in checks if c.used]
    if len(used) < 6:
        return None
    errors = np.array([c.error_m for c in used])
    times = np.array([c.time_s for c in used])
    predicted = np.empty(len(used))
    for k in range(len(used)):
        others = [c for c in used if abs(c.time_s - times[k]) > exclude_within_min * 60.0]
        predicted[k] = predict_error(times[k:k + 1], others, model)[0][0]
    return {
        "exclude_within_min": exclude_within_min,
        "n": len(used),
        "rms_uncorrected_m": round(float(np.sqrt(np.mean(errors ** 2))), 2),
        "rms_mean_only_m": round(float(np.sqrt(np.mean((errors - model.mean_m) ** 2))), 2),
        "rms_kriged_m": round(float(np.sqrt(np.mean((errors - predicted) ** 2))), 2),
    }


def build_correction(
    time_s: np.ndarray, altitude_msl_m: np.ndarray, speed_mps: np.ndarray, flight: np.ndarray, ground_m: np.ndarray,
    lat: np.ndarray | None = None, lon: np.ndarray | None = None, resting_offset_m: float = 0.0, exclude_sea_level: bool = True,
) -> SensorHeightCorrection:
    checks = find_ground_checks(time_s, altitude_msl_m, speed_mps, flight, ground_m, lat, lon, resting_offset_m, exclude_sea_level)
    model = fit_error_model(checks)
    return SensorHeightCorrection(checks=checks, model=model, resting_offset_m=resting_offset_m)
