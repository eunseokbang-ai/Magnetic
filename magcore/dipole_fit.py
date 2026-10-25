"""Compact near-surface magnetic target detection: automatically flags
localized anomalies distinct from broad geological trends (buried
ordnance, landmines, hidden vehicles - anything with a small ferrous
footprint) and fits each one to a single induced-magnetic-dipole model to
estimate its location, depth, and relative ferrous mass.

Physics: for a small object magnetized by induction along the ambient
field direction (moment aligned with f_hat, ignoring remanence and
self-demagnetization - the same "induced-only" simplification already
used by processing/inversion.py), the total-field anomaly at an
observation point offset by vector r from the dipole is the classic
magnetic dipole formula:

    delta_T(r) = (mu0 * |m| / (4*pi*|r|^3)) * (3*(f_hat . r_hat)^2 - 1)

This is the standard near-surface UXO/ordnance "dipole fit" model (see
e.g. Breiner 1999; the discrimination literature around magnetometer-only
UXO surveys) - much better suited to a single compact metallic object
than the smooth-body assumptions behind the 3D voxel inversion, and a
tighter, more localized fit than Euler deconvolution's sliding-window
structural-index search.

As with Euler deconvolution (processing/euler_deconvolution.py), the
observation surface is treated as flat at z=0; a solved depth_m is
therefore depth below that flat survey plane, not literal ground burial
depth. The fitted moment gives only a *relative* ferrous-mass size class
- magnetics alone cannot identify a target's type (mine vs scrap vs
UXO vs vehicle), and minimal-metal mines may produce no usable anomaly
at all. These caveats are surfaced in the summary returned to callers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

MU0 = 4e-7 * np.pi  # T*m/A

# Coarse, uncalibrated order-of-magnitude dipole-moment buckets drawn from
# published UXO/ordnance magnetometry literature. These are NOT a
# validated classifier - only a rough size hint to sort candidates by.
SIZE_CLASSES = [
    (1.0, "소형 (파편류/소형 금속물체 수준)"),
    (10.0, "중형 (개인화기/소형 포탄 수준)"),
    (100.0, "대형 (대형 포탄/드럼통 수준)"),
    (1000.0, "초대형 (차량급)"),
    (float("inf"), "최대형 (전차/대형 장비급)"),
]


def classify_moment(moment_am2: float) -> str:
    for upper, label in SIZE_CLASSES:
        if moment_am2 < upper:
            return label
    return SIZE_CLASSES[-1][1]


@dataclass
class TargetCandidate:
    x: float
    y: float
    depth_m: float
    moment_am2: float
    background_nt: float
    peak_anomaly_nt: float
    footprint_m: float
    rms_residual_nt: float
    fit_quality: float  # 0..1, 1 = residual negligible relative to signal amplitude
    n_points: int


def field_direction_enu(inclination_deg: float, declination_deg: float) -> np.ndarray:
    """Unit vector of the ambient field in local East-North-Up."""
    incl = np.radians(inclination_deg)
    decl = np.radians(declination_deg)
    horiz = np.cos(incl)
    fe = horiz * np.sin(decl)
    fn = horiz * np.cos(decl)
    fu = -np.sin(incl)  # inclination positive = field points down = negative "up"
    return np.array([fe, fn, fu])


def _find_candidates(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    threshold_nt: float,
    min_footprint_m: float,
    max_footprint_m: float,
) -> tuple[list[dict], float]:
    """Locate compact local-maximum anomalies and estimate each one's
    footprint from its half-amplitude width, not from how far a fixed
    absolute threshold happens to extend. A footprint measured at a fixed
    nT threshold scales with the target's amplitude (a large target's
    footprint balloons well past any reasonable size cap even though the
    source itself is still compact), so it can't distinguish "small
    object, strong signal" from "broad regional trend" - half-amplitude
    width is the standard normalization that does. A broad monotonic
    regional trend also has no local maximum at all, so it's excluded
    here regardless of its absolute amplitude."""
    finite = np.isfinite(grid_values)
    if finite.sum() < 4:
        return [], 0.0
    background = float(np.nanmedian(grid_values))
    residual = grid_values - background
    abs_residual = np.where(finite, np.abs(residual), -np.inf)

    footprint_px = max(3, int(round(max_footprint_m / cell_size_m)))
    if footprint_px % 2 == 0:
        footprint_px += 1
    local_max = ndimage.maximum_filter(abs_residual, size=footprint_px, mode="nearest")
    peak_mask = finite & (abs_residual >= local_max) & (abs_residual >= threshold_nt)

    n_rows, n_cols = grid_values.shape
    half_win_px = max(1, footprint_px // 2)
    candidates = []
    for r, c in zip(*np.nonzero(peak_mask)):
        peak_val = float(residual[r, c])
        half_amp = abs(peak_val) / 2.0
        r0, r1 = max(0, r - half_win_px), min(n_rows, r + half_win_px + 1)
        c0, c1 = max(0, c - half_win_px), min(n_cols, c + half_win_px + 1)
        window = residual[r0:r1, c0:c1]
        same_sign_above_half = finite[r0:r1, c0:c1] & (np.abs(window) >= half_amp) & (np.sign(window) == np.sign(peak_val))
        labeled, _ = ndimage.label(same_sign_above_half)
        peak_label = labeled[r - r0, c - c0]
        n_px = int((labeled == peak_label).sum())
        area_m2 = n_px * cell_size_m**2
        equiv_diam_m = 2.0 * np.sqrt(area_m2 / np.pi)
        if equiv_diam_m < min_footprint_m or equiv_diam_m > max_footprint_m:
            continue
        candidates.append(
            {
                "x": float(easting[c]),
                "y": float(northing[r]),
                "peak_anomaly_nt": peak_val,
                "footprint_m": float(equiv_diam_m),
            }
        )
    return candidates, background


def _fit_dipole(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    f_hat: np.ndarray,
    x0_guess: float,
    y0_guess: float,
    depth_guess: float,
    max_depth_m: float,
    search_radius_m: float,
) -> dict | None:
    peak_amp = float(np.max(np.abs(values - np.median(values))))
    if peak_amp <= 0:
        return None
    bg_guess = float(np.median(values))
    # analytic dipole formula, solved for |m| at the peak location assuming
    # the field-alignment factor (3*dot^2 - 1) is near its max magnitude (2)
    m_guess = max(abs(peak_amp) * 1e-9 * 4 * np.pi * depth_guess**3 / (MU0 * 2.0), 1e-4)
    log_m_guess = np.log(m_guess)

    def residual(params):
        x0, y0, depth, log_m, bg = params
        m = np.exp(log_m)
        rx = x - x0
        ry = y - y0
        rz = np.full_like(rx, depth)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        r = np.maximum(r, 0.05)
        dot = (f_hat[0] * rx + f_hat[1] * ry + f_hat[2] * rz) / r
        model_t = (MU0 * m / (4 * np.pi * r**3)) * (3 * dot**2 - 1)
        return model_t * 1e9 + bg - values

    lower = [x0_guess - search_radius_m, y0_guess - search_radius_m, 0.05, log_m_guess - 12, bg_guess - 10 * abs(peak_amp) - 10]
    upper = [x0_guess + search_radius_m, y0_guess + search_radius_m, max_depth_m, log_m_guess + 12, bg_guess + 10 * abs(peak_amp) + 10]
    x_init = [x0_guess, y0_guess, min(depth_guess, 0.9 * max_depth_m), log_m_guess, bg_guess]

    try:
        result = least_squares(
            residual, x_init, bounds=(lower, upper), loss="soft_l1", f_scale=max(1.0, 0.1 * abs(peak_amp))
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not result.success:
        return None

    x0, y0, depth, log_m, bg = result.x
    rms = float(np.sqrt(np.mean(result.fun**2)))
    quality = float(np.clip(1.0 - rms / max(abs(peak_amp), 1e-6), 0.0, 1.0))
    return {
        "x": float(x0),
        "y": float(y0),
        "depth_m": float(depth),
        "moment_am2": float(np.exp(log_m)),
        "background_nt": float(bg),
        "rms_residual_nt": rms,
        "fit_quality": quality,
    }


def detect_targets(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    inclination_deg: float,
    declination_deg: float,
    threshold_nt: float,
    min_footprint_m: float,
    max_footprint_m: float,
    fit_window_m: float,
    max_depth_m: float,
    min_fit_quality: float,
) -> list[TargetCandidate]:
    """Find compact anomalies in the gridded field and fit each to a
    single induced-dipole model using the nearby raw (ungridded) survey
    points, which resolves the target's location/depth more precisely
    than the coarse detection grid alone."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    candidates, _background = _find_candidates(
        grid_values, easting, northing, cell_size_m, threshold_nt, min_footprint_m, max_footprint_m
    )
    if not candidates:
        return []

    f_hat = field_direction_enu(inclination_deg, declination_deg)
    tree = cKDTree(np.column_stack([x, y]))

    results: list[TargetCandidate] = []
    for cand in candidates:
        idx = tree.query_ball_point([cand["x"], cand["y"]], r=fit_window_m)
        if len(idx) < 6:
            continue
        idx = np.asarray(idx)
        depth_guess = float(np.clip(cand["footprint_m"] / 1.3, 0.3, 0.7 * max_depth_m))
        fit = _fit_dipole(
            x[idx], y[idx], values[idx], f_hat, cand["x"], cand["y"], depth_guess, max_depth_m, fit_window_m
        )
        if fit is None or fit["fit_quality"] < min_fit_quality:
            continue
        results.append(
            TargetCandidate(
                x=fit["x"],
                y=fit["y"],
                depth_m=fit["depth_m"],
                moment_am2=fit["moment_am2"],
                background_nt=fit["background_nt"],
                peak_anomaly_nt=cand["peak_anomaly_nt"],
                footprint_m=cand["footprint_m"],
                rms_residual_nt=fit["rms_residual_nt"],
                fit_quality=fit["fit_quality"],
                n_points=len(idx),
            )
        )

    # Multiple nearby candidate blobs can converge to essentially the same
    # fitted target; keep the better-quality fit and drop close duplicates.
    results.sort(key=lambda t: -t.fit_quality)
    deduped: list[TargetCandidate] = []
    min_sep_m = max(0.5 * fit_window_m, 1.0)
    for t in results:
        if any(np.hypot(t.x - d.x, t.y - d.y) < min_sep_m for d in deduped):
            continue
        deduped.append(t)

    return deduped
