"""Pre-processing of the two-probe readings before anything is gridded:
the workflow of Bang, Jeong & Yim (2026, KSCE J. Civ. Environ. Eng.
Res. 46(4), 377-394), section 2.6, as it was run for that paper in
DroneMagAdv, now a module of its own.

1. Resample. 200 Hz is far more than a target needs: at a walking speed
   of 1 m/s, 10 Hz still puts ten readings on every metre, enough for an
   anti-personnel mine whose anomaly is about a metre wide. Linear
   interpolation onto a uniform time axis, per file and per probe, never
   across a gap longer than `gap_s`.
2. Adaptive Hampel despike (Pearson 2002): a sample more than
   `hampel_k` (3.0) MADs from its rolling median is replaced by the
   median; over a steep stretch - directly above a mine - the window
   widens so the anomaly's own flank is not taken for a spike. This is
   magcore.despike with the paper's constants, run after resampling as
   in the paper (a pass at the logger's 200 Hz was tried and clipped the
   anomaly itself).
3. Zero-phase Gaussian low-pass at `lpf_hz`. The carrier's periodic
   noise sits above 2 Hz (the quadruped's stride at 4.2 Hz forward,
   2.59 Hz backward, and harmonics) while a mine's anomaly, 1-1.5 m
   wide at 1 m/s, is 0.7-1 Hz; the paper measured that 2 Hz keeps the
   anomaly's shape and loses little amplitude, while 1 Hz took 25-38% of
   the peak and distorted it. A rotor drone's noise is higher and
   smaller, so 5 Hz there. A Gaussian kernel is symmetric, hence zero
   phase: the anomaly does not move along the track.
4. Path editing: inside each DataTool track, samples whose heading is
   more than `turn_angle_deg` off the track's own direction, or slower
   than `min_speed_m_s`, are the turn or the wait at the end and are
   excluded (line_id -1). Everything else is a survey line.
5. Heading correction: the forward/backward level step between
   alternate lines, measured with magcore.leveling's local-plane method
   on the quiet readings and applied only when the step is larger than
   its own spread - the same measure-always-apply-when-trusted rule as
   the mineral program.

Measured on the synthetic of tests/test_preprocess.py (two probes, 1 m/s,
a 1 m-wide 300 nT anomaly plus 80 nT of 4.2 Hz gait noise and 0.3 s
bursts): the 2 Hz Gaussian passes 4.2 Hz at 0.22, so the
fourth-difference noise drops about 4x, while the anomaly's peak stays
within 10% and does not move; the bursts are gone; the turn samples are
excluded. The reported noise is the fourth-difference rms at the
resampled rate, before and after, so the ratio is what the filter did.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from magcore.despike import despike
from magcore.leveling import apply_heading_correction, compute_heading_correction

from .gridding import dominant_azimuth_deg, line_spacing_m

DEFAULT_SAMPLES_PER_M = 10.0
DEFAULT_HAMPEL_K = 3.0
DEFAULT_HAMPEL_WINDOW = 11
DEFAULT_LPF_HZ = 2.0            # walking robot; a rotor drone takes 5 Hz
DEFAULT_TURN_ANGLE_DEG = 30.0
DEFAULT_MIN_SPEED_M_S = 0.3
DEFAULT_GAP_S = 1.0
# -3 dB cutoff of a Gaussian kernel: H(f) = exp(-2 pi^2 sigma^2 f^2) = 1/sqrt(2)
_GAUSS_SIGMA_PER_HZ = float(np.sqrt(np.log(2.0)) / (2.0 * np.pi))   # sigma [s] = this / fc


@dataclass
class PreprocessParams:
    resample_hz: float | None = None          # None = samples_per_m x the run's median speed
    samples_per_m: float = DEFAULT_SAMPLES_PER_M
    hampel_k: float = DEFAULT_HAMPEL_K
    hampel_window: int = DEFAULT_HAMPEL_WINDOW      # samples, at the working (resampled) rate
    lpf_hz: float = DEFAULT_LPF_HZ
    lpf_hz_by_file: dict = field(default_factory=dict)
    turn_angle_deg: float = DEFAULT_TURN_ANGLE_DEG
    min_speed_m_s: float = DEFAULT_MIN_SPEED_M_S
    heading_correction: bool = True
    gap_s: float = DEFAULT_GAP_S


_INTERP_COLS = ("lat", "lon", "x", "y", "alt", "tfa", "bx", "by", "bz")


def gaussian_lowpass(values: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Gaussian low-pass with its -3 dB point at cutoff_hz."""
    sigma_samples = _GAUSS_SIGMA_PER_HZ / cutoff_hz * fs_hz
    if sigma_samples < 0.3:
        return np.asarray(values, dtype=float).copy()
    return gaussian_filter1d(np.asarray(values, dtype=float), sigma_samples, mode="nearest")


def _resample_probe(g: pd.DataFrame, hz: float, gap_s: float) -> pd.DataFrame:
    """One probe of one file onto a uniform axis, segment by segment."""
    g = g.sort_values("t_ms")
    t = g["t_ms"].to_numpy(dtype=float) / 1000.0
    pieces = []
    breaks = np.flatnonzero(np.diff(t) > gap_s)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(t)]
    step = 1.0 / hz
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        ts, te = t[s], t[e - 1]
        new_t = np.arange(ts, te + 1e-9, step)
        seg = {"t_s": new_t}
        for c in _INTERP_COLS:
            if c in g.columns:
                v = g[c].to_numpy(dtype=float)[s:e]
                seg[c] = np.interp(new_t, t[s:e], v) if np.isfinite(v).all() else np.full(len(new_t), np.nan)
        # nearest track id
        tr = g["track"].to_numpy()[s:e]
        idx = np.searchsorted(t[s:e], new_t).clip(0, e - s - 1)
        seg["track"] = tr[idx]
        pieces.append(pd.DataFrame(seg))
    if not pieces:
        return pd.DataFrame(columns=["t_s", *_INTERP_COLS, "track"])
    return pd.concat(pieces, ignore_index=True)


def _segment_lines(p: pd.DataFrame, turn_angle_deg: float, min_speed: float) -> np.ndarray:
    """line_id per sample: the track id where the sample moves along the
    track's own direction at survey speed, -1 in the turns and waits."""
    line_id = np.full(len(p), -1, dtype=int)
    x, y, t = p["x"].to_numpy(), p["y"].to_numpy(), p["t_s"].to_numpy()
    for tr, idx in p.groupby("track").indices.items():
        if len(idx) < 5:
            continue
        xi, yi, ti = x[idx], y[idx], t[idx]
        dx, dy, dt = np.gradient(xi), np.gradient(yi), np.gradient(ti)
        dt = np.where(dt > 0, dt, np.nan)
        speed = np.hypot(dx, dy) / dt
        heading = np.degrees(np.arctan2(dx, dy)) % 180.0
        # the track's direction from its end points
        main = np.degrees(np.arctan2(xi[-1] - xi[0], yi[-1] - yi[0])) % 180.0
        dev = np.abs(((heading - main) + 90.0) % 180.0 - 90.0)
        keep = (dev <= turn_angle_deg) & (speed >= min_speed)
        line_id[idx[keep]] = int(tr)
    return line_id


def _fourth_diff_rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    if len(v) < 10:
        return float("nan")
    d4 = v[4:] - 4 * v[3:-1] + 6 * v[2:-2] - 4 * v[1:-3] + v[:-4]
    return float(np.sqrt(np.mean(d4 ** 2)) / np.sqrt(70.0))


def preprocess(df: pd.DataFrame, params: PreprocessParams | None = None) -> tuple[pd.DataFrame, dict]:
    """The loaded frame (io_/magdrone_r3.py, with x/y) -> a resampled,
    despiked, filtered, line-tagged frame with `tfa_proc`, and a report
    per file."""
    params = params or PreprocessParams()
    out_frames, reports = [], []
    infos = {i: f for i, f in enumerate(df.attrs.get("files") or [])}
    for fi, sub in df.groupby("file_index"):
        fi = int(fi)
        # speed of the first probe, for the automatic resampling rate
        s1 = sub[sub["sensor"] == sub["sensor"].min()].sort_values("t_ms")
        dt = np.diff(s1["t_ms"].to_numpy(dtype=float)) / 1000.0
        step = np.hypot(np.diff(s1["x"].to_numpy()), np.diff(s1["y"].to_numpy()))
        ok = (dt > 0) & (dt <= params.gap_s)
        speed = float(np.median(step[ok] / dt[ok])) if ok.any() else 1.0
        hz = float(params.resample_hz) if params.resample_hz else max(2.0, round(params.samples_per_m * speed, 1))
        fc = float(params.lpf_hz_by_file.get(fi, params.lpf_hz_by_file.get(str(fi), params.lpf_hz)))

        probes = []
        n_spikes = 0
        noise_before, noise_after = [], []
        for s, g in sub.groupby("sensor"):
            p = _resample_probe(g, hz, params.gap_s)
            if p.empty:
                continue
            raw = p["tfa"].to_numpy(dtype=float)
            noise_before.append(_fourth_diff_rms(raw))
            # The paper's order: resample, then Hampel at the working rate.
            # A pass at the logger's 200 Hz was tried first and clipped the
            # anomaly itself (its flank is steeper than the gait swing at
            # that rate), so it stays out.
            cleaned, mask = despike(raw, window_size=params.hampel_window, threshold_k=params.hampel_k, adaptive=True)
            n_spikes += int(mask.sum())
            # filter inside each contiguous segment so a gap is not smeared
            filt = cleaned.copy()
            t = p["t_s"].to_numpy()
            breaks = np.flatnonzero(np.diff(t) > params.gap_s)
            for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks + 1, len(t)]):
                filt[a:b] = gaussian_lowpass(cleaned[a:b], hz, fc)
            noise_after.append(_fourth_diff_rms(filt))
            p["sensor"] = int(s)
            p["tfa_raw"] = raw
            p["tfa_proc"] = filt
            p["spike"] = mask
            probes.append(p)
        if not probes:
            continue
        pf = pd.concat(probes, ignore_index=True)
        pf["file_index"] = fi
        pf["line_id"] = _segment_lines(pf, params.turn_angle_deg, params.min_speed_m_s)
        pf["timestamp"] = pd.Timestamp(sub["timestamp"].min().normalize()) + pd.to_timedelta(pf["t_s"], unit="s")
        pf["t_ms"] = pf["t_s"] * 1000.0

        # Heading correction per sensor: two sensors on one carrier have
        # their own orientation effects (Gongju J1 -2 nT, J2 +9 nT between
        # the two flight directions), so one offset for the pair would
        # under-correct one and over-correct the other.
        heading = {"enabled": params.heading_correction, "applied": False, "offset_nt": None, "reason": None, "per_sensor": {}}
        lines = pf[pf["line_id"] >= 0]
        spacing = None
        if lines["line_id"].nunique() >= 2:
            dominant = dominant_azimuth_deg(lines)
            spacing = line_spacing_m(lines, dominant)
        if params.heading_correction:
            if lines["line_id"].nunique() >= 3:
                for s in sorted(pf["sensor"].unique()):
                    ps = pf[pf["sensor"] == s]
                    res = compute_heading_correction(ps, "tfa_proc", dominant, spacing or 0.5)
                    offset = float(res.offset_nt) if res.offset_nt is not None else None
                    spread = float(res.offset_spread_nt) if res.offset_spread_nt is not None else None
                    # Trusted only when the step is larger than its own spread
                    # over the neighbourhoods - otherwise it is line noise, not
                    # a heading effect (the mineral program's rule).
                    trusted = res.applied and offset is not None and (spread is None or spread < abs(offset))
                    entry = {
                        "applied": bool(trusted),
                        "offset_nt": round(offset, 2) if offset is not None else None,
                        "offset_spread_nt": round(spread, 2) if spread is not None else None,
                        "reason": res.reason if not res.applied else (
                            None if trusted else "구역별 오프셋의 산포가 오프셋보다 커서 헤딩 효과로 볼 수 없어 적용하지 않았습니다."
                        ),
                    }
                    if trusted:
                        pf.loc[ps.index, "tfa_proc"] = apply_heading_correction(ps, "tfa_proc", res)
                    heading["per_sensor"][str(int(s))] = entry
                applied = [e for e in heading["per_sensor"].values() if e["applied"]]
                heading["applied"] = bool(applied)
                first = heading["per_sensor"].get(str(int(sorted(pf["sensor"].unique())[0])), {})
                heading["offset_nt"] = first.get("offset_nt")
                heading["offset_spread_nt"] = first.get("offset_spread_nt")
                if not applied:
                    heading["reason"] = next((e["reason"] for e in heading["per_sensor"].values() if e["reason"]), None)
            else:
                heading["reason"] = "측선이 3개 미만이라 헤딩 보정을 계산하지 않았습니다."

        reports.append({
            "file_index": fi,
            "filename": infos.get(fi, {}).get("filename"),
            "speed_m_s": round(speed, 2),
            "resample_hz": hz,
            "n_in": int(len(sub)),
            "n_out": int(len(pf)),
            "n_spikes": n_spikes,
            "lpf_hz": fc,
            "noise_before_nt": round(float(np.nanmean(noise_before)), 3),
            "noise_after_nt": round(float(np.nanmean(noise_after)), 3),
            "n_lines": int(pf.loc[pf["line_id"] >= 0, "line_id"].nunique()),
            "turn_fraction": round(float((pf["line_id"] < 0).mean()), 3),
            "line_spacing_m": round(spacing, 2) if spacing is not None else None,
            "heading_correction": heading,
        })
        out_frames.append(pf)
    if not out_frames:
        raise ValueError("전처리할 자료가 없습니다.")
    proc = pd.concat(out_frames, ignore_index=True)
    proc = proc.sort_values(["timestamp", "sensor"], kind="stable").reset_index(drop=True)
    proc["point_id"] = np.arange(len(proc))
    proc["tfa"] = proc["tfa_proc"]            # downstream modules read `tfa`
    proc.attrs["files"] = df.attrs.get("files")
    proc.attrs["preprocess"] = {"params": params.__dict__ | {"lpf_hz_by_file": dict(params.lpf_hz_by_file)}, "files": reports}
    return proc, proc.attrs["preprocess"]
