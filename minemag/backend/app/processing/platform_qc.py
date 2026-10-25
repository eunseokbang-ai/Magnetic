"""Platform-noise QC for a two-probe MagDrone survey: how much of what
the probes read is the carrier (drone, robot) rather than the ground.

Why this exists. On the KIGAM test site (2026-04) the same targets were
read from a drone at 0.5 m and from a quadruped robot, twice. The drone's
background was 7 nT rms; the robot's was 30 nT rms, a quasi-periodic
+-100 nT swing with a 0.55 m spatial period - its own gait (about 2 Hz at
1.1 m/s), seen in the spectrum at 1.5-2 Hz with a 4.5 Hz harmonic. The
two probes, 1 m apart, saw that noise only half-correlated (0.5-0.8), so
the horizontal gradient did not cancel it; it is a hardware problem (boom
length, magnetic parts), not a processing one. Whether a new data set has
that problem is what this module measures, so the operator learns it at
load time, not after a day of picking targets out of noise.

What is measured, per file and per probe, on the *quiet* samples (every
probe within `quiet_threshold_nt` of its median, so targets are excluded):

- full-band rms, and rms above 1 Hz and 5 Hz (what the along-track
  filters could and could not remove; at walking speed 1 Hz is a 1 m
  wavelength, the size of a mine's anomaly);
- the correlation of the two probes' high-passed noise and the rms of
  their difference: the gradient cancels only what is common to both;
- the amplitude spectrum of the longest quiet stretch, its strongest
  peaks, and the spatial period each implies at the median speed.

Thresholds come from that test site: a probe correlation under 0.9 means
the gradient will not clean it; a peak between 1 and 3.5 Hz on a walking
or flying platform is a gait or rotor period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

DEFAULT_QUIET_THRESHOLD_NT = 60.0
# Below this, the probe difference does not remove the noise.
GRADIENT_CANCELS_ABOVE_CORR = 0.9
# A spectral peak in this band is the carrier's periodic motion.
GAIT_BAND_HZ = (0.8, 6.0)   # the stride itself (about 2 Hz on the test robot) and its first harmonics
_HIGHPASS_HZ = (1.0, 5.0)
_MIN_QUIET_RUN = 64          # samples needed for a spectrum
_SPECTRUM_MAX_HZ = 20.0
_SPECTRUM_POINTS = 160       # returned to the frontend, decimated


def _rms(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if len(a) else float("nan")


def _highpass(v: np.ndarray, fc: float, fs: float) -> np.ndarray | None:
    if fs <= 2.5 * fc or len(v) < 30:
        return None
    b, a = butter(2, fc / (fs / 2.0), "high")
    return filtfilt(b, a, v)


def _round(v, d=2):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def _file_qc(sub: pd.DataFrame, quiet_threshold_nt: float) -> dict:
    sensors = sorted(int(s) for s in sub["sensor"].unique())
    primary = sensors[0]
    piv = sub.pivot_table(index="t_ms", columns="sensor", values="tfa", aggfunc="first").sort_index()
    piv = piv.dropna()
    t = piv.index.to_numpy(dtype=float)
    if len(t) < 10:
        return {"error": "표본이 너무 적습니다."}
    dt = np.diff(t)
    dt_ms = float(np.median(dt))
    fs = 1000.0 / dt_ms if dt_ms > 0 else float("nan")

    centered = {s: piv[s].to_numpy() - np.median(piv[s].to_numpy()) for s in sensors}
    # "Quiet" has to exclude targets but keep the carrier's own noise, or a
    # noisy platform would leave no quiet stretch to measure. So the
    # threshold is the larger of the given value and four robust standard
    # deviations of the readings themselves (1.4826 x MAD): a 7 nT drone
    # keeps the 60 nT bound, a +-100 nT gait swing raises it to ~250 nT.
    robust = max(1.4826 * float(np.median(np.abs(c))) for c in centered.values())
    threshold = max(quiet_threshold_nt, 4.0 * robust)
    quiet = np.ones(len(t), dtype=bool)
    for s in sensors:
        quiet &= np.abs(centered[s]) < threshold

    # speed of the primary probe, excluding gaps
    p = sub[sub["sensor"] == primary].sort_values("t_ms")
    pt = p["t_ms"].to_numpy(dtype=float)
    pdt = np.diff(pt)
    step = np.hypot(np.diff(p["x"].to_numpy()), np.diff(p["y"].to_numpy()))
    ok = (pdt > 0) & (pdt <= 1000.0)
    speed = float(np.median(step[ok] / (pdt[ok] / 1000.0))) if ok.any() else float("nan")

    per_probe = {}
    hp = {}
    for s in sensors:
        v = centered[s]
        h1 = _highpass(v, _HIGHPASS_HZ[0], fs)
        h5 = _highpass(v, _HIGHPASS_HZ[1], fs)
        hp[s] = h1
        per_probe[str(s)] = {
            "quiet_rms_nt": _round(_rms(v[quiet])),
            "quiet_rms_above_1hz_nt": _round(_rms(h1[quiet])) if h1 is not None else None,
            "quiet_rms_above_5hz_nt": _round(_rms(h5[quiet])) if h5 is not None else None,
        }

    gradient = None
    if len(sensors) >= 2 and hp[sensors[0]] is not None and hp[sensors[1]] is not None and quiet.sum() > 30:
        a, b = hp[sensors[0]][quiet], hp[sensors[1]][quiet]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        diff_rms = _rms(a - b)
        single = 0.5 * (_rms(a) + _rms(b))
        gradient = {
            "probe_correlation_above_1hz": _round(corr, 3),
            "difference_rms_nt": _round(diff_rms),
            "single_probe_rms_nt": _round(single),
            "cancellation_pct": _round(100.0 * (1.0 - diff_rms / single), 0) if single > 0 else None,
            "cancels": bool(np.isfinite(corr) and corr >= GRADIENT_CANCELS_ABOVE_CORR),
        }

    # Spectrum of the longest quiet, regularly sampled stretch of the primary probe.
    spectrum = None
    regular = np.r_[True, np.abs(dt - dt_ms) < 0.5 * dt_ms]
    usable = quiet & regular
    idx = np.flatnonzero(usable)
    if len(idx):
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        if len(run) >= _MIN_QUIET_RUN and np.isfinite(fs):
            seg = centered[primary][run]
            seg = seg - seg.mean()
            f, pxx = welch(seg, fs=fs, nperseg=min(1024, len(seg)))
            amp = np.sqrt(pxx)
            keep = (f > 0) & (f <= _SPECTRUM_MAX_HZ)
            f, amp = f[keep], amp[keep]
            peaks = []
            if len(f) > 8:
                pk, _props = find_peaks(np.log10(amp + 1e-12), prominence=0.3)
                pk = sorted((i for i in pk if f[i] >= 0.3), key=lambda i: -amp[i])[:3]
                for i in pk:
                    peaks.append({
                        "hz": _round(f[i], 2),
                        "amplitude": _round(amp[i], 3),
                        "spatial_period_m": _round(speed / f[i], 2) if np.isfinite(speed) and f[i] > 0 else None,
                        "in_gait_band": bool(GAIT_BAND_HZ[0] <= f[i] <= GAIT_BAND_HZ[1]),
                    })
            stride = max(1, len(f) // _SPECTRUM_POINTS)
            spectrum = {
                "hz": np.round(f[::stride], 3).tolist(),
                "amplitude": np.round(amp[::stride], 4).tolist(),
                "run_seconds": _round(len(run) / fs, 1),
                "peaks": peaks,
            }

    verdict = _verdict(sensors, per_probe, gradient, spectrum, speed)
    return {
        "sensors": sensors,
        "sample_rate_hz": _round(fs, 1),
        "n_paired_samples": int(len(t)),
        "quiet_threshold_used_nt": _round(threshold, 1),
        "quiet_fraction": _round(float(quiet.mean()), 3),
        "speed_m_s": _round(speed),
        "per_probe": per_probe,
        "gradient": gradient,
        "spectrum": spectrum,
        "verdict": verdict,
    }


def _verdict(sensors, per_probe, gradient, spectrum, speed) -> list[str]:
    lines = []
    worst = max((p["quiet_rms_nt"] or 0.0) for p in per_probe.values())
    if worst < 10:
        lines.append(f"배경 잡음 {worst:.1f} nT rms: 조용한 편입니다(시험장 드론 0.5 m 비행 7 nT 수준).")
    elif worst < 20:
        lines.append(f"배경 잡음 {worst:.1f} nT rms: 100 nT 아래의 작은 표적은 신중히 봐야 합니다.")
    else:
        lines.append(f"배경 잡음 {worst:.1f} nT rms: 큽니다. 300 nT 이상의 큰 표적만 확실히 보입니다(시험장 로봇 50 nT 사례).")
    if spectrum and spectrum["peaks"]:
        top = spectrum["peaks"][0]
        periodic = [p for p in spectrum["peaks"] if p["in_gait_band"]]
        if periodic:
            desc = ", ".join(f"{p['hz']} Hz" + (f"({p['spatial_period_m']} m)" if p["spatial_period_m"] else "") for p in periodic)
            lines.append(f"주기성 잡음 {desc}: 플랫폼의 주기 운동(보행 주기와 그 배음, 회전익)으로 보입니다.")
        else:
            lines.append(f"주 잡음 주파수 {top['hz']} Hz.")
    if gradient:
        c = gradient["probe_correlation_above_1hz"]
        if gradient["cancels"]:
            lines.append(f"두 탐침 잡음 상관 {c}: 수평 구배(탐침 차)가 잡음을 {gradient['cancellation_pct']}% 줄입니다.")
        elif worst >= 20:
            lines.append(
                f"두 탐침 잡음 상관 {c}: 수평 구배로는 잡음이 상쇄되지 않습니다(차의 rms {gradient['difference_rms_nt']} nT). "
                "처리보다 센서 거치(붐 길이, 비자성 부품)를 먼저 볼 문제입니다."
            )
        else:
            lines.append(f"두 탐침 잡음 상관 {c}: 구배로 상쇄되지는 않지만 잡음 자체가 작아 문제되지 않습니다.")
    return lines


def platform_qc(df: pd.DataFrame, quiet_threshold_nt: float = DEFAULT_QUIET_THRESHOLD_NT) -> dict:
    """QC per file of a loaded set (see io_/magdrone_r3.py for the frame)."""
    files = []
    infos = {i: f for i, f in enumerate(df.attrs.get("files") or [])}
    for fi, sub in df.groupby("file_index"):
        r = _file_qc(sub, quiet_threshold_nt)
        r["file_index"] = int(fi)
        r["filename"] = infos.get(int(fi), {}).get("filename")
        files.append(r)
    return {
        "quiet_threshold_nt": quiet_threshold_nt,
        "gradient_cancels_above_corr": GRADIENT_CANCELS_ABOVE_CORR,
        "gait_band_hz": list(GAIT_BAND_HZ),
        "files": files,
    }
