"""What two sensors on one carrier say about each other.

A pair mounted a few centimetres apart (the Log_Drone optically pumped
J1/J2, set at different orientations to tame the heading error) or 1 m
apart (the MagDrone R3 probes) reads the same field up to a fixed offset.
Everything else in their difference is system error - the orientation
(heading) effect of each sensor, attitude-coupled error, and noise - and
unlike the field it can be measured without knowing the ground. This
module reports, per file:

- the pair offset and how much the difference scatters when still and
  when flying (the still figure is the floor; the flying one is what
  the carrier adds);
- the direction effect of each sensor: its mean reading (less the file
  median) on the lines flown one way against the lines flown the other
  way, on background samples - the same quantity the pre-processing's
  per-sensor heading correction then removes (Gongju: J1 -2 / J2 +9 nT
  here, corrections +1.5 / -10.8 nT there). This is the heading error that the pre-processing's per-sensor
  heading correction removes, and tells which sensor has the smaller one;
- when roll and pitch were logged, how much of each sensor's background
  variation a linear fit on them explains - the gain an attitude
  compensation would give.

Measured on the Gongju grid flight (2026-10-19, Log_Drone J1/J2, nose
fixed, parallel lines at 163 deg): pair offset -52 nT, difference scatter
0.3 nT on the ground and 8.4 nT in flight; the difference correlates
-0.97 with pitch, but pitch itself is 6.8 deg on the south-bound lines and
1.5 deg on the north-bound ones, i.e. it is the flight direction. The
direction effect is -2 nT for J1 and +9 nT for J2; an attitude fit per
sensor returns 2.37 -> 2.36 nT (J1) and 3.50 -> 3.09 nT (J2), so attitude
compensation is not worth a step here and the per-sensor heading
correction is what matters. J1 is the cleaner sensor (background 2.4 vs
3.5 nT); an inverse-variance mean of the two gains 0.07 nT over J1 alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BACKGROUND_NT = 15.0        # |high-passed mean of the pair| below this is background
MIN_SPEED_M_S = 0.3         # moving / still split
ATTITUDE_WORTH_NT = 1.0     # an attitude fit must remove at least this much to be recommended


def _r(v, d=2):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def _hp(v: np.ndarray, window: int) -> np.ndarray:
    return v - pd.Series(v).rolling(max(5, window), center=True, min_periods=1).median().to_numpy()


def _direction_masks(sub1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float | None]:
    """forward / backward masks of the primary sensor's rows from the
    tracks' signed azimuths (the dominant axis is the median mod 180)."""
    n = len(sub1)
    fwd, back = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    col = "line_id" if "line_id" in sub1.columns else "track"
    if col not in sub1.columns:
        return fwd, back, None
    ids = sub1[col].to_numpy()
    x, y = sub1["x"].to_numpy(), sub1["y"].to_numpy()
    az = {}
    for k in np.unique(ids[ids >= 0]):
        idx = np.flatnonzero(ids == k)
        if len(idx) < 5:
            continue
        az[k] = float(np.degrees(np.arctan2(x[idx[-1]] - x[idx[0]], y[idx[-1]] - y[idx[0]])) % 360.0)
    if len(az) < 2:
        return fwd, back, None
    # doubled-angle mean, so 2 deg and 178 deg lines do not average to 90
    a2 = np.radians(2.0 * np.array(list(az.values())))
    dominant = float(np.degrees(0.5 * np.arctan2(np.mean(np.sin(a2)), np.mean(np.cos(a2)))) % 180.0)
    for k, a in az.items():
        dev = abs(((a - dominant) + 180.0) % 360.0 - 180.0)     # 0..180 against the dominant axis
        if dev <= 45.0:
            fwd[ids == k] = True
        elif dev >= 135.0:
            back[ids == k] = True                                # cross lines stay in neither group
    return fwd, back, dominant


def sensor_pair_report(sub: pd.DataFrame) -> dict | None:
    """One file's rows (long format with x, y) -> the pair report, or
    None with fewer than two sensors or too few paired samples."""
    sensors = sorted(int(s) for s in sub["sensor"].unique())
    if len(sensors) < 2:
        return None
    s1, s2 = sensors[0], sensors[1]
    cols = ["tfa", "x", "y"] + [c for c in ("roll", "pitch") if c in sub.columns]
    piv = sub.pivot_table(index="t_ms", columns="sensor", values=cols, aggfunc="first").sort_index()
    try:
        v1, v2 = piv[("tfa", s1)].to_numpy(dtype=float), piv[("tfa", s2)].to_numpy(dtype=float)
    except KeyError:
        return None
    ok = np.isfinite(v1) & np.isfinite(v2)
    if ok.sum() < 50:
        return None
    piv = piv[ok]
    v1, v2 = v1[ok], v2[ok]
    t = piv.index.to_numpy(dtype=float) / 1000.0
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    win = int(round(2.0 / max(dt, 1e-3)))
    x, y = piv[("x", s1)].to_numpy(dtype=float), piv[("y", s1)].to_numpy(dtype=float)
    step = np.hypot(np.gradient(x), np.gradient(y))
    speed = step / np.where(np.gradient(t) > 0, np.gradient(t), np.nan)
    moving = np.nan_to_num(speed) > MIN_SPEED_M_S
    still = ~moving

    d = v2 - v1
    mean = (v1 + v2) / 2.0
    bg = (np.abs(_hp(mean, win)) < BACKGROUND_NT) & moving
    hp = {s1: _hp(v1, win), s2: _hp(v2, win)}
    out = {
        "sensors": [s1, s2],
        "n_paired": int(len(t)),
        "offset_nt": _r(np.median(d)),
        "difference_std_still_nt": _r(np.std(d[still])) if still.sum() >= 50 else None,
        "difference_std_flight_nt": _r(np.std(d[bg])) if bg.sum() >= 50 else None,
        "background_std_nt": {str(s): _r(np.std(hp[s][bg])) for s in (s1, s2)} if bg.sum() >= 50 else None,
    }

    # direction effect per sensor
    sub1 = sub[sub["sensor"] == s1].sort_values("t_ms")
    sub1 = sub1[np.isin(sub1["t_ms"].to_numpy(dtype=float) / 1000.0, t)] if len(sub1) != len(t) else sub1
    direction = None
    if len(sub1) == len(t):
        fwd, back, dominant = _direction_masks(sub1.reset_index(drop=True))
        f, b = fwd & bg, back & bg
        if f.sum() >= 50 and b.sum() >= 50:
            direction = {
                "dominant_azimuth_deg": _r(dominant, 0),
                "n_forward": int(f.sum()), "n_backward": int(b.sum()),
                # on the readings less the file median - a 2 s high-pass would
                # remove the very line-level offset this is measuring
                "forward_minus_backward_nt": {str(s): _r(np.mean(v[f] - np.median(v)) - np.mean(v[b] - np.median(v)))
                                              for s, v in ((s1, v1), (s2, v2))},
                "pair_difference_change_nt": _r(d[f].mean() - d[b].mean()),
            }
    out["direction"] = direction

    # attitude coupling, when roll and pitch were logged
    attitude = None
    if ("roll", s1) in piv.columns and ("pitch", s1) in piv.columns:
        roll, pitch = piv[("roll", s1)].to_numpy(dtype=float), piv[("pitch", s1)].to_numpy(dtype=float)
        m = bg & np.isfinite(roll) & np.isfinite(pitch)
        if m.sum() >= 100 and np.std(pitch[m]) > 0.1:
            X = np.column_stack([np.ones(m.sum()), roll[m] - roll[m].mean(), pitch[m] - pitch[m].mean()])
            per = {}
            for s in (s1, s2):
                beta, *_ = np.linalg.lstsq(X, hp[s][m], rcond=None)
                res = hp[s][m] - X @ beta
                per[str(s)] = {"roll_nt_per_deg": _r(beta[1]), "pitch_nt_per_deg": _r(beta[2]),
                               "std_before_nt": _r(np.std(hp[s][m])), "std_after_nt": _r(np.std(res))}
            attitude = {
                "corr_difference_pitch": _r(np.corrcoef(d[m], pitch[m])[0, 1]),
                "corr_difference_roll": _r(np.corrcoef(d[m], roll[m])[0, 1]),
                "pitch_mean_deg": _r(pitch[m].mean(), 1), "pitch_std_deg": _r(pitch[m].std(), 1),
                "per_sensor": per,
                "worth_it": any(p["std_before_nt"] - p["std_after_nt"] >= ATTITUDE_WORTH_NT for p in per.values()),
            }
    out["attitude"] = attitude
    out["verdict"] = _verdict(out, s1, s2)
    return out


def _verdict(r: dict, s1: int, s2: int) -> list[str]:
    lines = []
    still, flight = r["difference_std_still_nt"], r["difference_std_flight_nt"]
    if flight is not None:
        lines.append(f"두 센서 차의 산포: 비행 중 {flight} nT" + (f", 정지 시 {still} nT" if still is not None else "")
                     + " - 정지값이 센서 잡음의 바닥, 비행값에서 늘어난 만큼이 운반체·방향에 따른 시스템 오차입니다.")
    if r["direction"]:
        fb = r["direction"]["forward_minus_backward_nt"]
        worse = max(fb, key=lambda k: abs(fb[k] or 0))
        lines.append(f"방향 반전 효과(왕복 측선 평균 차): 센서 {s1} {fb[str(s1)]:+} nT, 센서 {s2} {fb[str(s2)]:+} nT - 센서 {worse}의 헤딩 오차가 더 큽니다. "
                     "전처리의 헤딩 보정은 센서별로 따로 계산·적용됩니다.")
    bg = r["background_std_nt"]
    if bg:
        cleaner = min(bg, key=lambda k: bg[k] if bg[k] is not None else np.inf)
        lines.append(f"배경 산포(표적 제외, 2 s 고역): 센서 {s1} {bg[str(s1)]} nT, 센서 {s2} {bg[str(s2)]} nT - 단독으로 쓴다면 센서 {cleaner}. "
                     "'두 탐침'은 격자에서 센서별 중앙값을 맞춘 뒤 평균하므로 독립 잡음만 1/√2로 줄어듭니다.")
    att = r["attitude"]
    if att:
        gains = {k: round(p["std_before_nt"] - p["std_after_nt"], 2) for k, p in att["per_sensor"].items()}
        lines.append(f"자세(롤·피치) 회귀로 줄어드는 배경 산포: " + ", ".join(f"센서 {k} {g} nT" for k, g in gains.items())
                     + (" - 자세 보상을 적용할 만합니다." if att["worth_it"] else " - 1 nT 미만이라 자세 보상은 넣지 않습니다.")
                     + (f" (차와 피치의 상관 {att['corr_difference_pitch']}는 피치가 비행 방향에 따라 달라서 생긴 것으로, 방향 반전 효과와 같은 현상입니다.)"
                        if att["corr_difference_pitch"] is not None and abs(att["corr_difference_pitch"]) > 0.7 and not att["worth_it"] else ""))
    return lines
