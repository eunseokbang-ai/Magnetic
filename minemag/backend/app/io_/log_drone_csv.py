"""The "Log_Drone" CSV of the two-sensor optically pumped scalar
magnetometer flown at the Gongju ADD site (2026-10-19), read into the
same long-format frame as the MagDrone R3 export.

File layout (comma separated, UTF-8 with BOM, one header row):

    LaserCurr_J1[mA], LaserCurr_J2[mA], LaserTemp_J1[℃], LaserTemp_J2[℃],
    ModFreq_J1[KHz], ModFreq_J2[KHz], ModAmp_J1[mApp], ModAmp_J2[mApp],
    HTPWR_J1[%], HTPWR_J2[%], CellTemp_J1[Ohm], CellTemp_J2[Ohm],
    MagField_J1[uT], MagField_J2[uT], SeqNum, TOW, Lat[deg], Lon[deg],
    Alt[m], Roll[s], Pitch[uT], Yaw[s], Vel. N, Vel. E, Vel. D

What was measured on the two files of case8 (격자비행):

- Two sensors J1 and J2 (modulation ~354 kHz = twice the caesium Larmor
  frequency at 50.6 uT, so a Cs optically pumped scalar magnetometer),
  logging the total field in microtesla at 50 Hz (TOW step 0.020 s).
  Dropped packets show as SeqNum skips of 2-9 (955 of 17,126 rows in
  the grid flight, longest 0.18 s) - far below the 1 s gap threshold.
- One GNSS position per row, shared by both sensors: the sensor
  separation cannot be measured from the file (the R3 export has a
  position per probe). J1 - J2 is 42 nT on the ground and 52 +- 10 nT in
  flight, so the pair is a gradiometer with a fixed offset.
- TOW is GPS seconds of week; the date is only in the file name
  (Log_Drone_YYYYMMDD_HHMMSS.csv). The local UTC offset is derived from
  the file-name time against the first TOW (9 h here), not assumed.
- No track numbers: the drone flies with its nose fixed (Yaw constant
  -16 deg) and sweeps parallel lines, so tracks are made here from the
  path - straight, moving stretches of 5 m or more (segment_tracks).
  The grid flight gives 14 tracks of heading 163 deg, 54-76 m long,
  1.5-3 m apart. The speed bound matters: at 0.8 m/s the pilot's
  slow-downs mid-line cut two lines in two and dropped 35% of the
  samples; at 0.3 m/s (the pre-processing's own bound) 28% is off-track,
  which is the turns (24%) and the line ends. Samples the pre-processing
  then leaves out are a further 0.2%.
- MagField is the total field, not an anomaly: tfa = field - IGRF at the
  reading's position and date, minus the file's median of that difference
  (-180 nT on the grid flight: regional field, model error and sensor
  offset together), so it sits around zero like the R3's "total field
  anomaly" and the grid, picking and p2p classes behave the same. The
  J1 - J2 offset is kept (one level for both sensors). The static file (54 s on the ground) has J1 noise 0.3 nT rms at
  50 Hz - an order of magnitude quieter than the R3 fluxgates.
- Roll/Pitch/Yaw carry wrong unit labels in the header ([s], [uT]); they
  are degrees and are kept as roll/pitch/yaw columns.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from magcore.igrf import compute_igrf_total_field

from .magdrone_r3 import FileInfo, MagDroneLoadError

# Leap seconds between GPS time and UTC (constant since 2017-01-01).
GPS_UTC_LEAP_S = 18
_FILENAME_RE = re.compile(r"Log_Drone_(?P<date>\d{8})_(?P<time>\d{6})", re.IGNORECASE)
_SENSOR_RE = re.compile(r"MagField_(J\d+)", re.IGNORECASE)

# segment_tracks defaults, from the case8 grid flight (see module docstring)
TRACK_SMOOTH_S = 0.5            # half-width of the position smoothing window
TRACK_MIN_SPEED_M_S = 0.3       # slower is a hover, a turn-around or the ground (same bound as the pre-processing)
TRACK_MAX_TURN_DEG = 30.0       # heading change over 1 s that still counts as straight
TRACK_MIN_LENGTH_M = 5.0        # shorter straight stretches are not lines


class LogDroneLoadError(MagDroneLoadError):
    pass


def is_log_drone_csv(data: bytes | str) -> bool:
    head = (data[:2048].decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data[:2048])
    first = head.splitlines()[0] if head.splitlines() else ""
    return "MagField_" in first and "TOW" in first


def _norm(name: str) -> str:
    return re.sub(r"\s*\[.*?\]\s*", "", name).strip()


def segment_tracks(x, y, t, *, smooth_s=TRACK_SMOOTH_S, min_speed=TRACK_MIN_SPEED_M_S,
                   max_turn_deg=TRACK_MAX_TURN_DEG, min_length_m=TRACK_MIN_LENGTH_M) -> np.ndarray:
    """Track number (1, 2, ...) per sample from the flight path alone, -1
    off-line: a sample is on a line when the smoothed path moves at survey
    speed and its heading changes by less than `max_turn_deg` over one
    second; each such stretch at least `min_length_m` long is a track."""
    x, y, t = (np.asarray(a, dtype=float) for a in (x, y, t))
    n = len(x)
    track = np.full(n, -1, dtype=int)
    if n < 10:
        return track
    dt_med = float(np.nanmedian(np.diff(t))) if n > 1 else 0.02
    w = max(1, int(round(smooth_s / max(dt_med, 1e-3))))
    xs = pd.Series(x).rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
    ys = pd.Series(y).rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
    dx, dy, dt = np.gradient(xs), np.gradient(ys), np.gradient(t)
    speed = np.hypot(dx, dy) / np.where(dt > 0, dt, np.nan)
    heading = np.degrees(np.arctan2(dx, dy))
    lag = max(1, int(round(1.0 / max(dt_med, 1e-3))))
    ahead = np.r_[heading[lag:], np.repeat(heading[-1], lag)]
    behind = np.r_[np.repeat(heading[0], lag), heading[:-lag]]
    turn = np.abs(((ahead - behind) + 180.0) % 360.0 - 180.0)
    straight = (speed > min_speed) & (turn < max_turn_deg)
    straight = np.nan_to_num(straight, nan=False).astype(bool)
    k, i = 0, 0
    while i < n:
        if not straight[i]:
            i += 1
            continue
        j = i
        while j < n and straight[j]:
            j += 1
        if np.hypot(xs[j - 1] - xs[i], ys[j - 1] - ys[i]) >= min_length_m:
            k += 1
            track[i:j] = k
        i = j
    return track


def parse_filename(filename: str) -> tuple[str | None, int | None]:
    """(YYYY-MM-DD, seconds of day of the local start time) from
    Log_Drone_YYYYMMDD_HHMMSS; (None, None) when the name is something else."""
    m = _FILENAME_RE.search(filename.split("/")[-1].split("\\")[-1])
    if not m:
        return None, None
    d, hms = m.group("date"), m.group("time")
    return f"{d[:4]}-{d[4:6]}-{d[6:]}", int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:])


def _timestamps(tow: np.ndarray, start_date: str | None, local_start_s: int | None) -> tuple[pd.Series, np.ndarray, float]:
    """Local naive timestamps from GPS time of week + the file-name date.
    Returns (timestamp, ms since local midnight, utc offset hours)."""
    base = pd.Timestamp(start_date) if start_date else pd.Timestamp("2000-01-01")
    gps_dow = (base.weekday() + 1) % 7                     # GPS week starts on Sunday
    utc_s = tow - gps_dow * 86400.0 - GPS_UTC_LEAP_S
    if local_start_s is not None and np.isfinite(utc_s[0]):
        offset = round((local_start_s - utc_s[0]) / 1800.0) * 1800.0
        offset = ((offset + 12 * 3600) % (24 * 3600)) - 12 * 3600
    else:
        offset = 0.0
    local_s = utc_s + offset
    ts = base + pd.to_timedelta(local_s, unit="s")
    day_ms = ((local_s % 86400.0) * 1000.0)
    return ts, day_ms, offset / 3600.0


def load_log_drone_csv(data: bytes | str, filename: str = "") -> pd.DataFrame:
    """One Log_Drone CSV -> long-format frame, one row per sensor per
    sample, with the MagDrone R3 loader's columns plus roll/pitch/yaw.
    attrs["file_info"] holds the FileInfo dict."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    try:
        raw = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    except Exception as exc:                     # noqa: BLE001 - any parse failure is a format error
        raise LogDroneLoadError(f"Log_Drone CSV를 읽지 못했습니다: {exc}") from exc
    raw.columns = [_norm(c) for c in raw.columns]
    mag_cols = {m.group(1).upper(): c for c in raw.columns for m in [_SENSOR_RE.match(c)] if m}
    needed = {"TOW", "Lat", "Lon"}
    if not mag_cols or not needed.issubset(raw.columns):
        raise LogDroneLoadError(f"Log_Drone CSV에 필수 열이 없습니다 (MagField_J*, TOW, Lat, Lon). 헤더: {list(raw.columns)}")

    num = lambda c: pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float) if c in raw.columns else np.full(len(raw), np.nan)
    tow, lat, lon, alt = num("TOW"), num("Lat"), num("Lon"), num("Alt")
    ok = np.isfinite(tow) & np.isfinite(lat) & np.isfinite(lon)
    for c in mag_cols.values():
        ok &= np.isfinite(num(c))
    if not ok.any():
        raise LogDroneLoadError("Log_Drone CSV에 유효한 데이터 행이 없습니다.")
    order = np.argsort(tow[ok], kind="stable")
    keep = np.flatnonzero(ok)[order]
    tow, lat, lon, alt = tow[keep], lat[keep], lon[keep], alt[keep]
    roll, pitch, yaw = num("Roll")[keep], num("Pitch")[keep], num("Yaw")[keep]

    start_date, local_start_s = parse_filename(filename)
    ts, day_ms, utc_offset_h = _timestamps(tow, start_date, local_start_s)

    # tracks from the path (local metric coordinates are enough here)
    lat0 = float(np.nanmean(lat))
    x = (lon - np.nanmean(lon)) * 111320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540.0
    track = segment_tracks(x, y, tow)

    # IGRF at each reading; the anomaly is what the rest of the program reads
    alt_for_igrf = np.where(np.isfinite(alt), alt, 50.0)
    try:
        igrf = compute_igrf_total_field(lat, lon, alt_for_igrf, pd.Series(ts))
        igrf_note = "IGRF"
    except Exception:                           # noqa: BLE001 - offline model file missing etc.
        igrf = None
        igrf_note = "median"

    # The absolute sensor reading differs from IGRF by a few hundred nT
    # (regional field, model error, sensor offset): the file's median of
    # B - IGRF is removed so the anomaly is centred on zero like the R3's,
    # with one value for both sensors so their mutual offset stays visible.
    b_all = np.concatenate([num(c)[keep] * 1000.0 for c in mag_cols.values()])
    ref = np.tile(igrf, len(mag_cols)) if igrf is not None else 0.0
    level = float(np.nanmedian(b_all - ref))
    frames = []
    for s, (jname, col) in enumerate(sorted(mag_cols.items()), start=1):
        b_nt = num(col)[keep] * 1000.0
        tfa = (b_nt - igrf if igrf is not None else b_nt) - level
        frames.append(pd.DataFrame({
            "timestamp": ts, "t_ms": day_ms, "track": track, "sensor": s,
            "lat": lat, "lon": lon, "alt": alt, "tfa": tfa,
            "bx": np.nan, "by": np.nan, "bz": np.nan, "b_total": b_nt,
            "roll": roll, "pitch": pitch, "yaw": yaw,
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["timestamp", "sensor"], kind="stable").reset_index(drop=True)

    info = FileInfo(filename=filename, device="Log_Drone OPM", serial=None, start_date=start_date,
                    settings=[f"{'·'.join(sorted(mag_cols))}", "50Hz" if np.isclose(np.nanmedian(np.diff(tow)), 0.02, atol=0.005) else f"{1 / max(np.nanmedian(np.diff(tow)), 1e-3):.0f}Hz",
                              f"UTC{utc_offset_h:+.0f}", f"anomaly={igrf_note}{level:+.0f}nT", f"tracks={int(track.max()) if track.max() > 0 else 0}"])
    info.format = "log_drone_csv"
    info.n_rows = int(len(df))
    info.columns_present = ["t_ms", "sensor", "lat", "lon", "alt", "tfa", "roll", "pitch", "yaw"]
    info.columns_missing = ["track", "bx", "by", "bz"]
    df.attrs["file_info"] = info.to_dict()
    return df
