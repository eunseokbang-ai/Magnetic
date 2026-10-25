"""SENSYS MagDrone R3 ASC export, as the landmine program reads it.

The R3 is a two-probe fluxgate array (probes 1 m apart on a rod) logging
at 200 Hz. The DataTool export this reads is a tab-separated table with
one row per probe per sample:

    Timestamp [ms]  Time [hh:mm:ss.sss]  Track ID  Sensor ID  Latitude [°]
    Longitude [°]  Altitude [m]  Total field anomaly [nT]  Mag-X [nT]
    Mag-Y [nT]  Mag-Z [nT]

Older exports have fewer columns (no Time, Track ID or vector
components); columns are matched by name, and whatever is missing comes
back as NaN or -1 rather than failing. Unlike the mineral program's
loader, which keeps the first probe only and drops the altitude, this one
keeps everything: for target detection the second probe is half the data
and the probe height is what sets the anomaly amplitude.

"Timestamp [ms]" is milliseconds since midnight (it equals the Time
column), and the calendar date is only in the file name
(YYYYMMDD_HHMMSS_MD-R3_#serial_tokens.asc), so the two are combined here
into real timestamps, rolling over at midnight. The remaining file-name
tokens (noff, sf5, mm3(ROBOT1), 2Hz ...) are the operator's export
settings and are kept as text for the record.

Measured on the KIGAM test site files (2026-04): the positions of the two
probes at one timestamp are 1.00 m apart, the sample interval is 5 ms,
and a quadruped robot run has one 11-minute gap - all of which the
summary reports.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class MagDroneLoadError(ValueError):
    pass


# Column names after stripping the unit in brackets and lower-casing.
_COLUMNS = {
    "t_ms": ("timestamp",),
    "time": ("time",),
    "track": ("track id", "track"),
    "sensor": ("sensor id", "sensor", "probe id", "probe"),
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "lon"),
    "alt": ("altitude", "alt", "height"),
    "tfa": ("total field anomaly", "total field", "tmi", "anomaly"),
    "bx": ("mag-x", "x"),
    "by": ("mag-y", "y"),
    "bz": ("mag-z", "z"),
}
_REQUIRED = ("t_ms", "sensor", "lat", "lon", "tfa")
_FILENAME_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})_(?P<device>MD-R\d)_?(?P<serial>#\d+)?(?P<rest>.*)$")
# A sample interval longer than this is a gap (a pause, a battery change).
GAP_THRESHOLD_MS = 1000.0
# The fourth difference of white noise has variance 70 sigma^2.
_FOURTH_DIFF_NORM = np.sqrt(70.0)


@dataclass
class FileInfo:
    filename: str
    device: str | None
    serial: str | None
    start_date: str | None          # from the file name; None = unknown (placeholder date used)
    settings: list = field(default_factory=list)
    n_rows: int = 0
    columns_present: list = field(default_factory=list)
    columns_missing: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _norm(name: str) -> str:
    return re.sub(r"\s*\[.*?\]\s*", "", name).strip().lower()


def _map_columns(header: list[str]) -> dict[str, str]:
    normed = {_norm(h): h for h in header}
    out = {}
    for key, aliases in _COLUMNS.items():
        for alias in aliases:
            if alias in normed:
                out[key] = normed[alias]
                break
    missing = [k for k in _REQUIRED if k not in out]
    if missing:
        raise MagDroneLoadError(
            f"MagDrone R3 ASC 파일에 필수 열이 없습니다: {missing}. 헤더: {header}"
        )
    return out


def parse_filename(filename: str) -> FileInfo:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename.split("/")[-1].split("\\")[-1])
    m = _FILENAME_RE.match(stem)
    if not m:
        return FileInfo(filename=filename, device=None, serial=None, start_date=None)
    d = m.group("date")
    rest = m.group("rest").strip("_")
    settings = [tok for tok in rest.split("_") if tok]
    return FileInfo(
        filename=filename,
        device=m.group("device"),
        serial=m.group("serial"),
        start_date=f"{d[:4]}-{d[4:6]}-{d[6:]}",
        settings=settings,
    )


def _split_header(line: str) -> list[str]:
    if "\t" in line:
        return [h.strip() for h in line.split("\t")]
    if ";" in line:
        return [h.strip() for h in line.split(";")]
    return [h.strip() for h in re.split(r"\s{2,}", line.strip())]


def load_magdrone_r3(data: bytes | str, filename: str = "") -> pd.DataFrame:
    """One ASC export -> long-format frame, one row per probe per sample.

    Columns: timestamp, t_ms, track, sensor, lat, lon, alt, tfa, bx, by,
    bz, b_total. attrs["file_info"] holds the FileInfo dict.
    """
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if _norm(ln.split("\t")[0]).startswith("timestamp")), None)
    if header_idx is None:
        raise MagDroneLoadError("MagDrone R3 ASC 파일의 헤더(Timestamp [ms] ...)를 찾을 수 없습니다.")
    header = _split_header(lines[header_idx])
    sep = "\t" if "\t" in lines[header_idx] else (";" if ";" in lines[header_idx] else r"\s+")
    colmap = _map_columns(header)

    body = "\n".join(lines[header_idx + 1:])
    raw = pd.read_csv(io.StringIO(body), sep=sep, header=None, names=header, engine="python" if sep == r"\s+" else "c",
                      dtype=str, skip_blank_lines=True)
    if raw.empty:
        raise MagDroneLoadError("MagDrone R3 ASC 파일에 데이터 행이 없습니다.")

    def num(key):
        if key not in colmap:
            return np.full(len(raw), np.nan)
        return pd.to_numeric(raw[colmap[key]], errors="coerce").to_numpy(dtype=float)

    df = pd.DataFrame({
        "t_ms": num("t_ms"),
        "sensor": num("sensor"),
        "lat": num("lat"),
        "lon": num("lon"),
        "alt": num("alt"),
        "tfa": num("tfa"),
        "bx": num("bx"),
        "by": num("by"),
        "bz": num("bz"),
    })
    df["track"] = num("track")
    ok = np.isfinite(df["t_ms"]) & np.isfinite(df["sensor"]) & np.isfinite(df["lat"]) & np.isfinite(df["lon"]) & np.isfinite(df["tfa"])
    df = df[ok].reset_index(drop=True)
    if df.empty:
        raise MagDroneLoadError("MagDrone R3 ASC 파일에 유효한 데이터 행이 없습니다.")
    df["sensor"] = df["sensor"].astype(int)
    df["track"] = df["track"].fillna(-1).astype(int)
    df["b_total"] = np.sqrt(df["bx"] ** 2 + df["by"] ** 2 + df["bz"] ** 2)

    info = parse_filename(filename)
    info.n_rows = int(len(df))
    info.columns_present = [k for k in _COLUMNS if k in colmap]
    info.columns_missing = [k for k in _COLUMNS if k not in colmap]

    # Timestamp [ms] is ms since midnight; the date comes from the file name.
    # The export lists each probe's whole timeline back to back, so the
    # midnight rollover (a drop of more than 12 h between consecutive
    # samples) is looked for within each probe's block, in file order, and
    # the rows are then put into time order.
    base = pd.Timestamp(info.start_date) if info.start_date else pd.Timestamp("2000-01-01")
    ms = df["t_ms"].to_numpy()
    rollover = np.zeros(len(ms))
    for s in df["sensor"].unique():
        idx = np.flatnonzero(df["sensor"].to_numpy() == s)
        jumps = np.diff(ms[idx]) < -12 * 3600 * 1000
        if jumps.any():
            rollover[idx] = np.r_[0.0, np.cumsum(jumps)] * 86400000.0
    df["timestamp"] = base + pd.to_timedelta(ms + rollover, unit="ms")
    df = df.sort_values(["timestamp", "sensor"], kind="stable").reset_index(drop=True)
    df.attrs["file_info"] = info.to_dict()
    cols = ["timestamp", "t_ms", "track", "sensor", "lat", "lon", "alt", "tfa", "bx", "by", "bz", "b_total"]
    return df[cols]


def load_magdrone_r3_files(buffers: list, filenames: list[str] | None = None) -> pd.DataFrame:
    """Several exports (several flights or runs) -> one frame sorted by
    time, with file_index and point_id. attrs["files"] lists each file's
    FileInfo dict."""
    frames, infos = [], []
    for i, buf in enumerate(buffers):
        name = filenames[i] if filenames and i < len(filenames) and filenames[i] else f"file_{i}"
        data = buf.read() if hasattr(buf, "read") else buf
        df = load_magdrone_r3(data, name)
        df["file_index"] = i
        infos.append(df.attrs["file_info"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "sensor"], kind="stable").reset_index(drop=True)
    out["point_id"] = np.arange(len(out))
    out.attrs["files"] = infos
    return out


# ------------------------------------------------------------- summary
def _stats(a) -> dict | None:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    return {"min": round(float(a.min()), 2), "median": round(float(np.median(a)), 2),
            "max": round(float(a.max()), 2), "std": round(float(a.std()), 2)}


def _local_xy(lat, lon):
    lat0 = float(np.nanmean(lat))
    return (lon - np.nanmean(lon)) * 111320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 110540.0


def summarize(df: pd.DataFrame) -> dict:
    """What the operator needs to know about a loaded set before any
    processing: probes and their separation, sampling and gaps, tracks,
    speed, altitude, field range, per-probe noise and offset."""
    sensors = sorted(int(s) for s in df["sensor"].unique())
    primary = sensors[0]
    s1 = df[df["sensor"] == primary].sort_values("timestamp")
    # Sample intervals, gaps, duration and speed are measured inside each
    # file: the days between two flights are not a gap in either of them.
    file_col = s1["file_index"].to_numpy() if "file_index" in s1 else np.zeros(len(s1), dtype=int)
    t_all = s1["timestamp"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    x_all, y_all = _local_xy(s1["lat"].to_numpy(), s1["lon"].to_numpy())
    dts, speeds, duration_ms = [], [], 0.0
    for f in np.unique(file_col):
        m = file_col == f
        t_f = t_all[m]
        if len(t_f) > 1:
            d = np.diff(t_f)
            dts.append(d)
            duration_ms += float(t_f[-1] - t_f[0])
            step = np.hypot(np.diff(x_all[m]), np.diff(y_all[m]))
            ok = (d > 0) & (d <= GAP_THRESHOLD_MS)
            speeds.append(step[ok] / (d[ok] / 1000.0))
    dt = np.concatenate(dts) if dts else np.array([])
    gaps = dt[dt > GAP_THRESHOLD_MS] if len(dt) else np.array([])
    speed = np.concatenate(speeds) if speeds else np.array([])
    speed = speed[np.isfinite(speed)]
    t = t_all
    x, y = x_all, y_all

    # Probe separation: distance between probes logged at the same instant.
    separation = {}
    if len(sensors) > 1:
        key = ["file_index", "t_ms"] if "file_index" in df else ["t_ms"]
        piv = df.pivot_table(index=key, columns="sensor", values=["lat", "lon"], aggfunc="first")
        for s in sensors[1:]:
            if s in piv["lat"].columns:
                lat1, lon1 = piv["lat"][primary].to_numpy(), piv["lon"][primary].to_numpy()
                lat2, lon2 = piv["lat"][s].to_numpy(), piv["lon"][s].to_numpy()
                dx = (lon2 - lon1) * 111320.0 * np.cos(np.radians(np.nanmean(lat1)))
                dy = (lat2 - lat1) * 110540.0
                d = np.hypot(dx, dy)
                d = d[np.isfinite(d)]
                if len(d):
                    separation[str(s)] = round(float(np.median(d)), 2)

    per_sensor = {}
    for s in sensors:
        g = df[df["sensor"] == s].sort_values("timestamp")
        v = g["tfa"].to_numpy()
        noise = None
        if len(v) > 10:
            d4 = v[4:] - 4 * v[3:-1] + 6 * v[2:-2] - 4 * v[1:-3] + v[:-4]
            noise = round(float(np.sqrt(np.mean(d4 ** 2)) / _FOURTH_DIFF_NORM), 3)
        per_sensor[str(s)] = {
            "n": int(len(g)),
            "tfa_mean": round(float(v.mean()), 2),
            "tfa_std": round(float(v.std()), 2),
            "noise_4th_diff_nt": noise,
            "b_total_mean": round(float(np.nanmean(g["b_total"])), 1) if np.isfinite(g["b_total"]).any() else None,
        }
    # Offset of each probe against the first, on samples logged at the same instant.
    offsets = {}
    if len(sensors) > 1:
        key = ["file_index", "t_ms"] if "file_index" in df else ["t_ms"]
        piv = df.pivot_table(index=key, columns="sensor", values="tfa", aggfunc="first")
        for s in sensors[1:]:
            if s in piv.columns:
                d = (piv[s] - piv[primary]).dropna()
                if len(d):
                    offsets[str(s)] = round(float(np.median(d)), 2)

    # Track ids restart at 1 in every export, so count them per file.
    on_track = df[df["track"] >= 0]
    if "file_index" in df:
        n_tracks = int(on_track.groupby("file_index")["track"].nunique().sum())
    else:
        n_tracks = int(on_track["track"].nunique())
    files = df.attrs.get("files") or ([df.attrs["file_info"]] if "file_info" in df.attrs else [])
    return {
        "n_rows": int(len(df)),
        "n_files": int(df["file_index"].nunique()) if "file_index" in df else 1,
        "files": files,
        "sensors": sensors,
        "sensor_separation_m": separation,
        "sample_interval_ms": round(float(np.median(dt)), 2) if len(dt) else None,
        "n_gaps": int(len(gaps)),
        "longest_gap_s": round(float(gaps.max()) / 1000.0, 1) if len(gaps) else 0.0,
        "duration_min": round(duration_ms / 60000.0, 1),
        "time_range": [str(s1["timestamp"].iloc[0]), str(s1["timestamp"].iloc[-1])],
        "n_tracks": n_tracks,
        "speed_m_s": _stats(speed) if len(speed) else None,
        "altitude_m": _stats(df["alt"]),
        "tfa_nt": _stats(df["tfa"]),
        "has_vector": bool(np.isfinite(df["bx"]).any()),
        "per_sensor": per_sensor,
        "sensor_offset_nt": offsets,
        "extent_m": {"x": round(float(np.ptp(x)), 1), "y": round(float(np.ptp(y)), 1)} if len(x) else None,
    }
