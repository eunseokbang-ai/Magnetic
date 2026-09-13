"""Base station (diurnal) magnetometer file loader.

Auto-detects between three base logger export shapes:
  - The common no-header CSV export: `flag, mag(nT), flag,
    "오전/오후 h:mm:ss", MM/DD/YY, flag`, saved with a UTF-8 BOM and Korean
    12-hour AM/PM markers. Each row carries its own date, so no external
    date source is needed.
  - A whitespace-separated whole-day text export: `HH MM SS X Y Z F` per
    line (one line per second, no header, no date column - GPS time-of-day
    only). The 7th column (F) is an independently-measured scalar total
    field (not derived from X/Y/Z - a real fluxgate+scalar-sensor base
    station reports both, and they don't exactly agree even though they
    measure the same field, which is expected and is why the scalar
    column is used directly rather than recomputed from X/Y/Z). Since the
    file has no date of its own, the date is parsed from the filename
    (an 8-digit YYYYMMDD run, as in "cyg202607151s.txt"); if that's not
    found, the server's current date is used as a last-resort fallback
    (flagged via df.attrs["date_fallback_used"] so callers can warn).
  - This app's own header'd `timestamp, mag_nT` CSV export (see
    store.py::export_nearest_intermagnet_csv) - lets an INTERMAGNET
    nearest-observatory diurnal estimate saved from one project be
    re-uploaded as a normal base station file in another, without redoing
    the lookup. Detected by header keywords, not exact column order.
"""
from __future__ import annotations

import io
import re
from datetime import date as _date
from datetime import datetime

import pandas as pd

_TIME_RE = re.compile(r"(오전|오후)\s*(\d{1,2}):(\d{1,2}):(\d{1,2})")
_FILENAME_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_HMS_LINE_RE = re.compile(r"^\s*\d{1,2}\s+\d{1,2}\s+\d{1,2}\s")


class BaseLoadError(ValueError):
    pass


def _parse_korean_time(value: str):
    if not isinstance(value, str):
        return None
    m = _TIME_RE.match(value.strip())
    if not m:
        return None
    period, h, mi, s = m.groups()
    h = int(h)
    if period == "오전":
        if h == 12:
            h = 0
    else:
        if h != 12:
            h += 12
    return h, int(mi), int(s)


def _load_korean_ampm_format(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None)

    if df.shape[1] < 5:
        raise BaseLoadError("베이스 파일 형식을 인식할 수 없습니다 (컬럼 수 부족).")
    df = df.iloc[:, :6] if df.shape[1] >= 6 else df

    # column layout: flag0, mag, flag2, korean_time, mmddyy_date, flag5(optional)
    mag_col, time_col, date_col = 1, 3, 4
    df = df.rename(columns={mag_col: "mag", time_col: "time_str", date_col: "date_str"})

    parsed = df["time_str"].apply(_parse_korean_time)
    valid = parsed.notna()
    df = df[valid].reset_index(drop=True)
    parsed = parsed[valid].reset_index(drop=True)
    if df.empty:
        raise BaseLoadError("베이스 파일에서 시간 형식을 인식하지 못했습니다.")

    dates = pd.to_datetime(df["date_str"].astype(str).str.strip(), format="%m/%d/%y", errors="coerce")
    hms = pd.to_timedelta([f"{h:02d}:{mi:02d}:{s:02d}" for h, mi, s in parsed])
    df["timestamp"] = dates + hms
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df = df.dropna(subset=["timestamp", "mag"])
    if df.empty:
        raise BaseLoadError("베이스 파일에 유효한 자력 데이터가 없습니다.")

    return df[["timestamp", "mag"]].sort_values("timestamp").reset_index(drop=True)


def _load_timestamp_mag_csv_format(text: str) -> pd.DataFrame:
    """A plain header'd (timestamp, mag[_nT]) CSV - specifically this
    app's own "주변 관측소 자료" export (store.py::export_nearest_
    intermagnet_csv), so a project can reuse a saved nearest-observatory
    diurnal estimate as a normal base-station upload later, in another
    project, without redoing the INTERMAGNET lookup. Detected by header
    keywords rather than a fixed column layout so it isn't tied to the
    exact export column order/casing."""
    df = pd.read_csv(io.StringIO(text))
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    ts_col = cols_lower.get("timestamp")
    mag_col = cols_lower.get("mag_nt") or cols_lower.get("mag")
    if ts_col is None or mag_col is None:
        raise BaseLoadError("베이스 파일 형식을 인식할 수 없습니다 (timestamp/mag 컬럼을 찾을 수 없습니다).")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "mag": pd.to_numeric(df[mag_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "mag"])
    if out.empty:
        raise BaseLoadError("베이스 파일에 유효한 자력 데이터가 없습니다.")
    return out.sort_values("timestamp").reset_index(drop=True)


def _date_from_filename(filename: str | None) -> _date | None:
    if not filename:
        return None
    m = _FILENAME_DATE_RE.search(filename)
    if not m:
        return None
    y, mo, d = (int(g) for g in m.groups())
    try:
        return _date(y, mo, d)
    except ValueError:
        return None


def _load_hms_xyzf_format(text: str, filename: str | None) -> pd.DataFrame:
    date_val = _date_from_filename(filename)
    date_fallback_used = date_val is None
    if date_val is None:
        date_val = datetime.now().date()

    hh, mm, ss, mag = [], [], [], []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 7:
            continue
        try:
            h, m, s = int(tokens[0]), int(tokens[1]), int(tokens[2])
            f = float(tokens[6])
        except ValueError:
            continue
        hh.append(h)
        mm.append(m)
        ss.append(s)
        mag.append(f)

    if not mag:
        raise BaseLoadError("베이스 파일(HH MM SS X Y Z F 형식)에서 유효한 데이터 행을 찾지 못했습니다.")

    base_ts = pd.Timestamp(date_val)
    timestamp = (
        base_ts
        + pd.to_timedelta(hh, unit="h")
        + pd.to_timedelta(mm, unit="m")
        + pd.to_timedelta(ss, unit="s")
    )
    df = pd.DataFrame({"timestamp": timestamp, "mag": mag})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["date_fallback_used"] = date_fallback_used
    return df


def load_base_csv(path_or_buffer, filename: str | None = None) -> pd.DataFrame:
    """Parse a base station file (either supported format - see module
    docstring) into a normalized DataFrame with columns: timestamp, mag."""
    if hasattr(path_or_buffer, "read"):
        raw = path_or_buffer.read()
    else:
        with open(path_or_buffer, "rb") as f:
            raw = f.read()
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw

    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    if not first_line:
        raise BaseLoadError("베이스 파일이 비어 있습니다.")

    header_lower = first_line.lower()
    if "timestamp" in header_lower and "mag" in header_lower:
        return _load_timestamp_mag_csv_format(text)
    if "," in first_line:
        return _load_korean_ampm_format(text)
    if _HMS_LINE_RE.match(first_line):
        return _load_hms_xyzf_format(text, filename)
    raise BaseLoadError("베이스 파일 형식을 인식할 수 없습니다.")


def load_base_csvs(buffers: list, filenames: list | None = None) -> pd.DataFrame:
    """Load and concatenate multiple base station files (e.g. logs split
    across days, or several deployments), re-sorted by timestamp and
    deduplicated on exact-timestamp collisions (e.g. an overlapping
    re-upload of the same log). The removed-duplicate count is attached
    via combined.attrs for the caller to surface to the user, along with
    whether any file's date had to fall back to "today" (see
    _load_hms_xyzf_format) rather than being read from its filename."""
    if not buffers:
        raise BaseLoadError("베이스 파일이 없습니다.")
    names = filenames if filenames is not None else [None] * len(buffers)
    parts = [load_base_csv(buf, name) for buf, name in zip(buffers, names)]
    any_date_fallback = any(p.attrs.get("date_fallback_used", False) for p in parts)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    n_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    combined.attrs["n_duplicate_timestamps_removed"] = n_before_dedup - len(combined)
    combined.attrs["date_fallback_used"] = any_date_fallback
    return combined
