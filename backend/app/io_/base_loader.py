"""Base station (diurnal) magnetometer CSV loader.

Handles the common no-header export shape used by portable base
magnetometer loggers: `flag, mag(nT), flag, "오전/오후 h:mm:ss", MM/DD/YY, flag`,
saved with a UTF-8 BOM and Korean 12-hour AM/PM markers.
"""
from __future__ import annotations

import re

import pandas as pd

_TIME_RE = re.compile(r"(오전|오후)\s*(\d{1,2}):(\d{1,2}):(\d{1,2})")


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


def load_base_csv(path_or_buffer) -> pd.DataFrame:
    """Parse a base station CSV into a normalized DataFrame.

    Returns columns: timestamp, mag.
    """
    df = pd.read_csv(path_or_buffer, header=None, encoding="utf-8-sig")

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

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "mag"]]


def load_base_csvs(buffers: list) -> pd.DataFrame:
    """Load and concatenate multiple base station CSVs (e.g. logs split
    across days, or several deployments), re-sorted by timestamp and
    deduplicated on exact-timestamp collisions (e.g. an overlapping
    re-upload of the same log). The removed-duplicate count is attached
    via combined.attrs for the caller to surface to the user."""
    if not buffers:
        raise BaseLoadError("베이스 파일이 없습니다.")
    parts = [load_base_csv(buf) for buf in buffers]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    n_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    combined.attrs["n_duplicate_timestamps_removed"] = n_before_dedup - len(combined)
    return combined
