"""Substitute base-station data from a public INTERMAGNET geomagnetic
observatory, for surveys where no local base station was measured at all.

A local base log captures the true local diurnal (Sq) variation plus
whatever local/instrumental noise it happens to have. A remote observatory
cannot capture the local part, but the diurnal correction applied elsewhere
in this app (processing/diurnal.py::apply_diurnal_correction) only ever
uses (base_interp - reference_within_flight_window) - i.e. it removes the
*variation* relative to the survey window's own mean, never the absolute
field level. That means a remote station's absolute baseline offset from
the true local field cancels out automatically, so its time series can be
plugged in as a drop-in "base_raw" substitute to capture the shared,
regional/solar-driven part of the variation (a reasonable approximation
when nothing local was recorded), while still missing any genuinely local
diurnal/cultural signal a real local base station would have caught.

Two ways to get an observatory's data into this shape:
  - parse_iaga2002(): parse an IAGA-2002 text file (the standard format
    INTERMAGNET/observatories publish), whether fetched automatically or
    downloaded by hand from https://intermagnet.org and uploaded.
  - fetch_iaga2002_text(): pull that same text automatically from the BGS
    Global INTERMAGNET Node's public web service, for observatories/dates
    the network policy of wherever this server runs allows reaching.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests


class IagaParseError(ValueError):
    pass


class IntermagnetFetchError(RuntimeError):
    pass


@dataclass
class IagaObservatoryData:
    station_name: str
    iaga_code: str
    lat: float | None
    lon: float | None
    elevation_m: float | None
    reported: str
    df: pd.DataFrame  # columns: timestamp, mag (nT total field)


# Component-code combinations that can be turned into a total field F
# without needing declination (D), which IAGA-2002 files store in units
# (minutes vs. degrees) that vary enough between sources to be a footgun
# for an unattended conversion - so if a file doesn't report F directly or
# give us X/Y/Z or H/Z to combine, we raise rather than guess.
def _total_field(values: dict[str, np.ndarray]) -> np.ndarray:
    if "F" in values:
        return values["F"]
    if {"X", "Y", "Z"}.issubset(values):
        return np.sqrt(values["X"] ** 2 + values["Y"] ** 2 + values["Z"] ** 2)
    if {"H", "Z"}.issubset(values):
        return np.sqrt(values["H"] ** 2 + values["Z"] ** 2)
    raise IagaParseError(
        f"총자력(F)을 계산할 수 없는 성분 조합입니다: {sorted(values)} "
        "(F, 또는 X/Y/Z, 또는 H/Z 성분이 필요합니다 - 편각(D)만으로는 변환하지 않습니다)."
    )


_HEADER_LABEL_END = 24  # IAGA-2002 fixed-width header: label occupies columns [1:24)
_FILL_THRESHOLD = 90000.0  # standard fill value is 99999.00; anything this large is missing


def parse_iaga2002(text: str) -> IagaObservatoryData:
    """Parse IAGA-2002 formatted text (as published by INTERMAGNET
    observatories/GINs) into an IagaObservatoryData with a (timestamp, mag)
    DataFrame ready to use as a diurnal-correction base series."""
    lines = text.splitlines()
    header: dict[str, str] = {}
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped.startswith(("DATE", "date")):
            data_start = i + 1
            break
        if line.startswith(" ") and not line.startswith(" #"):
            content = stripped[:-1] if stripped.endswith("|") else stripped
            label = content[1:_HEADER_LABEL_END].strip()
            value = content[_HEADER_LABEL_END:].strip()
            if label:
                header[label] = value

    if data_start is None or data_start >= len(lines):
        raise IagaParseError("IAGA-2002 형식이 아닙니다 (DATE로 시작하는 컬럼 헤더 줄을 찾을 수 없음).")

    reported = header.get("Reported", "").strip()
    components = [c for c in reported if c.isalpha()]
    if not components:
        raise IagaParseError("헤더에 'Reported' 필드(측정 성분, 예: XYZF)가 없습니다.")

    timestamps: list = []
    values_by_component: dict[str, list] = {c: [] for c in components}
    for line in lines[data_start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 3 + len(components):
            continue
        date_str, time_str = tokens[0], tokens[1]
        value_tokens = tokens[3 : 3 + len(components)]
        try:
            ts = pd.Timestamp(f"{date_str} {time_str}")
            vals = [float(v) for v in value_tokens]
        except ValueError:
            continue
        timestamps.append(ts)
        for c, v in zip(components, vals):
            values_by_component[c].append(v)

    if not timestamps:
        raise IagaParseError("IAGA-2002 자료에서 유효한 데이터 행을 찾지 못했습니다.")

    arrays = {c: np.array(v, dtype=float) for c, v in values_by_component.items()}
    for arr in arrays.values():
        arr[arr >= _FILL_THRESHOLD] = np.nan

    mag = _total_field(arrays)
    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "mag": mag})
    df = df.dropna(subset=["mag"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise IagaParseError("총자력 계산 후 유효한 데이터가 남지 않았습니다 (결측값 비율이 너무 높음).")

    lat = _parse_float(header.get("Geodetic Latitude"))
    lon = _parse_float(header.get("Geodetic Longitude"))
    if lon is not None and lon > 180:
        lon -= 360  # IAGA-2002 reports longitude as 0-360 east; normalize to -180..180
    elevation = _parse_float(header.get("Elevation"))

    return IagaObservatoryData(
        station_name=header.get("Station Name", "").strip() or header.get("IAGA CODE", "").strip() or "unknown",
        iaga_code=(header.get("IAGA CODE", "") or header.get("IAGA Code", "")).strip(),
        lat=lat,
        lon=lon,
        elevation_m=elevation,
        reported=reported,
        df=df,
    )


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    m = re.search(r"-?\d+\.?\d*", value)
    return float(m.group()) if m else None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# BGS Edinburgh is one of INTERMAGNET's three official Global INTERMAGNET
# Nodes (GINs) and serves definitive/adjusted/reported minute data for the
# whole INTERMAGNET network through this public web service. Reachability
# depends on the deploying server's own outbound network policy - see
# fetch_iaga2002_text's docstring.
_GIN_BASE_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"


def fetch_iaga2002_text(iaga_code: str, start_date: date, days: int = 2, timeout_seconds: float = 20.0) -> str:
    """Download `days` days of minute-resolution IAGA-2002 data for
    observatory `iaga_code` starting at `start_date`, from the BGS
    INTERMAGNET GIN web service.

    This makes a live outbound HTTPS request. It has not been exercised
    against the real service in every deployment environment - some
    networks (locked-down corporate/sandboxed egress policies in
    particular) block it entirely, in which case this raises
    IntermagnetFetchError and the caller should fall back to a manually
    downloaded IAGA-2002 file instead (see the /base/iaga2002 upload
    endpoint, which uses the same parse_iaga2002() and needs no outbound
    network access at all)."""
    if not re.fullmatch(r"[A-Za-z0-9]{3,4}", iaga_code or ""):
        raise IntermagnetFetchError(f"올바르지 않은 IAGA 관측소 코드입니다: {iaga_code!r}")

    params = {
        "Request": "GetData",
        "format": "iaga2002",
        "testObsys": "0",
        "observatoryIagaCode": iaga_code.upper(),
        "samplesPerDay": "minute",
        "publicationState": "adj-or-rep",
        "dataStartDate": f"{start_date.isoformat()}T00:00:00.000Z",
        "dataDuration": str(max(1, int(days))),
    }
    try:
        resp = requests.get(_GIN_BASE_URL, params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise IntermagnetFetchError(
            f"INTERMAGNET 관측소 자료 요청에 실패했습니다 (네트워크 접근이 막혀 있을 수 있습니다): {exc}"
        ) from exc

    if resp.status_code != 200:
        raise IntermagnetFetchError(
            f"INTERMAGNET 서버가 오류를 반환했습니다 (HTTP {resp.status_code}). "
            "관측소 코드/날짜를 확인하거나, 해당 날짜 자료가 아직 게시되지 않았을 수 있습니다."
        )
    text = resp.text
    if "Reported" not in text or "DATE" not in text:
        raise IntermagnetFetchError("응답이 IAGA-2002 형식이 아닙니다 - 관측소 코드나 날짜를 확인하세요.")
    return text
