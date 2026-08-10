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
from dataclasses import dataclass, replace
from datetime import date, timedelta

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
        "format": "IAGA2002",
        "testObsys": "0",
        "observatoryIagaCode": iaga_code.upper(),
        # Total samples per UTC day, not a word - the GIN service silently
        # rejects/misinterprets a non-numeric value here. 1440 = one-minute
        # cadence (60*24), the standard INTERMAGNET publication resolution
        # and plenty fine for diurnal correction (confirmed against a
        # known-working third-party GIN client's exact request string,
        # which uses 86400 for 1-second data - 1440 is the minute-cadence
        # equivalent of that same numeric convention).
        "samplesPerDay": "1440",
        "publicationState": "adj-or-rep",
        "recordTermination": "UNIX",
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


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (0-360, 0=north) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _candidates_by_rough_distance(target_lat: float, target_lon: float, limit: int):
    from .intermagnet_stations import COUNTRY_CENTROIDS, OBSERVATORY_ROSTER

    ranked = []
    for entry in OBSERVATORY_ROSTER:
        centroid = COUNTRY_CENTROIDS.get(entry.country)
        if centroid is None:
            continue
        d = haversine_km(target_lat, target_lon, centroid[0], centroid[1])
        ranked.append((d, entry))
    ranked.sort(key=lambda t: t[0])
    return [entry for _, entry in ranked[:limit]]


def _pick_direction_diverse(
    probed: list[tuple[float, float, "IagaObservatoryData"]], n_stations: int
) -> list[tuple[float, float, "IagaObservatoryData"]]:
    """Picks up to n_stations, preferring the nearest station in each
    unclaimed 90-degree compass quadrant (N/E/S/W) around the target
    first, then fills any remaining slots with the next-nearest stations
    overall. A handful of stations clustered on one side of the target
    would otherwise dominate a plain nearest-N pick, giving a
    directionally lopsided (and so less trustworthy) interpolation."""
    by_quadrant: dict[int, tuple[float, float, "IagaObservatoryData"]] = {}
    for d, bearing, data in probed:
        q = int(((bearing + 45) % 360) // 90)  # 0=N, 1=E, 2=S, 3=W
        if q not in by_quadrant or d < by_quadrant[q][0]:
            by_quadrant[q] = (d, bearing, data)

    selected = sorted(by_quadrant.values(), key=lambda t: t[0])[:n_stations]
    selected_codes = {data.iaga_code for _, _, data in selected}
    if len(selected) < n_stations:
        for d, bearing, data in probed:
            if data.iaga_code in selected_codes:
                continue
            selected.append((d, bearing, data))
            selected_codes.add(data.iaga_code)
            if len(selected) >= n_stations:
                break
    return selected


_MINUTES_PER_DAY = 24 * 60
_MIN_DAY_COVERAGE = 0.5  # below this fraction of samples present, treat the whole day as missing rather than gap-interpolate it


def fill_missing_days(df: pd.DataFrame, start_date: date, end_date: date) -> tuple[pd.DataFrame, list[str]]:
    """Estimates any calendar day within [start_date, end_date] that's
    entirely or mostly missing from an observatory's downloaded data - a
    common gap for BGS GIN feeds, since the most recent day or two often
    isn't published yet (definitive/adjusted data lags real time), and any
    single day here or there may simply be absent for other reasons.

    The estimate is a "typical day" template: for each minute-of-day, the
    median value across whichever OTHER days in this same request DO have
    data. That template is then anchored (shifted by a constant offset) to
    match the closest real data available for the missing day - its own
    sparse samples if it has any, otherwise the nearest good day's own
    offset from the template - so the filled-in day doesn't jump away from
    real neighboring values. This is a best-effort fallback for an
    unattended pipeline, not a scientific reconstruction of what the field
    actually did that day; callers should disclose which dates were
    estimated this way (the second return value) rather than silently
    treating them as measured.

    Interior gaps of a few minutes within an otherwise-present day are
    left to the existing small-gap linear interpolation (see
    estimate_base_from_observatories) - only whole/mostly-missing days are
    templated here."""
    if df.empty:
        return df, []

    grid = pd.date_range(start_date, pd.Timestamp(end_date) + pd.Timedelta(days=1), freq="1min", inclusive="left")
    series = df.drop_duplicates(subset="timestamp").set_index("timestamp")["mag"].reindex(grid)
    values = series.to_numpy(dtype=float)

    day_of = grid.normalize()
    minute_of_day = (grid.hour * 60 + grid.minute).to_numpy()
    valid = ~np.isnan(values)

    unique_days = [pd.Timestamp(d) for d in pd.unique(day_of)]
    coverage = {d: valid[day_of == d].mean() if (day_of == d).any() else 0.0 for d in unique_days}
    good_days = [d for d, c in coverage.items() if c >= _MIN_DAY_COVERAGE]
    bad_days = [d for d in unique_days if d not in good_days]
    if not good_days or not bad_days:
        return df, []

    good_mask = day_of.isin(good_days) & valid
    template = np.full(_MINUTES_PER_DAY, np.nan)
    for m in range(_MINUTES_PER_DAY):
        sel = good_mask & (minute_of_day == m)
        if sel.any():
            template[m] = np.nanmedian(values[sel])
    template = pd.Series(template).interpolate(limit_direction="both").to_numpy()

    filled = values.copy()
    filled_dates: list[str] = []
    for d in bad_days:
        day_mask = day_of == d
        day_values = values[day_mask]
        day_minutes = minute_of_day[day_mask]
        day_template = template[day_minutes]
        day_valid = valid[day_mask]
        if day_valid.any():
            offset = np.nanmedian(day_values[day_valid] - day_template[day_valid])
        else:
            nearest_day = min(good_days, key=lambda gd: abs((gd - d).days))
            nearest_mask = day_of == nearest_day
            nearest_values = values[nearest_mask]
            nearest_template = template[minute_of_day[nearest_mask]]
            nearest_valid = valid[nearest_mask]
            offset = np.nanmedian(nearest_values[nearest_valid] - nearest_template[nearest_valid])

        day_filled = day_values.copy()
        to_fill = ~day_valid
        day_filled[to_fill] = day_template[to_fill] + offset
        filled[day_mask] = day_filled
        filled_dates.append(str(d.date()))

    out_series = pd.Series(filled, index=grid).interpolate(limit=5, limit_area="inside")
    out = pd.DataFrame({"timestamp": out_series.index, "mag": out_series.to_numpy()}).dropna(subset=["mag"]).reset_index(drop=True)
    return out, filled_dates


def _day_coverage(df: pd.DataFrame, d: date) -> float:
    """Fraction of a single calendar day's expected 1-minute samples that
    are actually present in df."""
    if df.empty:
        return 0.0
    grid = pd.date_range(d, periods=_MINUTES_PER_DAY, freq="1min")
    series = df.drop_duplicates(subset="timestamp").set_index("timestamp")["mag"].reindex(grid)
    return float(series.notna().mean())


_EMPTY_MAG_DF = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "mag": pd.Series(dtype="float64")})


def fetch_observatory_dates(
    iaga_code: str, dates: list[date], timeout_seconds: float = 15.0
) -> tuple[IagaObservatoryData, list[str]]:
    """Downloads exactly the calendar dates a survey actually needs - one
    IAGA-2002 fetch per requested date - rather than the full inclusive
    span between its earliest and latest flight date, which for a survey
    flown on a handful of separate days weeks apart would mostly be empty
    padding (see Project._survey_dates).

    Any requested date that comes back missing or mostly missing is
    estimated from its own immediate day-before/day-after neighbors alone
    (fetched specially just for that one date's estimate, via
    fill_missing_days scoped to that 3-day window) - never from unrelated
    data elsewhere in the survey's date span. A date with neither real nor
    estimable data is silently dropped from the result (matching
    estimate_base_from_observatories's existing behavior for any gap).

    Returns the observatory's data (station metadata taken from whichever
    requested date's fetch succeeded first) restricted to just the
    requested dates, plus the list of dates that had to be estimated."""
    dates = sorted(set(dates))
    day_data: dict[date, IagaObservatoryData | None] = {}

    def _fetch_day(d: date) -> IagaObservatoryData | None:
        if d in day_data:
            return day_data[d]
        try:
            text = fetch_iaga2002_text(iaga_code, d, days=1, timeout_seconds=timeout_seconds)
            day_data[d] = parse_iaga2002(text)
        except (IntermagnetFetchError, IagaParseError):
            day_data[d] = None
        return day_data[d]

    for d in dates:
        _fetch_day(d)
    header = next((v for v in day_data.values() if v is not None), None)

    result_frames = []
    estimated_dates: list[str] = []
    for d in dates:
        data = day_data[d]
        df = data.df if data is not None else _EMPTY_MAG_DF
        if _day_coverage(df, d) >= _MIN_DAY_COVERAGE:
            result_frames.append(df)
            continue

        neighbor_frames = [df]
        for neighbor in (d - timedelta(days=1), d + timedelta(days=1)):
            neighbor_data = _fetch_day(neighbor)
            if header is None and neighbor_data is not None:
                header = neighbor_data
            neighbor_frames.append(neighbor_data.df if neighbor_data is not None else _EMPTY_MAG_DF)

        local_df = pd.concat(neighbor_frames, ignore_index=True)
        filled, local_filled_dates = fill_missing_days(local_df, d - timedelta(days=1), d + timedelta(days=1))
        day_result = filled[filled["timestamp"].dt.date == d]
        if not day_result.empty:
            result_frames.append(day_result)
            # only count it as an estimate if fill_missing_days actually
            # synthesized values for d - a below-threshold day that
            # couldn't be improved (e.g. no usable neighbor either) still
            # keeps whatever sparse real samples it had, unchanged
            if str(d) in local_filled_dates:
                estimated_dates.append(str(d))

    if header is None:
        raise IntermagnetFetchError(f"{iaga_code} 관측소의 요청한 날짜 자료를 하나도 받아오지 못했습니다.")

    combined = pd.concat(result_frames, ignore_index=True) if result_frames else _EMPTY_MAG_DF
    combined = (
        combined.dropna(subset=["mag"]).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    )
    if combined.empty:
        raise IntermagnetFetchError(f"{iaga_code} 관측소의 요청한 날짜 자료를 하나도 받아오지 못했습니다.")

    return replace(header, df=combined), estimated_dates


DEFAULT_MAX_STATION_DISTANCE_KM = 2000.0
"""Recommended cutoff for how far an INTERMAGNET observatory can be from the
survey area and still be trusted for diurnal correction. The IDW combine
weight (~1/distance**2, see estimate_base_from_observatories) already makes
distant stations nearly irrelevant most of the time, but on exactly the days
a *nearby* station has a gap, a distant substitute's own signal is used
almost undiluted for that stretch - and a very distant station's diurnal
(Sq) curve isn't just a weaker copy of the local one, it can be shifted
in both phase and amplitude: Sq peaks near local solar noon, so an east-west
separation shows up as a peak-time offset (~1h per ~15 degrees of longitude,
i.e. roughly one hour of shift per ~1300km at mid-latitudes), and Sq
amplitude itself depends strongly on geomagnetic latitude (auroral/
equatorial electrojet zones a swing away). 2000km keeps that phase/latitude
drift modest for typical mid-latitude surveys while still reaching most of a
region's INTERMAGNET network; it is a default, not a hard physical
threshold - callers can loosen or disable it (max_distance_km=None)."""


def select_nearest_observatories(
    target_lat: float,
    target_lon: float,
    dates: list[date],
    n_stations: int = 4,
    max_candidates: int = 20,
    timeout_seconds: float = 15.0,
    max_distance_km: float | None = DEFAULT_MAX_STATION_DISTANCE_KM,
) -> tuple[list[IagaObservatoryData], dict[str, list[str]]]:
    """Finds up to n_stations INTERMAGNET observatories spread around
    (target_lat, target_lon), for combining into a substitute base series
    when no local base station was measured at all (see
    estimate_base_from_observatories).

    Rather than relying on a hand-maintained (and inevitably imprecise or
    stale) table of station coordinates, this narrows the full ~130-station
    roster to the max_candidates most plausible ones by a coarse
    country-centroid distance (see intermagnet_stations.py), then actually
    fetches each candidate's real data for a single day - just enough to
    read coordinates and rank candidates, cheap even with many candidates
    - and reads its authoritative coordinates straight from that file's
    own IAGA-2002 header - so the final selection and every distance
    reported to the user is always based on real, current observatory
    metadata, never a guess. It tries dates[0] first and only falls
    through to later dates in `dates` if literally none of the candidates
    published anything that day (e.g. the survey's earliest date happens
    to be one none of them have posted yet); candidates with no data for
    whichever probe date succeeds are silently skipped.

    Once the n_stations winners are picked, each is re-fetched for every
    date in `dates` via fetch_observatory_dates() (only the selected few,
    not every candidate) - dates is typically the survey's own distinct
    flight dates (see Project._survey_dates), not a padded calendar
    range, so a survey flown on a handful of separate days weeks apart
    only ever downloads those exact days. Any requested date that comes
    back missing is estimated from its own immediate neighbors - see
    fetch_observatory_dates's docstring. Returns both the station data and
    a {iaga_code: [estimated ISO dates]} map so callers can disclose
    exactly which dates in the result are real measurements vs.
    best-effort estimates.

    This makes up to max_candidates + a handful of requests per selected
    station live outbound HTTPS requests - see fetch_iaga2002_text's
    docstring on network reachability.

    max_distance_km (default DEFAULT_MAX_STATION_DISTANCE_KM) drops any
    candidate farther than that from (target_lat, target_lon) BEFORE the
    direction-diverse pick runs, so a quadrant with no truly nearby
    observatory simply contributes fewer than n_stations rather than
    reaching arbitrarily far to fill its slot - i.e. n_stations is a
    ceiling, not a guaranteed count, whenever the cutoff binds. Pass
    max_distance_km=None to disable the cutoff and restore the old
    reach-as-far-as-needed behavior."""
    candidates = _candidates_by_rough_distance(target_lat, target_lon, max_candidates)
    probed: list[tuple[float, float, IagaObservatoryData]] = []
    for probe_date in dates:
        for entry in candidates:
            try:
                text = fetch_iaga2002_text(entry.iaga_code, probe_date, days=1, timeout_seconds=timeout_seconds)
                data = parse_iaga2002(text)
            except (IntermagnetFetchError, IagaParseError):
                continue
            if data.lat is None or data.lon is None:
                continue
            d = haversine_km(target_lat, target_lon, data.lat, data.lon)
            if max_distance_km is not None and d > max_distance_km:
                continue
            bearing = _bearing_deg(target_lat, target_lon, data.lat, data.lon)
            probed.append((d, bearing, data))
        if probed:
            break  # found real coordinates for at least one candidate on this date - no need to try later dates too

    if not probed:
        distance_note = (
            f" (최대 거리 {max_distance_km:.0f}km 이내 관측소가 없었을 수 있습니다)" if max_distance_km is not None else ""
        )
        raise IntermagnetFetchError(
            "근처 INTERMAGNET 관측소 자료를 하나도 받아오지 못했습니다 (네트워크 접근이 막혀 있거나, "
            f"해당 날짜 자료가 아직 게시되지 않았을 수 있습니다){distance_note}."
        )

    selected = _pick_direction_diverse(probed, n_stations)
    selected.sort(key=lambda t: t[0])

    result_stations: list[IagaObservatoryData] = []
    estimated_dates_by_code: dict[str, list[str]] = {}
    for _, _, probe_data in selected:
        if len(dates) == 1:
            station_data = probe_data  # already have exactly what's needed
        else:
            try:
                station_data, estimated_dates = fetch_observatory_dates(
                    probe_data.iaga_code, dates, timeout_seconds=timeout_seconds
                )
            except IntermagnetFetchError:
                # fall back to the single already-probed day rather than
                # dropping a station that was reachable a moment ago
                station_data, estimated_dates = probe_data, []
            if estimated_dates:
                estimated_dates_by_code[station_data.iaga_code] = estimated_dates
        result_stations.append(station_data)

    return result_stations, estimated_dates_by_code


def estimate_base_from_observatories(
    stations: list[IagaObservatoryData], target_lat: float, target_lon: float, power: float = 2.0
) -> pd.DataFrame:
    """Combines multiple observatories' scalar total-field (F) series into
    one virtual base station series for (target_lat, target_lon), using
    inverse-distance weighting (weight proportional to 1/distance**power,
    normalized to sum to 1 - the standard IDW spatial interpolation
    scheme).

    Weighting the raw F values directly (rather than each station's
    deviation from its own daily mean) is deliberate and equivalent for
    this app's purposes: diurnal correction (processing/diurnal.py) only
    ever uses (base_interp - reference_within_flight_window), i.e. the
    *variation* relative to the survey window's own mean. Since IDW is a
    linear combination with fixed (time-independent) weights,
    weighted_avg(F_a, F_b) - mean(weighted_avg(F_a, F_b)) equals
    weighted_avg(F_a - mean(F_a), F_b - mean(F_b)) - so the arbitrary
    absolute-level differences between stations at different latitudes
    cancel out downstream exactly as they would for a single real station,
    without needing to separately detrend each series here first."""
    if not stations:
        raise IntermagnetFetchError("결합할 관측소 자료가 없습니다.")

    weights = np.array([1.0 / max(haversine_km(target_lat, target_lon, s.lat, s.lon), 1.0) ** power for s in stations])
    weights = weights / weights.sum()

    start = min(s.df["timestamp"].min() for s in stations)
    end = max(s.df["timestamp"].max() for s in stations)
    grid = pd.date_range(start, end, freq="1min")

    weighted_sum = np.zeros(len(grid))
    weight_total = np.zeros(len(grid))
    for w, s in zip(weights, stations):
        series = s.df.drop_duplicates(subset="timestamp").set_index("timestamp")["mag"].reindex(grid)
        # Interpolate only small internal gaps (a station's own brief
        # dropouts) - never extrapolate past a station's real coverage.
        series = series.interpolate(limit=5, limit_area="inside")
        valid = series.notna().to_numpy()
        weighted_sum[valid] += w * series.to_numpy()[valid]
        weight_total[valid] += w

    has_data = weight_total > 0
    mag = np.full(len(grid), np.nan)
    mag[has_data] = weighted_sum[has_data] / weight_total[has_data]

    out = pd.DataFrame({"timestamp": grid, "mag": mag}).dropna(subset=["mag"]).reset_index(drop=True)
    if out.empty:
        raise IntermagnetFetchError("선택된 관측소들의 자료가 겹치는 시간대가 없습니다.")
    return out
