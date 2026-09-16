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

from .despike import despike


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
        # Fill remaining slots by TRUE (header-derived) distance, nearest
        # first - not by probed's own iteration order, which just follows
        # _candidates_by_rough_distance's coarse country-centroid ranking
        # and can disagree with the real distance now on hand for every
        # candidate.
        for d, bearing, data in sorted(probed, key=lambda t: t[0]):
            if data.iaga_code in selected_codes:
                continue
            selected.append((d, bearing, data))
            selected_codes.add(data.iaga_code)
            if len(selected) >= n_stations:
                break
    return selected


_MINUTES_PER_DAY = 24 * 60
# Minutes two stations must share before one's level can be measured
# against the other's. An hour is long enough that the difference
# measured is the stations' levels rather than one disturbed stretch,
# and short enough that a station present for only part of a day
# still gets levelled instead of falling back to its own mean.
_MIN_STATION_OVERLAP_SAMPLES = 60
# Minutes either side of a station handover used to measure the step it
# produced. Half an hour is long enough to average out ordinary minute-to
# -minute variation and short enough that the real diurnal curve barely
# bends across it.
_HANDOVER_WINDOW_MINUTES = 30
_MIN_HANDOVER_SAMPLES = 10
# Below this a handover is not worth levelling - it is within what two
# observatories' curves ordinarily disagree by from one minute to the next.
_MIN_HANDOVER_STEP_NT = 1.0
# Real samples either side of a rebuilt day used to anchor it onto its
# neighbours, and the fewest that makes the measurement worth trusting.
_ANCHOR_WINDOW_MINUTES = 120
_MIN_ANCHOR_SAMPLES = 20
_MIN_DAY_COVERAGE = 0.5
# A station needing more than this share of its days estimated is not
# measuring the period in any useful sense - see fetch_observatory_span.
MAX_ESTIMATED_DAY_FRACTION = 0.34

# Despiking applied to each station before blending. Conservative: a
# 1-minute observatory series is smooth, so only a sample standing far out
# from its own neighbours is touched.
_STATION_DESPIKE_WINDOW = 11
_STATION_DESPIKE_K = 6.0  # below this fraction of samples present, treat the whole day as missing rather than gap-interpolate it


def _edge_offset(
    values: np.ndarray, template_at: np.ndarray, valid: np.ndarray, take_last: bool
) -> float | None:
    """How far real data sits above the template at one end of a run, from
    the `_ANCHOR_WINDOW_MINUTES` real samples closest to that end. None if
    there are too few to measure with."""
    idx = np.flatnonzero(valid)
    if len(idx) < _MIN_ANCHOR_SAMPLES:
        return None
    idx = idx[-_ANCHOR_WINDOW_MINUTES:] if take_last else idx[:_ANCHOR_WINDOW_MINUTES]
    return float(np.median(values[idx] - template_at[idx]))


def _anchor_offsets(
    values: np.ndarray,
    template: np.ndarray,
    valid: np.ndarray,
    day_of: pd.DatetimeIndex,
    minute_of_day: np.ndarray,
    d: pd.Timestamp,
    n_minutes: int,
) -> np.ndarray | None:
    """A per-minute offset that makes a templated day join the real data on
    both sides of it, ramped linearly from one edge to the other. None when
    the day is not bounded by real data on both sides.

    A synthesized day anchored by one constant offset joins whichever
    neighbour the offset was taken from and steps at its other end. That
    step is not a small matter downstream: the diurnal correction is
    drone - (base - reference), so a step in the base lands in the survey
    one for one, and because a day boundary is midnight it lands on any
    flight that starts at 00:00. Measured here on a station whose field
    drifts 4 nT/day: one day removed and rebuilt sat 4.00 nT off and left
    an 8.10 nT step at the following midnight.

    Continuity is therefore preferred over the day's own sparse samples
    when both are available. The day is an estimate either way - a couple
    of nT of level error in it is ordinary estimate error, while a step is
    an artifact that propagates into the anomaly undiminished.
    """
    before = (day_of < d)
    after = (day_of > d)
    if not (before.any() and after.any()):
        return None

    left = _edge_offset(values[before], template[minute_of_day[before]], valid[before], take_last=True)
    right = _edge_offset(values[after], template[minute_of_day[after]], valid[after], take_last=False)
    if left is None or right is None:
        return None
    return np.linspace(left, right, n_minutes)


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
        # Join the real data either side of this day where that is
        # possible, so the rebuilt day does not step at midnight - see
        # _anchor_offsets for why that is preferred over the day's own
        # sparse samples.
        offset = _anchor_offsets(values, template, valid, day_of, minute_of_day, d, int(day_mask.sum()))
        if offset is None and day_valid.any():
            offset = np.nanmedian(day_values[day_valid] - day_template[day_valid])
        elif offset is None:
            nearest_day = min(good_days, key=lambda gd: abs((gd - d).days))
            nearest_mask = day_of == nearest_day
            nearest_values = values[nearest_mask]
            nearest_template = template[minute_of_day[nearest_mask]]
            nearest_valid = valid[nearest_mask]
            offset = np.nanmedian(nearest_values[nearest_valid] - nearest_template[nearest_valid])

        day_filled = day_values.copy()
        to_fill = ~day_valid
        day_offset = offset[to_fill] if isinstance(offset, np.ndarray) else offset
        day_filled[to_fill] = day_template[to_fill] + day_offset
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


def fetch_observatory_span(
    iaga_code: str,
    start_date: date,
    end_date: date,
    timeout_seconds: float = 15.0,
    max_estimated_fraction: float = MAX_ESTIMATED_DAY_FRACTION,
) -> tuple[IagaObservatoryData, list[str]] | None:
    """One unbroken series per observatory, covering every calendar day
    from start_date to end_date, or None if this station cannot provide
    one.

    Fetching only the survey's own flight dates is cheaper, and it is what
    this used to do. But it leaves each station's coverage full of holes,
    and - the part that matters - a station that fails on one date and
    succeeds on the next *drops in and out of the blend*. The set of
    contributing stations then changes, the per-minute weights renormalise,
    and the blended series steps. Because observatory data is published one
    UT day at a time, that change lands at 00:00 UT.

    Which is 09:00 in Korea. Drone surveys fly mid-morning, so a UT day
    boundary falls in the middle of a flight rather than safely between
    them: on the 2026-09 HaeNam survey, four of the nine flight sessions
    crossed UT midnight, including the one whose block came out stepped
    against its neighbours on both sides.

    A real base station logging continuously across midnight has no step
    there, and a substitute for one should behave the same way. So each
    station is made continuous over the whole project period before any
    blending, and a station that cannot be made continuous is dropped
    entirely rather than being allowed to come and go. With every station
    spanning the whole period the active set cannot change, and the
    mechanism is gone rather than corrected for afterwards.

    Missing days inside the span are estimated against the whole span's
    template and joined to the real data on both sides (fill_missing_days),
    not against a three-day window. A station needing more than
    `max_estimated_fraction` of its days estimated is not a measurement of
    anything useful and is dropped.
    """
    days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    frames = []
    for d in days:
        try:
            frames.append(parse_iaga2002(
                fetch_iaga2002_text(iaga_code, d, days=1, timeout_seconds=timeout_seconds)))
        except (IntermagnetFetchError, IagaParseError):
            continue
    if not frames:
        return None
    header = frames[0]

    combined = pd.concat([f.df for f in frames], ignore_index=True)
    combined = (combined.dropna(subset=["mag"]).drop_duplicates(subset="timestamp")
                .sort_values("timestamp").reset_index(drop=True))
    if combined.empty:
        return None

    filled, estimated = fill_missing_days(combined, start_date, end_date)
    if len(estimated) > max_estimated_fraction * len(days):
        return None

    covered = {d for d in days if _day_coverage(filled, d) >= _MIN_DAY_COVERAGE}
    if len(covered) < len(days):
        return None

    return replace(header, df=filled), estimated


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
    dropped_codes: list[str] = []
    for _, _, probe_data in selected:
        if len(dates) == 1:
            station_data, estimated_dates = probe_data, []
        else:
            # The whole project period, unbroken - not just the flight
            # dates. See fetch_observatory_span for why a station that
            # comes and goes is worse than one that is absent throughout.
            span = fetch_observatory_span(
                probe_data.iaga_code, min(dates), max(dates), timeout_seconds=timeout_seconds
            )
            if span is None:
                dropped_codes.append(probe_data.iaga_code)
                continue
            station_data, estimated_dates = span
        if estimated_dates:
            estimated_dates_by_code[station_data.iaga_code] = estimated_dates
        result_stations.append(station_data)

    # Every station that got this far spans the whole period, so the set
    # contributing to the blend cannot change and there is nothing to step
    # at a UT day boundary. If none of them can - a short-lived station, a
    # date the network has not published yet - fall back to the old
    # per-date fetch rather than leaving the operator with no base at all,
    # and say so, because that result can carry the seams this exists to
    # avoid.
    if not result_stations:
        for _, _, probe_data in selected:
            try:
                station_data, estimated_dates = fetch_observatory_dates(
                    probe_data.iaga_code, dates, timeout_seconds=timeout_seconds
                )
            except IntermagnetFetchError:
                station_data, estimated_dates = probe_data, []
            if estimated_dates:
                estimated_dates_by_code[station_data.iaga_code] = estimated_dates
            result_stations.append(station_data)

    if not result_stations:
        raise IntermagnetFetchError("주변 관측소 자료를 하나도 받아오지 못했습니다.")
    return result_stations, estimated_dates_by_code


def _level_stations_onto_one_baseline(
    series_by_station: list[pd.Series], weights: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Put every station on a common level, measured where they overlap.

    This is what keeps the blended series continuous when the set of
    contributing stations changes. Referencing each station to its OWN
    mean instead leaves them at levels that differ by however much their
    diurnal curves differ over their own coverage - so the instant a
    station drops in or out and the weights renormalise, the blend steps
    by that difference.

    That is not hypothetical. Observatory data is fetched one calendar day
    at a time per station (fetch_observatory_dates), so a station's
    coverage begins and ends at midnight and the active set can only ever
    change there. The 2026-09 HaeNam base file built by the older code has
    exactly that signature: its 1-minute changes have a median of 0.13 nT
    and a 99.9th percentile of 2.12 nT, and it jumps 20.86 nT at
    2026-09-09 00:00:00 and 26.34 nT at 2026-08-03 00:00:00 - both at
    midnight, both far outside anything the field does in a minute.

    That step lands in the survey one for one. The diurnal correction is
    drone - (base - reference), so between two times the reference
    cancels exactly:

        corrected(t2) - corrected(t1)
            = [drone(t2) - drone(t1)] - [base(t2) - base(t1)]

    No choice of reference - per day, per flight, or one for the whole
    survey - can remove a step that is inside the base series itself. It
    has to not be there. On the HaeNam block that 20.86 nT base jump
    measured as a 20.86 nT level step between the blocks flown either side
    of that midnight.

    Levels are fixed to the highest-weighted (nearest) station where
    possible, chaining through an intermediate station for one that does
    not overlap it directly; a station overlapping nothing keeps its own
    mean, which is the old behaviour and the best available when there is
    no shared minute to measure against.
    """
    n = len(series_by_station)
    values = [s.to_numpy(dtype=float) for s in series_by_station]
    valid = [np.isfinite(v) for v in values]
    offsets = np.array([
        float(np.mean(v[m])) if m.any() else 0.0 for v, m in zip(values, valid)
    ])

    anchor = int(np.argmax(weights))
    resolved = {anchor}
    # Breadth-first over "shares enough minutes with something already
    # resolved", so a station reachable only through a third one still
    # gets a measured level rather than falling back to its own mean.
    progress = True
    while progress and len(resolved) < n:
        progress = False
        for i in range(n):
            if i in resolved:
                continue
            best_j, best_overlap = None, 0
            for j in resolved:
                shared = int((valid[i] & valid[j]).sum())
                if shared > best_overlap:
                    best_j, best_overlap = j, shared
            if best_j is None or best_overlap < _MIN_STATION_OVERLAP_SAMPLES:
                continue
            shared = valid[i] & valid[best_j]
            # median, not mean: robust to a spike or a short disturbed
            # stretch in either station over the minutes they share
            offsets[i] = float(np.median(
                values[i][shared] - (values[best_j][shared] - offsets[best_j])
            ))
            resolved.add(i)
            progress = True

    levelled = [v - off for v, off in zip(values, offsets)]
    return levelled, offsets


def _edge_value(values: np.ndarray, at_start: bool) -> float:
    """Where a straight line through `values` reaches the edge of the
    window - its start if `at_start`, otherwise just past its end."""
    n = len(values)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    return float(intercept + slope * (0.0 if at_start else n - 1 + 1.0))


def _remove_handover_steps(
    mag: np.ndarray, active: np.ndarray, window: int = _HANDOVER_WINDOW_MINUTES
) -> list[dict]:
    """Make the blend continuous where the set of contributing stations
    changes. Edits `mag` in place; returns what it did, per handover.

    Levelling the stations onto a common baseline
    (_level_stations_onto_one_baseline) removes the *constant* part of
    their disagreement, which is the large part - hundreds of nT, set by
    latitude. It cannot remove the rest: two observatories' Sq curves
    differ in amplitude and in phase (roughly an hour of phase per 15
    degrees of longitude), so at any given minute their levelled values
    still differ by something like ten nT. The moment the active set
    changes, the blend moves by that instantaneous difference, and a
    station's coverage begins and ends at midnight because the data is
    fetched a calendar day at a time - so the jump lands at midnight,
    which is where flights that start at 00:00 sit.

    There is no way to cross-fade out of it: at the minute a station's
    data stops there is nothing left of it to fade. But the step is known
    to be an artifact - the field did not move ten nT in a minute, the
    estimator changed its mind about who to listen to - and the absolute
    level of this series carries no meaning for the diurnal correction,
    which only ever uses (base - reference). So the series after each
    handover is shifted onto the level before it, measured as the
    difference of medians over `window` minutes either side.

    The one thing this can get wrong is a genuine rapid change that
    happens to coincide with a handover, which would be partly absorbed.
    That is the better trade: a sudden storm onset at exactly midnight is
    rare, the median window is half an hour wide, and the alternative is a
    twenty-nanotesla artifact going into the survey unannounced - measured
    at 20.86 nT on the 2026-09 HaeNam base, and landing in the anomaly as
    a 20.86 nT step between the blocks flown either side of it.
    """
    changes = [
        k for k in range(1, len(mag))
        if active[k] != active[k - 1] and np.isfinite(mag[k]) and np.isfinite(mag[k - 1])
    ]
    applied: list[dict] = []
    for k in changes:
        before = mag[max(0, k - window):k]
        after = mag[k:k + window]
        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]
        if len(before) < _MIN_HANDOVER_SAMPLES or len(after) < _MIN_HANDOVER_SAMPLES:
            continue
        # Fit a line on each side and read both at the break, rather than
        # differencing two medians: the curves either side have different
        # slopes, and a median sits half a window away from the boundary,
        # so a median difference measures the step plus half a window of
        # slope mismatch.
        step = float(_edge_value(after, at_start=True) - _edge_value(before, at_start=False))
        if abs(step) < _MIN_HANDOVER_STEP_NT:
            continue
        mag[k:] -= step
        applied.append({"index": k, "step_nt": step,
                        "before": active[k - 1], "after": active[k]})
    return applied


def estimate_base_from_observatories(
    stations: list[IagaObservatoryData], target_lat: float, target_lon: float, power: float = 2.0
) -> pd.DataFrame:
    """Combines multiple observatories' scalar total-field (F) series into
    one virtual base station series for (target_lat, target_lon), using
    inverse-distance weighting (weight proportional to 1/distance**power,
    normalized to sum to 1 - the standard IDW spatial interpolation
    scheme).

    The per-timestamp weights are not fixed: whichever stations have data
    at a given minute get their weights renormalised to sum to 1 at that
    minute, so the effective weighting changes as stations drop in and out
    of coverage. Blending raw absolute F values under a changing weight
    set would make the combined series jump by however much those
    stations' absolute levels differ - tens of thousands of nT, since that
    is dominated by latitude/IGRF main-field strength rather than by
    anything the field is doing that day.

    So the stations are first put on one common level, measured over the
    minutes they share (_level_stations_onto_one_baseline), and that level
    is added back once at the end. Measuring it from the overlap rather
    than from each station's own mean is what makes the blend continuous
    across a change in the active set - see that function for the real
    base file this exists because of, and for why no choice of diurnal
    reference can repair such a step afterwards.

    The absolute level of the result carries no meaning for the diurnal
    correction, which only ever uses (base - reference); it is set to the
    nearest contributing station's own level so the series still reads as
    a plausible field strength.
    """
    if not stations:
        raise IntermagnetFetchError("결합할 관측소 자료가 없습니다.")

    weights = np.array([1.0 / max(haversine_km(target_lat, target_lon, s.lat, s.lon), 1.0) ** power for s in stations])
    weights = weights / weights.sum()

    start = min(s.df["timestamp"].min() for s in stations)
    end = max(s.df["timestamp"].max() for s in stations)
    grid = pd.date_range(start, end, freq="1min")

    series_by_station = []
    for s in stations:
        series = s.df.drop_duplicates(subset="timestamp").set_index("timestamp")["mag"].reindex(grid)
        # Interpolate only small internal gaps (a station's own brief
        # dropouts) - never extrapolate past a station's real coverage.
        series = series.interpolate(limit=5, limit_area="inside")
        # Despike each station before it is blended, not just the blend
        # afterwards. The nearest station carries most of the weight, so
        # one of its spikes arrives nearly undiluted - Cheongyang jumped
        # 39 nT for a single minute on 2026-09-08 and put a 33 nT spike in
        # the combined series - and a spike also corrupts the levels and
        # handover steps measured from the minutes around it.
        values = series.to_numpy(dtype=float)
        ok = np.isfinite(values)
        if ok.sum() > _STATION_DESPIKE_WINDOW:
            cleaned, _ = despike(values[ok], window_size=_STATION_DESPIKE_WINDOW,
                                 threshold_k=_STATION_DESPIKE_K, adaptive=False)
            values[ok] = cleaned
            series = pd.Series(values, index=series.index)
        series_by_station.append(series)

    levelled, offsets = _level_stations_onto_one_baseline(series_by_station, weights)
    common_level = float(offsets[int(np.argmax(weights))])

    weighted_sum = np.zeros(len(grid))
    weight_total = np.zeros(len(grid))
    for w, deviation in zip(weights, levelled):
        ok = np.isfinite(deviation)
        weighted_sum[ok] += w * deviation[ok]
        weight_total[ok] += w

    has_data = weight_total > 0
    mag = np.full(len(grid), np.nan)
    mag[has_data] = common_level + weighted_sum[has_data] / weight_total[has_data]

    # Which stations actually contributed to each minute. A change here is
    # the only thing that can step the blend once the stations share a
    # level - see _remove_handover_steps.
    codes = [s.iaga_code for s in stations]
    present = np.array([np.isfinite(dev) for dev in levelled])       # (n_stations, n_minutes)
    active = np.array([
        ",".join(c for c, on in zip(codes, present[:, i]) if on) or "-"
        for i in range(len(grid))
    ])
    handovers = _remove_handover_steps(mag, active)

    out = pd.DataFrame({"timestamp": grid, "mag": mag}).dropna(subset=["mag"]).reset_index(drop=True)
    out.attrs["station_handovers"] = [
        {"timestamp": str(grid[h["index"]]), "step_nt": h["step_nt"],
         "stations_before": h["before"], "stations_after": h["after"]}
        for h in handovers
    ]
    if out.empty:
        raise IntermagnetFetchError("선택된 관측소들의 자료가 겹치는 시간대가 없습니다.")
    return out
