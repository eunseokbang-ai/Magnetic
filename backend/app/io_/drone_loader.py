"""Drone magnetic survey CSV/ASC loader.

Handles the common logger export shape: one row per sample with
Date/Time/Latitude/Longitude/Mag plus a mix of GNSS/IMU fields where some
columns (Altitude, Hdop, ...) are only populated on GGA fix rows. This
"generic" schema also covers Geometrics MagArrow exports (both the
Original and post-processed/Filtered variants), since their column names
match it directly.

In addition, the file header is sniffed to auto-detect three other
commercial magnetometer export formats (mirroring DroneMagAdv's supported
device list) and normalize them to the same internal schema:
- MicroInfinity Mag (22-column CSV with MagField_J1/J2[uT] + TOW)
- SENSYS MagDrone R1 (CSV with a precomputed TMI column)
- SENSYS MagDrone R3 raw log (semicolon-delimited body after a text header
  block) and its "R3 ASC" first-stage-processed export (whitespace
  columns, dual-probe rows tagged by Sensor ID)
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Time", "Latitude", "Longitude", "Mag"]
SPARSE_NUMERIC_COLUMNS = ["Altitude", "HeightOverEllipsoid", "Hdop", "SpeedOverGround"]

# MicroInfinity Mag's MagField_Jn columns are in microtesla; internally
# every other loader/downstream step works in nT.
_UT_TO_NT = 1000.0


class DroneLoadError(ValueError):
    pass


def _read_text(path_or_buffer) -> str:
    if hasattr(path_or_buffer, "read"):
        raw = path_or_buffer.read()
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
    else:
        with open(path_or_buffer, "rb") as f:
            raw = f.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8-sig", errors="replace")
    return raw


def _detect_format(text: str) -> str:
    """Sniff the file header to pick a parser. Order matters: the more
    distinctive/rare signatures are checked first so a coincidental
    substring match in a generic export can't shadow them."""
    head = text[:4000]
    first_line = head.splitlines()[0].strip() if head else ""
    if re.match(r"^\d{8}_\d{6}_MD-R3", first_line) or "MagDroneR3:" in head:
        return "sensys_r3_raw"
    if "Total field anomaly" in head and "Sensor ID" in head:
        return "sensys_r3_asc"
    if "MagField_J1" in head and "MagField_J2" in head:
        return "microinfinity"
    if "Next WP" in head and re.search(r"\bTMI\b", head):
        return "sensys_r1"
    return "generic"


def _parse_generic(text: str) -> pd.DataFrame:
    # GgaSentence/RmcSentence fields embed commas inside single-quoted
    # strings, and every data row carries one trailing empty field from a
    # final comma; quotechar+index_col=False keep the C parser from
    # mis-aligning columns or inferring a spurious index column.
    df = pd.read_csv(
        io.StringIO(text),
        skipinitialspace=True,
        quotechar="'",
        index_col=False,
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DroneLoadError(f"드론 파일에 필수 컬럼이 없습니다: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )
    for col in ["Latitude", "Longitude", "Mag"] + SPARSE_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "MagValid" in df.columns:
        df["MagValid"] = pd.to_numeric(df["MagValid"], errors="coerce")
        df = df[(df["MagValid"].isna()) | (df["MagValid"] != 0)]

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "lat": df["Latitude"],
            "lon": df["Longitude"],
            "mag_raw": df["Mag"],
            "altitude_msl_m": df["Altitude"] if "Altitude" in df.columns else np.nan,
            "geoid_separation_m": df["HeightOverEllipsoid"] if "HeightOverEllipsoid" in df.columns else np.nan,
            "speed_over_ground": df["SpeedOverGround"] if "SpeedOverGround" in df.columns else np.nan,
        }
    )


def _parse_sensys_r1(text: str) -> pd.DataFrame:
    """SENSYS MagDrone R1: CSV with a header row and a precomputed TMI
    (total magnetic intensity) column - used directly as mag_raw rather
    than reconstructed from Bx/By/Bz, matching what the instrument itself
    reports."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True, index_col=False, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = ["Date", "Time", "Latitude", "Longitude", "TMI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DroneLoadError(f"SENSYS MagDrone R1 파일에 필수 컬럼이 없습니다: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )
    for col in ["Latitude", "Longitude", "TMI", "Altitude", "Altitude AGL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "lat": df["Latitude"],
            "lon": df["Longitude"],
            "mag_raw": df["TMI"],
            "altitude_msl_m": df["Altitude"] if "Altitude" in df.columns else np.nan,
            "geoid_separation_m": np.nan,
            "speed_over_ground": np.nan,
        }
    )


def _parse_microinfinity(text: str) -> pd.DataFrame:
    """MicroInfinity Mag: 22-column CSV, dual-channel (MagField_J1/J2[uT])
    with a GPS time-of-week (TOW, seconds) column instead of a calendar
    date/time pair.

    Uses channel 1 alone (MagField_J1) as mag_raw, matching DroneMagAdv's
    default "Use Channel" setting. Since the format carries no calendar
    date, a placeholder reference date (the processing day) is used to
    build timestamps - inter-sample spacing (from TOW) is exact, but the
    absolute date is only a best-effort estimate, same as DroneMagAdv's
    own "infer from file modification time" fallback for this format.
    This means diurnal correction against a base-station log recorded on
    a different calendar day will not line up in absolute time.
    """
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True, index_col=False, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = ["MagField_J1[uT]", "TOW", "lat[deg]", "lon[deg]"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DroneLoadError(f"MicroInfinity Mag 파일에 필수 컬럼이 없습니다: {missing}")

    for col in ["MagField_J1[uT]", "TOW", "lat[deg]", "lon[deg]", "alt[m]"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ref_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    timestamp = ref_date + pd.to_timedelta(df["TOW"], unit="s")

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "lat": df["lat[deg]"],
            "lon": df["lon[deg]"],
            "mag_raw": df["MagField_J1[uT]"] * _UT_TO_NT,
            "altitude_msl_m": df["alt[m]"] if "alt[m]" in df.columns else np.nan,
            "geoid_separation_m": np.nan,
            "speed_over_ground": np.nan,
        }
    )


_R3_DATE_RE = re.compile(r"^Date:\s*(\d{2})\.(\d{2})\.(\d{4})", re.MULTILINE)
_R3_TIME_RE = re.compile(r"^Time:\s*(\d{1,2}):(\d{2}):(\d{2})", re.MULTILINE)


def _parse_sensys_r3_raw(text: str) -> pd.DataFrame:
    """SENSYS MagDrone R3 raw log: a text metadata header (Date:/Time:/
    Samples:/... lines) followed by a semicolon-delimited data block with
    two magnetometer probes (B1x/y/z, B2x/y/z, nT) and low-rate GPS fields
    (Latitude/Longitude/Altitude logged as 0 on samples between GPS
    updates - a much sparser update rate than the >=10 Hz magnetic
    sampling rate).

    Probe 1's total field magnitude (sqrt(B1x^2+B1y^2+B1z^2)) is used as
    mag_raw; position/altitude are linearly interpolated over time to fill
    the gaps between GPS updates.
    """
    date_m = _R3_DATE_RE.search(text)
    time_m = _R3_TIME_RE.search(text)
    if not date_m or not time_m:
        raise DroneLoadError("SENSYS MagDrone R3 파일에서 시작 날짜/시각(Date:/Time: 헤더)을 찾을 수 없습니다.")
    dd, mm, yyyy = date_m.groups()
    hh, mi, ss = time_m.groups()
    start = pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd), hour=int(hh), minute=int(mi), second=int(ss))

    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("Timestamp [ms]")), None)
    if header_idx is None:
        raise DroneLoadError("SENSYS MagDrone R3 파일에서 데이터 헤더(Timestamp [ms];...)를 찾을 수 없습니다.")

    header = [c.strip() for c in lines[header_idx].split(";") if c.strip()]
    required = ["Timestamp [ms]", "B1x [nT]", "B1y [nT]", "B1z [nT]", "Latitude [Decimal Degrees]", "Longitude [Decimal Degrees]"]
    missing = [c for c in required if c not in header]
    if missing:
        raise DroneLoadError(f"SENSYS MagDrone R3 파일의 데이터 헤더에 필수 컬럼이 없습니다: {missing}")

    data_lines = [ln.rstrip(";").strip() for ln in lines[header_idx + 1 :] if ln.strip()]
    if not data_lines:
        raise DroneLoadError("SENSYS MagDrone R3 파일에 데이터 행이 없습니다.")
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=";", names=header, index_col=False, engine="python")
    for col in header:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Timestamp [ms]"])

    timestamp = start + pd.to_timedelta(df["Timestamp [ms]"], unit="ms")

    mag_raw = np.sqrt(df["B1x [nT]"] ** 2 + df["B1y [nT]"] ** 2 + df["B1z [nT]"] ** 2)

    lat = df["Latitude [Decimal Degrees]"]
    lon = df["Longitude [Decimal Degrees]"]
    alt = df["Altitude [m]"] if "Altitude [m]" in df.columns else pd.Series(np.nan, index=df.index)
    # rows without a GPS fix yet log 0/0 rather than leaving the field blank
    no_fix = (lat == 0) & (lon == 0)
    lat = lat.mask(no_fix)
    lon = lon.mask(no_fix)
    alt = alt.mask(no_fix)

    out = pd.DataFrame(
        {
            "timestamp": timestamp.to_numpy(),
            "lat": lat.to_numpy(),
            "lon": lon.to_numpy(),
            "mag_raw": mag_raw.to_numpy(),
            "altitude_msl_m": alt.to_numpy(),
            "geoid_separation_m": np.nan,
            "speed_over_ground": np.nan,
        }
    )
    # lat/lon/altitude are only logged at the (much slower) GPS update
    # rate - interpolate over time so every high-rate magnetic sample has
    # a usable position, the same treatment as the generic loader's
    # GGA-only sparse fields.
    for col in ["lat", "lon", "altitude_msl_m"]:
        if out[col].notna().any():
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
    return out


_R3_ASC_COLUMNS = ["timestamp_ms", "sensor_id", "lat", "lon", "total_field_nt", "mag_x_nt", "mag_y_nt", "mag_z_nt"]
_R3_ASC_DATA_ROW_RE = re.compile(r"^\s*\d+\s+\d+\s")


def _parse_sensys_r3_asc(text: str) -> pd.DataFrame:
    """SENSYS MagDrone R3 ASC: whitespace-delimited first-stage export
    with a fixed 8-field layout (Timestamp[ms], Sensor ID, Latitude,
    Longitude, Total field anomaly[nT], Mag-X/Y/Z[nT]). A dual-probe unit
    logs one full timeline per Sensor ID back to back in the same file;
    only the first Sensor ID encountered is used as the primary channel.

    Like MicroInfinity Mag, this format has no calendar date column, so
    timestamps are built from a placeholder reference date (see
    _parse_microinfinity for the same caveat).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    data_lines = [ln for ln in lines if _R3_ASC_DATA_ROW_RE.match(ln)]
    if not data_lines:
        raise DroneLoadError("SENSYS MagDrone R3 ASC 파일에서 데이터 행을 찾을 수 없습니다.")

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)), sep=r"\s+", engine="python", header=None, names=_R3_ASC_COLUMNS
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["sensor_id"])
    if df.empty:
        raise DroneLoadError("SENSYS MagDrone R3 ASC 파일에서 유효한 데이터 행을 찾을 수 없습니다.")
    primary_id = df["sensor_id"].iloc[0]
    df = df[df["sensor_id"] == primary_id]

    ref_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    timestamp = ref_date + pd.to_timedelta(df["timestamp_ms"], unit="ms")

    return pd.DataFrame(
        {
            "timestamp": timestamp.to_numpy(),
            "lat": df["lat"].to_numpy(),
            "lon": df["lon"].to_numpy(),
            "mag_raw": df["total_field_nt"].to_numpy(),
            "altitude_msl_m": np.nan,
            "geoid_separation_m": np.nan,
            "speed_over_ground": np.nan,
        }
    )


_FORMAT_PARSERS = {
    "sensys_r3_raw": _parse_sensys_r3_raw,
    "sensys_r3_asc": _parse_sensys_r3_asc,
    "microinfinity": _parse_microinfinity,
    "sensys_r1": _parse_sensys_r1,
    "generic": _parse_generic,
}


def _finalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing applied after any format-specific parser:
    numeric coercion, essential-field/coordinate validation, time
    ordering, sparse-altitude interpolation, and ellipsoidal-height
    reconstruction. See load_drone_csv for the ellipsoidal-height note.
    """
    df = raw.copy()
    for col in ["lat", "lon", "mag_raw", "altitude_msl_m", "geoid_separation_m", "speed_over_ground"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp", "lat", "lon", "mag_raw"])
    if df.empty:
        raise DroneLoadError("유효한 위치/자력 데이터가 없습니다.")

    # Reject rows with out-of-range coordinates and the (0, 0) "GPS fix
    # never acquired" sentinel some loggers write instead of leaving the
    # field blank - both are junk, not real positions.
    n_before_coord = len(df)
    valid_coord = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & ~((df["lat"] == 0) & (df["lon"] == 0))
    )
    df = df[valid_coord]
    n_invalid_coords_removed = n_before_coord - len(df)
    if df.empty:
        raise DroneLoadError("유효한 위치/자력 데이터가 없습니다 (좌표가 모두 비정상입니다).")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # GGA-only fields (Altitude, Hdop, ...) are sparse in several source
    # formats; interpolate over time so every sample has a usable value.
    for col in ["altitude_msl_m", "geoid_separation_m", "speed_over_ground"]:
        if df[col].notna().any():
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    altitude_msl = df["altitude_msl_m"].to_numpy(dtype=float)
    geoid_sep = df["geoid_separation_m"].to_numpy(dtype=float)
    altitude_ellipsoidal = np.where(
        ~np.isnan(altitude_msl) & ~np.isnan(geoid_sep),
        altitude_msl + geoid_sep,
        altitude_msl,
    )

    out = pd.DataFrame(
        {
            "point_id": np.arange(len(df), dtype=np.int64),
            "timestamp": df["timestamp"].values,
            "lat": df["lat"].astype(float).values,
            "lon": df["lon"].astype(float).values,
            "mag_raw": df["mag_raw"].astype(float).values,
            "altitude_msl_m": altitude_msl,
            "geoid_separation_m": geoid_sep,
            "altitude_ellipsoidal_m": altitude_ellipsoidal,
            "speed_over_ground": df["speed_over_ground"].astype(float).values,
        }
    )
    out.attrs["n_invalid_coords_removed"] = n_invalid_coords_removed
    out.attrs["source_format"] = raw.attrs.get("source_format", "generic")
    return out


def load_drone_csv(path_or_buffer) -> pd.DataFrame:
    """Parse a drone magnetometer CSV/ASC into a normalized DataFrame.

    The source format (generic/Geometrics MagArrow, MicroInfinity Mag,
    SENSYS MagDrone R1, SENSYS MagDrone R3 raw, or SENSYS MagDrone R3 ASC)
    is auto-detected from the file header - see _detect_format.

    Returns columns: point_id, timestamp, lat, lon, mag_raw, altitude_msl_m,
    geoid_separation_m, altitude_ellipsoidal_m, speed_over_ground (may be
    NaN if not present in source). Result carries the detected format name
    in `.attrs["source_format"]`.

    Note: despite its name, the generic source's `HeightOverEllipsoid`
    column matches the GGA "geoid separation" field position/magnitude
    (near-constant, tens of meters), not a true ellipsoidal height, so it
    is treated as geoid separation here: ellipsoidal height = MSL altitude
    + geoid separation. Other formats without a geoid-separation field
    fall back to using MSL/reported altitude directly as an approximation.
    """
    text = _read_text(path_or_buffer)
    fmt = _detect_format(text)
    raw = _FORMAT_PARSERS[fmt](text)
    raw.attrs["source_format"] = fmt
    return _finalize(raw)


def load_drone_csvs(buffers: list) -> pd.DataFrame:
    """Load and concatenate multiple drone CSVs (e.g. several flights).
    Files may mix different source formats.

    Each file is parsed (and its sparse GGA-only fields interpolated)
    independently before concatenation, since interpolating across a time
    gap between two separate flights would be meaningless. The combined
    result is re-sorted by timestamp, deduplicated on exact-timestamp
    collisions (only meaningful once files are combined - the same
    instant logged twice, e.g. an overlapping re-upload of the same
    flight), and point_id is reassigned 0..N-1. QC counts are attached
    via combined.attrs for the caller to surface to the user.
    """
    if not buffers:
        raise DroneLoadError("드론 파일이 없습니다.")
    parts = [load_drone_csv(buf) for buf in buffers]
    n_invalid_coords_removed = sum(p.attrs.get("n_invalid_coords_removed", 0) for p in parts)
    source_formats = sorted({p.attrs.get("source_format", "generic") for p in parts})

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    n_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    n_duplicate_timestamps_removed = n_before_dedup - len(combined)

    combined["point_id"] = np.arange(len(combined), dtype=np.int64)
    combined.attrs["n_invalid_coords_removed"] = n_invalid_coords_removed
    combined.attrs["n_duplicate_timestamps_removed"] = n_duplicate_timestamps_removed
    combined.attrs["source_formats"] = source_formats
    return combined
