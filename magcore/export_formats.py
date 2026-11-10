"""Exports in a coordinate system the recipient asked for, and two more
formats: Geosoft XYZ for line data and GeoPackage for points and grids.

Grids are computed in the project's projected CRS (a UTM zone, or a
Korean TM belt when chosen at processing). Until 2026-11 the GeoTIFF and
Surfer exports could only leave in that CRS, so a client working in
Korea 2000 Central Belt (EPSG:5186) or in geographic coordinates had to
reproject every file themselves - and did it with whatever resampling
their GIS defaulted to.

Reprojecting a grid means resampling it: the cells of the target CRS are
not the cells of the source. That is done here once, on purpose, with
bilinear interpolation for values (a nearest-neighbour warp of a smooth
field leaves visible steps; cubic overshoots at the nodata edge) and
nearest neighbour for an already-coloured image (colours must not be
averaged across the transparent edge). The grid stays north-up in the
target CRS at GDAL's default resolution for the warp, which preserves the
cell count along the diagonal. Point data are not resampled at all: each
point's coordinates are transformed and its value kept.

The exported numbers are therefore the same field, but not the same
array: statistics agree to well under a percent, single cells do not
match one to one. Anything that must reproduce the program's own grid
exactly should be exported without a target CRS.
"""
from __future__ import annotations

import io
import os
import sqlite3
import struct
import tempfile

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject


class ExportError(ValueError):
    pass


def check_epsg(epsg: int) -> CRS:
    try:
        return CRS.from_epsg(int(epsg))
    except Exception as exc:  # noqa: BLE001 - pyproj raises several types for an unknown code
        raise ExportError(f"알 수 없는 좌표계입니다: EPSG:{epsg}") from exc


def crs_label(epsg: int) -> str:
    crs = check_epsg(epsg)
    return f"EPSG:{int(epsg)} ({crs.name})"


# ------------------------------------------------------------------- grids
def reproject_geotiff(data: bytes, dst_epsg: int, coloured: bool = False) -> bytes:
    """A GeoTIFF as written by render.grid_to_geotiff_bytes(_colored),
    warped into EPSG:`dst_epsg`."""
    dst_crs = rasterio.crs.CRS.from_wkt(check_epsg(dst_epsg).to_wkt())
    with rasterio.io.MemoryFile(data) as src_file, src_file.open() as src:
        if src.crs == dst_crs:
            return data
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        resampling = Resampling.nearest if coloured else Resampling.bilinear
        with rasterio.io.MemoryFile() as dst_file:
            with dst_file.open(**profile) as dst:
                for band in range(1, src.count + 1):
                    out = np.full((height, width), src.nodata if src.nodata is not None else 0, dtype=profile["dtype"])
                    reproject(
                        source=rasterio.band(src, band), destination=out,
                        src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                        dst_transform=transform, dst_crs=dst_crs, dst_nodata=src.nodata,
                        resampling=resampling,
                    )
                    dst.write(out, band)
                if src.count >= 3:
                    dst.colorinterp = src.colorinterp
            return bytes(dst_file.read())


def geotiff_grid(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values with row 0 = south, x of cell centres, y of cell centres)
    of a single-band north-up GeoTIFF - the program's own grid layout."""
    with rasterio.io.MemoryFile(data) as f, f.open() as src:
        values = src.read(1, masked=True).astype(float).filled(np.nan)
        t = src.transform
        x = t.c + (np.arange(src.width) + 0.5) * t.a
        y = t.f + (np.arange(src.height) + 0.5) * t.e          # t.e < 0: row 0 is north
        return np.flipud(values), x, y[::-1]


def geotiff_to_gpkg(data: bytes, layer: str = "grid") -> bytes:
    """The same raster as a GeoPackage (tiled gridded coverage for float
    values, a tile pyramid for a coloured image)."""
    with rasterio.io.MemoryFile(data) as f, f.open() as src:
        profile = {"driver": "GPKG", "height": src.height, "width": src.width, "count": src.count, "dtype": src.dtypes[0],
                   "crs": src.crs, "transform": src.transform, "RASTER_TABLE": layer}
        if src.count == 1 and np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
            # Float tiles stored as TIFF: the coverage extension's default (16-bit PNG
            # with a scale and offset) quantises the values - measured here, a grid
            # with median 138 nT read back as -100.
            profile["TILE_FORMAT"] = "TIFF"
            profile["nodata"] = -9999.0 if src.nodata is None or np.isnan(src.nodata) else src.nodata   # the extension wants a number
        arrays = [src.read(b) for b in range(1, src.count + 1)]
    fd, path = tempfile.mkstemp(suffix=".gpkg")
    os.close(fd)
    os.unlink(path)                      # GDAL creates the database itself
    try:
        with rasterio.open(path, "w", **profile) as dst:
            for band, array in enumerate(arrays, start=1):
                if "nodata" in profile and np.issubdtype(array.dtype, np.floating):
                    array = np.where(np.isfinite(array), array, profile["nodata"]).astype(array.dtype)
                dst.write(array, band)
        if "nodata" in profile:
            # The driver writes the values but leaves the coverage's "no data" entry
            # empty (measured: the file read back with nodata None and -9999 counted
            # as a reading), so it is filled in where the standard keeps it.
            con = sqlite3.connect(path)
            con.execute("UPDATE gpkg_2d_gridded_coverage_ancillary SET data_null = ?", (float(profile["nodata"]),))
            con.commit()
            con.close()
        with open(path, "rb") as fh:
            return fh.read()
    except Exception as exc:  # noqa: BLE001 - driver availability differs between GDAL builds
        raise ExportError(f"GeoPackage 격자를 쓰지 못했습니다: {exc}") from exc
    finally:
        for leftover in (path, path + "-journal", path + ".aux.xml"):
            if os.path.exists(leftover):
                os.unlink(leftover)


def grid_to_xyz_in_crs(values: np.ndarray, easting: np.ndarray, northing: np.ndarray, src_epsg: int, dst_epsg: int) -> bytes:
    """Space-delimited "x y value" rows with the cell centres in
    EPSG:`dst_epsg` - no resampling, each cell keeps its own value. A
    two-line header names the CRS and the columns; geographic CRSs are
    written lon lat, as GIS software expects for x y."""
    dst = check_epsg(dst_epsg)
    to_dst = Transformer.from_crs(f"EPSG:{src_epsg}", dst, always_xy=True)
    e2d, n2d = np.meshgrid(easting, northing)
    finite = np.isfinite(values)
    x, y = to_dst.transform(e2d[finite], n2d[finite])
    digits = 7 if dst.is_geographic else 2
    names = "lon lat" if dst.is_geographic else "x y"
    rows = [f"# {crs_label(dst_epsg)}", f"# {names} value"]
    rows += [f"{a:.{digits}f} {b:.{digits}f} {v:.4f}" for a, b, v in zip(x, y, values[finite])]
    return ("\n".join(rows) + "\n").encode("utf-8")


# ------------------------------------------------------------------ points
def _transformed_xy(df: pd.DataFrame, src_epsg: int, dst_epsg: int) -> tuple[np.ndarray, np.ndarray, CRS]:
    dst = check_epsg(dst_epsg)
    x, y = Transformer.from_crs(f"EPSG:{src_epsg}", dst, always_xy=True).transform(df["x"].to_numpy(), df["y"].to_numpy())
    return np.asarray(x), np.asarray(y), dst


def points_to_geosoft_xyz(df: pd.DataFrame, src_epsg: int, dst_epsg: int, tie_line_id: np.ndarray | None = None) -> bytes:
    """The processed readings as a Geosoft XYZ file (Oasis montaj's line
    import): comment lines start with "/", each line of the survey is a
    block headed "Line <n>" (or "Tie <n>"), a row per reading, "*" for a
    missing value. Columns: X Y, then lat lon, anomaly, TMI, altitude
    and the time of day in seconds. Readings that belong to no line are
    left out - the format has no place for them."""
    x, y, dst = _transformed_xy(df, src_epsg, dst_epsg)
    digits = 7 if dst.is_geographic else 2
    seconds = (df["timestamp"] - df["timestamp"].dt.normalize()).dt.total_seconds().to_numpy()
    alt = df["altitude_ellipsoidal_m"].to_numpy() if "altitude_ellipsoidal_m" in df.columns else np.full(len(df), np.nan)
    cols = [x, y, df["lat"].to_numpy(), df["lon"].to_numpy(), df["anomaly"].to_numpy(), df["tmi"].to_numpy(), alt, seconds]
    formats = [f"{{:.{digits}f}}", f"{{:.{digits}f}}", "{:.7f}", "{:.7f}", "{:.3f}", "{:.3f}", "{:.2f}", "{:.2f}"]
    line_id = df["line_id"].to_numpy()
    tie = np.asarray(tie_line_id) if tie_line_id is not None else np.full(len(df), -1)
    out = io.StringIO()
    out.write(f"/ DroneMag Studio - {crs_label(dst_epsg)}\n")
    out.write(f"/ {'LON' if dst.is_geographic else 'X':>13} {'LAT' if dst.is_geographic else 'Y':>14} {'LATITUDE':>12} {'LONGITUDE':>12} {'ANOMALY_NT':>11} {'TMI_NT':>11} {'ALT_M':>9} {'TIME_S':>10}\n")
    fmt = lambda v, f: "*" if not np.isfinite(v) else f.format(v)
    for lid in sorted(int(v) for v in np.unique(line_id[line_id >= 0])):
        rows = np.flatnonzero(line_id == lid)
        is_tie = bool((tie[rows] >= 0).mean() > 0.5)
        out.write(f"{'Tie' if is_tie else 'Line'} {lid + 1}\n")
        for i in rows:
            out.write(" ".join(fmt(c[i], f).rjust(w) for c, f, w in zip(cols, formats, (14, 14, 12, 12, 11, 11, 9, 10))) + "\n")
    return out.getvalue().encode("utf-8")


def _gpkg_point(x: float, y: float, srs_id: int) -> bytes:
    """GeoPackage binary geometry for a 2D point: the "GP" header (version
    0, little-endian, no envelope) followed by standard WKB."""
    return b"GP" + struct.pack("<BBi", 0, 0x01, srs_id) + struct.pack("<BIdd", 1, 1, x, y)


def points_to_gpkg(df: pd.DataFrame, src_epsg: int, dst_epsg: int, table: str = "points", active: np.ndarray | None = None) -> bytes:
    """The processed readings as a GeoPackage point layer (opens in QGIS
    and ArcGIS as it is). Written with sqlite3 directly - the tables the
    GeoPackage standard requires for a feature layer, nothing else - so
    no GIS library is needed to produce it."""
    x, y, dst = _transformed_xy(df, src_epsg, dst_epsg)
    fd, path = tempfile.mkstemp(suffix=".gpkg")
    os.close(fd)
    try:
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute("PRAGMA application_id = 1196444487")            # "GPKG"
        cur.execute("PRAGMA user_version = 10300")                   # GeoPackage 1.3
        cur.executescript('''
            CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY, organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT);
            CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE, description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER,
                CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
            CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL, m TINYINT NOT NULL, CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
                CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id));
        ''')
        srs = [("Undefined Cartesian SRS", -1, "NONE", -1, "undefined", None), ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
               ("WGS 84 geodetic", 4326, "EPSG", 4326, CRS.from_epsg(4326).to_wkt(), None)]
        if int(dst_epsg) != 4326:
            srs.append((dst.name, int(dst_epsg), "EPSG", int(dst_epsg), dst.to_wkt(), None))
        cur.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)", srs)
        cur.execute(f'''CREATE TABLE "{table}" (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, geom POINT, point_id INTEGER, timestamp TEXT,
                        lat DOUBLE, lon DOUBLE, anomaly_nt DOUBLE, tmi_nt DOUBLE, altitude_m DOUBLE, line_id INTEGER, excluded BOOLEAN)''')
        cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id) VALUES (?, 'features', ?, ?, ?, ?, ?, ?)",
                    (table, table, float(np.nanmin(x)), float(np.nanmin(y)), float(np.nanmax(x)), float(np.nanmax(y)), int(dst_epsg)))
        cur.execute("INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', 'POINT', ?, 0, 0)", (table, int(dst_epsg)))
        alt = df["altitude_ellipsoidal_m"].to_numpy() if "altitude_ellipsoidal_m" in df.columns else np.full(len(df), np.nan)
        excluded = np.zeros(len(df), dtype=bool) if active is None else ~np.asarray(active, dtype=bool)
        none = lambda v: None if not np.isfinite(v) else float(v)
        rows = (
            (_gpkg_point(float(px), float(py), int(dst_epsg)), int(pid), str(ts), float(la), float(lo), none(a), none(t), none(h), int(lid), bool(ex))
            for px, py, pid, ts, la, lo, a, t, h, lid, ex in zip(
                x, y, df["point_id"].to_numpy(), df["timestamp"].astype(str).to_numpy(), df["lat"].to_numpy(), df["lon"].to_numpy(),
                df["anomaly"].to_numpy(), df["tmi"].to_numpy(), alt, df["line_id"].to_numpy(), excluded)
        )
        cur.executemany(f'INSERT INTO "{table}" (geom, point_id, timestamp, lat, lon, anomaly_nt, tmi_nt, altitude_m, line_id, excluded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)
        con.commit()
        con.close()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)
