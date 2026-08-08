"""Render a 2D grid (with NaN outside coverage) into a colored PNG plus the
lat/lon bounding box Leaflet needs for an ImageOverlay."""
from __future__ import annotations

import base64
import struct
from io import BytesIO

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import LightSource, Normalize
from scipy.stats import norm as _scipy_norm
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Surfer 6 Binary Grid (DSBB) NoData sentinel - Golden Software's own
# documented value (close to float32's max), not an arbitrary choice.
_SURFER_NODATA = 1.70141e38


class _EqualizeNorm(Normalize):
    """Histogram-equalized "norm": maps each value to its empirical CDF
    rank (0-1) among the finite grid cells instead of a linear fraction of
    [vmin, vmax]. Spreads the color range evenly across however the data is
    actually distributed, so a handful of extreme outlier anomalies no
    longer wash out the color contrast over the rest of a long-tailed
    survey - a standard alternative stretch in geophysical/remote-sensing
    display software (Geosoft, ArcGIS "histogram equalize" stretch).
    Duck-types matplotlib's Normalize (vmin/vmax + __call__) so it can be
    passed anywhere a plain Normalize is used, hillshade included."""

    def __init__(self, sorted_finite_values: np.ndarray):
        super().__init__(vmin=float(sorted_finite_values[0]), vmax=float(sorted_finite_values[-1]), clip=False)
        self._sorted = sorted_finite_values
        self._ranks = np.linspace(0.0, 1.0, len(sorted_finite_values))

    def __call__(self, value, clip=None):
        arr = np.ma.asarray(value, dtype=float)
        filled = arr.filled(self._sorted[0]) if np.ma.is_masked(arr) else np.asarray(arr)
        result = np.interp(filled, self._sorted, self._ranks)
        mask = np.ma.getmaskarray(arr) if np.ma.is_masked(arr) else False
        return np.ma.array(result, mask=mask)


class _NormalNorm(Normalize):
    """"Normal distribution" stretch (per the UAV magnetics guidelines'
    common-stretches list): maps each value through the Gaussian CDF
    fitted to the grid's own mean/std, rather than either a plain linear
    fraction of [vmin, vmax] or the empirical-rank _EqualizeNorm above.
    Unlike equalize (which reproduces whatever the actual distribution
    shape is, exactly, via ranks), this assumes the data is close to
    normally distributed and stretches accordingly - most of the color
    range concentrates within a few standard deviations of the mean,
    which suits a grid that genuinely is roughly bell-shaped around a
    background level (typical for anomaly grids dominated by background
    noise with a few real anomalies), without needing every individual
    rank to be preserved."""

    def __init__(self, mean: float, std: float):
        std = std if std > 0 else 1.0
        super().__init__(vmin=mean - 3.0 * std, vmax=mean + 3.0 * std, clip=False)
        self._mean = mean
        self._std = std

    def __call__(self, value, clip=None):
        arr = np.ma.asarray(value, dtype=float)
        filled = arr.filled(self._mean) if np.ma.is_masked(arr) else np.asarray(arr)
        result = _scipy_norm.cdf(filled, loc=self._mean, scale=self._std)
        mask = np.ma.getmaskarray(arr) if np.ma.is_masked(arr) else False
        return np.ma.array(result, mask=mask)


def _render_rgba(
    grid_values: np.ndarray,
    cmap_name: str,
    symmetric: bool,
    vmin: float | None,
    vmax: float | None,
    hillshade: bool,
    hillshade_azimuth_deg: float,
    hillshade_altitude_deg: float,
    hillshade_exaggeration: float,
    cell_size_m: float,
    stretch: str,
) -> tuple[np.ndarray, float, float]:
    """Shared color-mapping logic behind both the PNG map overlay and the
    colored GeoTIFF export, so hillshade/stretch behave identically in
    each output. Returns (rgba uint8 array in original row order, vmin,
    vmax)."""
    finite = grid_values[np.isfinite(grid_values)]
    if finite.size == 0:
        raise ValueError("표시할 유효한 그리드 값이 없습니다.")

    if stretch == "equalize":
        # rank-based, so outlier clipping/symmetric-range logic (meant for
        # a linear stretch) doesn't apply - every finite cell contributes.
        norm = _EqualizeNorm(np.sort(finite))
        vmin, vmax = float(finite.min()), float(finite.max())
    elif stretch == "normal":
        norm = _NormalNorm(float(np.mean(finite)), float(np.std(finite)))
        vmin, vmax = norm.vmin, norm.vmax
    else:
        explicit_range = vmin is not None and vmax is not None
        if vmin is None:
            vmin = float(np.nanpercentile(finite, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(finite, 98))
        if symmetric and not explicit_range:
            m = max(abs(vmin), abs(vmax))
            vmin, vmax = -m, m
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = matplotlib.colormaps[cmap_name]

    if hillshade:
        # Geosoft Oasis Montaj-style "color-shaded relief": treat the
        # grid's own values as a pseudo-terrain and illuminate it, so
        # subtle gradients/edges in the anomaly pattern show up as
        # raised/shadowed texture instead of disappearing into flat
        # color bands - a standard display mode for airborne geophysics.
        # LightSource can't handle NaN when computing slopes, so gaps are
        # filled with the mean before shading and then re-masked to
        # transparent afterward exactly as in the non-hillshade path.
        filled = np.where(np.isfinite(grid_values), grid_values, float(np.mean(finite)))
        light = LightSource(azdeg=hillshade_azimuth_deg, altdeg=hillshade_altitude_deg)
        shaded = light.shade(
            filled,
            cmap=cmap,
            norm=norm,
            blend_mode="overlay",
            vert_exag=hillshade_exaggeration,
            dx=cell_size_m,
            dy=cell_size_m,
        )
        rgba = (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
    else:
        rgba = (cmap(norm(grid_values)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(grid_values), 255, 0).astype(np.uint8)
    return rgba, vmin, vmax


def grid_to_png_overlay(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    utm_epsg: int,
    cmap_name: str = "viridis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    hillshade: bool = False,
    hillshade_azimuth_deg: float = 315.0,
    hillshade_altitude_deg: float = 45.0,
    hillshade_exaggeration: float = 3.0,
    cell_size_m: float = 1.0,
    stretch: str = "linear",
) -> dict:
    rgba, vmin, vmax = _render_rgba(
        grid_values, cmap_name, symmetric, vmin, vmax, hillshade,
        hillshade_azimuth_deg, hillshade_altitude_deg, hillshade_exaggeration, cell_size_m, stretch,
    )

    # array row 0 = southmost northing; image row 0 must be the top (north).
    img_array = np.flipud(rgba)
    image = Image.fromarray(img_array, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    # Leaflet's ImageOverlay stretches the PNG (one pixel per grid cell) to
    # exactly fill `bounds`, treating each pixel as covering an *area* (its
    # own square swath of ground), not as a point sample. easting/northing
    # are each cell's *center* coordinate, so the image must extend half a
    # cell past the outermost cell centers on every side for pixel i's
    # rendered center to land exactly on easting[i]/northing[i] - otherwise
    # every pixel is stretched across the wrong-sized box and each one's
    # true screen position drifts away from its real coordinate (zero at
    # the grid's own center, growing to half a cell at the edges). This is
    # the same "pixel is area" convention already used correctly for the
    # GeoTIFF export below (see from_origin(easting[0] - cell_size / 2,
    # ...)) - kept consistent here so the interactive map overlay and the
    # exported GeoTIFF agree on where each cell actually sits, and so a
    # map click (see store.py::sample_overlay_value) samples the exact
    # cell its color is drawn from instead of a neighboring one.
    half_e = (easting[1] - easting[0]) / 2.0 if len(easting) > 1 else cell_size_m / 2.0
    half_n = (northing[1] - northing[0]) / 2.0 if len(northing) > 1 else cell_size_m / 2.0
    # Order: SW, SE, NW, NE.
    corners_e = [easting.min() - half_e, easting.max() + half_e, easting.min() - half_e, easting.max() + half_e]
    corners_n = [northing.min() - half_n, northing.min() - half_n, northing.max() + half_n, northing.max() + half_n]
    lon_c, lat_c = transformer.transform(corners_e, corners_n)

    bounds = [[float(min(lat_c)), float(min(lon_c))], [float(max(lat_c)), float(max(lon_c))]]

    # `bounds` above is only a north/south/east/west axis-aligned envelope
    # of the 4 corners - it is NOT the image's true footprint. A UTM grid's
    # cell rows/columns are only exactly north-south/east-west along its
    # own zone's central meridian; anywhere else, "grid north" is rotated a
    # few tenths of a degree to a few degrees away from true north (map
    # convergence), so the grid's real shape on a lat/lon map is a slightly
    # sheared/rotated rectangle, not an axis-aligned one. A plain Leaflet
    # ImageOverlay can only stretch this PNG into an axis-aligned `bounds`
    # box, which silently discards that shear - every pixel not on the two
    # corners that happen to be simultaneously N/S- and E/W-extreme drifts
    # away from its true position, worse the farther the survey sits from
    # its UTM zone's central meridian and the larger the survey (measured
    # this at up to ~70m for a real multi-line survey in this app's own
    # test fixtures - enough to visibly mismatch the "지점값 확인"
    # click-to-inspect readout, which instead re-projects each click
    # exactly and is never affected by this). The frontend therefore uses
    # these exact 3 corners (leaflet-imageoverlay-rotated, which applies a
    # CSS affine transform instead of an axis-aligned stretch) to place the
    # image without that distortion; `bounds` is kept only as a fallback/
    # fit-to-view helper.
    topleft = [float(lat_c[2]), float(lon_c[2])]
    topright = [float(lat_c[3]), float(lon_c[3])]
    bottomleft = [float(lat_c[0]), float(lon_c[0])]

    return {
        "image_data_url": f"data:image/png;base64,{png_b64}",
        "bounds": bounds,
        "topleft": topleft,
        "topright": topright,
        "bottomleft": bottomleft,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap_name,
        "stretch": stretch,
    }


def grid_to_xyz_bytes(grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray, utm_epsg: int) -> bytes:
    """Plain-text XYZ export (space-delimited "lon lat value" rows, no
    header, NaN cells skipped) - the conventional geophysics text-grid
    format for import into other packages that don't want a GeoTIFF.
    Coordinates are reprojected to lon/lat so the file is self-describing
    without needing to also ship the local UTM zone/EPSG separately."""
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    e2d, n2d = np.meshgrid(easting, northing)
    finite = np.isfinite(grid_values)
    lon, lat = transformer.transform(e2d[finite], n2d[finite])
    values = grid_values[finite]

    lines = [f"{lo:.7f} {la:.7f} {v:.4f}" for lo, la, v in zip(lon, lat, values)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def grid_to_geotiff_bytes(grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray, utm_epsg: int) -> bytes:
    """Single-band float32 GeoTIFF with the raw (uncolored) grid values -
    unlike the PNG overlay above (which bakes in a colormap for display
    in this app), this preserves the actual numbers so the grid can be
    reopened and re-styled in Oasis Montaj, QGIS, ArcGIS, etc."""
    cell_size = float(easting[1] - easting[0])
    # array row 0 = southmost northing; GeoTIFF row 0 must be the top (north).
    data = np.flipud(grid_values).astype("float32")
    transform = from_origin(easting[0] - cell_size / 2, northing[-1] + cell_size / 2, cell_size, cell_size)
    crs = CRS.from_epsg(utm_epsg)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
        ) as dst:
            dst.write(data, 1)
        return bytes(memfile.read())


def grid_to_geotiff_bytes_colored(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    utm_epsg: int,
    cmap_name: str = "viridis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    hillshade: bool = False,
    hillshade_azimuth_deg: float = 315.0,
    hillshade_altitude_deg: float = 45.0,
    hillshade_exaggeration: float = 3.0,
    stretch: str = "linear",
) -> bytes:
    """4-band (RGBA) uint8 GeoTIFF baking in the same colormap/hillshade/
    stretch rendering shown on screen - unlike grid_to_geotiff_bytes above
    (raw float values, for reopening and re-analyzing), this is meant to
    be viewed as-is: dropped straight into Google Earth, a slide deck's
    GIS viewer, or any tool that just wants a georeferenced picture."""
    cell_size = float(easting[1] - easting[0])
    rgba, _, _ = _render_rgba(
        grid_values, cmap_name, symmetric, vmin, vmax, hillshade,
        hillshade_azimuth_deg, hillshade_altitude_deg, hillshade_exaggeration, cell_size, stretch,
    )
    # array row 0 = southmost northing; GeoTIFF row 0 must be the top (north).
    data = np.flipud(rgba).transpose(2, 0, 1)  # rasterio wants (band, row, col)
    transform = from_origin(easting[0] - cell_size / 2, northing[-1] + cell_size / 2, cell_size, cell_size)
    crs = CRS.from_epsg(utm_epsg)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=4,
            dtype="uint8",
            crs=crs,
            transform=transform,
            photometric="RGB",
            compress="deflate",
        ) as dst:
            dst.write(data)
        return bytes(memfile.read())


def grid_to_surfer_grd_bytes(grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray) -> bytes:
    """Surfer 6 Binary Grid (DSBB) export - the de facto grid interchange
    format for Golden Software Surfer, still widely used alongside/instead
    of GeoTIFF in domestic geophysical survey workflows. Layout: a 56-byte
    header ("DSBB" magic, int16 nx/ny, then float64 xmin/xmax/ymin/ymax/
    zmin/zmax), followed by nx*ny float32 values written row-major from
    south to north (each row west to east) - the format's documented
    origin convention, which already matches this app's internal grid
    array orientation (row 0 = southmost northing), so unlike the
    north-up PNG/GeoTIFF exports no vertical flip is needed here.
    """
    ny, nx = grid_values.shape
    if nx > 32767 or ny > 32767:
        raise ValueError("Surfer GRD 형식은 nx, ny가 각각 32767을 넘을 수 없습니다 (셀 크기를 키워보세요).")
    finite = grid_values[np.isfinite(grid_values)]
    if finite.size == 0:
        raise ValueError("내보낼 유효한 그리드 값이 없습니다.")
    zmin, zmax = float(finite.min()), float(finite.max())

    data = np.where(np.isfinite(grid_values), grid_values, _SURFER_NODATA).astype("<f4")
    header = struct.pack(
        "<4shhdddddd",
        b"DSBB",
        nx,
        ny,
        float(easting[0]),
        float(easting[-1]),
        float(northing[0]),
        float(northing[-1]),
        zmin,
        zmax,
    )
    return header + data.tobytes()


def polygon_to_bln_bytes(x: np.ndarray, y: np.ndarray) -> bytes:
    """Surfer Blanking File (.bln): a closed polygon boundary in local
    projected coordinates (meters), the standard way to restrict a
    Surfer grid display/mask to the actually-flown survey area (or any
    other user-drawn region of interest). First line is "point_count,0" -
    the trailing 0 is Surfer's documented flag for "blank everything
    OUTSIDE the polygon" (i.e. keep the interior), matching how this file
    is meant to be used to mask a grid down to the survey footprint. The
    first vertex is repeated at the end to close the ring if the caller
    didn't already do so.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("BLN 저장을 위해서는 폴리곤 점이 3개 이상 필요합니다.")
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    lines = [f"{len(x)},0"] + [f"{xi:.3f},{yi:.3f}" for xi, yi in zip(x, y)]
    return ("\n".join(lines) + "\n").encode("utf-8")
