from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LineParams(BaseModel):
    heading_lag_seconds: float = 1.0
    heading_tolerance_deg: float = Field(20.0, ge=1, le=90)
    min_speed_mps: float = Field(1.5, ge=0)
    min_line_length_m: float = Field(150.0, ge=0)
    turn_buffer_m: float = Field(15.0, ge=0)
    max_gap_seconds: float = Field(1.0, ge=0.1)
    # "heading_histogram" (default) or "pca" - see processing/lines.py.
    direction_method: Literal["heading_histogram", "pca"] = "heading_histogram"
    # Bridge brief mid-line interruptions (wind, a momentary heading/GPS
    # blip) back into the same line instead of leaving a data gap - see
    # processing/lines.py::_bridge_line_gaps.
    bridge_gaps: bool = True
    bridge_max_gap_m: float = Field(100.0, ge=0)
    bridge_max_offset_m: float = Field(15.0, ge=0)
    bridge_straightness_factor: float = Field(2.0, ge=1.0)


class DiurnalParams(BaseModel):
    time_offset_seconds: float = 0.0
    reference: str = "mean"  # "mean" | "first" | numeric string
    # "base_station" (default) requires an uploaded/fetched base log and
    # subtracts its interpolated, mean-referenced variation from the drone
    # reading. "assume_constant" is for when no base station was measured
    # at all: the Earth's field is assumed steady over the (short) survey
    # window, so the diurnal correction step is skipped entirely and the
    # drone's own filtered reading is used as-is (mag_diurnal_corrected =
    # mag_filtered) - see store.py::run_pipeline.
    mode: Literal["base_station", "assume_constant"] = "base_station"


class BaseQCParams(BaseModel):
    # Trims the noisy installation/pickup transient at the start/end of
    # the base station log (handling, mechanical shock, proximity to the
    # operator) before it's used for diurnal correction - see
    # processing/base_qc.py:trim_base_transients.
    trim_enabled: bool = True
    trim_window_seconds: float = Field(30.0, gt=0)
    trim_threshold_k: float = Field(6.0, gt=0)
    trim_confirm_seconds: float = Field(60.0, gt=0)
    trim_max_fraction: float = Field(0.2, ge=0, le=0.49)
    # Despikes the (already-trimmed) interior with the same robust
    # median/MAD logic as the drone despike step - see processing/despike.py.
    despike_enabled: bool = True
    despike_window_size: int = Field(11, ge=3, le=101)
    despike_threshold_k: float = Field(5.0, gt=0)


class HeadingCorrectionParams(BaseModel):
    # On by default (as of the 2026-09 HaeNam M350 survey): this platform's
    # opposite-heading lines fly at markedly different pitch (theta ~63 deg
    # vs ~19 deg near the survey's dominant azimuth, vs. ~47/31 deg on the
    # older M400 rig - see MagArrow-heading-error-calibration's Phase M/N
    # investigation), which the sensor's heading effect turns into a real,
    # well-constrained forward/reverse offset. Measured on the 42-file M350
    # HaeNam block (89 lines): with statistical_leveling already applied,
    # turning this on as well nearly halves the residual stripe ratio
    # (1.75% -> 0.90%; jitter 2.27 -> 1.16 nT), and the fitted offset
    # (5.58 nT) is well above its own spread (2.22 nT), so it does not
    # trigger the "unreliable" warning below.
    #
    # On the 32-file M400 block from the same site the estimate *does*
    # trigger that warning (offset 1.26 nT < spread 3.03 nT) - store.py's
    # pipeline now actually skips applying the correction in that case
    # (see `_heading_correction_should_apply`), rather than only showing a
    # warning while still writing the shift. Before that gate existed,
    # applying an unreliable estimate anyway barely moved the *aggregate*
    # stripe_ratio metric (0.95% either way) but visibly added wrinkles to
    # the M400 grid - a user caught this by eye, which the metric alone
    # did not. So: leaving this enabled is intended to be safe by design
    # (self-declining, like statistical_leveling), but the self-declining
    # part has to actually withhold the correction, not just warn about it.
    enabled: bool = True
    quiet_percentile: float = Field(40.0, ge=1, le=100)
    max_match_distance_m: Optional[float] = None
    # "local_plane" (default) fits a local plane plus a forward/reverse
    # step over overlapping neighbourhoods, so the estimate is not
    # contaminated by the geological gradient across the line gap the way
    # the older "nearest_pair" point-matching is. See processing/leveling.py.
    method: Literal["local_plane", "nearest_pair"] = "local_plane"
    # local_plane only: neighbourhood radius as a multiple of the line
    # spacing. Must stay >= 1.5 or a neighbourhood can't hold the three
    # lines needed to separate gradient from step.
    neighborhood_radius_factor: float = Field(2.0, ge=1.5, le=6.0)


class StatisticalLevelingParams(BaseModel):
    """Per-line level correction from the survey's own neighbouring lines -
    no tie lines, no calibration flight. This is the correction that
    targets visible striping; see processing/statistical_leveling.py."""

    # On by default, unlike the other leveling steps: per-line level error
    # is present in most surveys and is the usual cause of visible
    # striping, and this is the only correction here that can remove it.
    # It declines rather than guesses when the survey is too small to
    # separate error from geology (see processing/statistical_leveling.py),
    # so leaving it on costs nothing on data it cannot help.
    enabled: bool = True
    # How many lines the local trend is fitted over. There is a sweet spot
    # rather than a safe direction - too narrow and the trend follows the
    # error itself, too wide and it stops following the geology and its
    # own misfit gets removed as if it were error. See the measured table
    # in processing/statistical_leveling.py. The survey needs two more
    # lines than this, or the correction is skipped.
    trend_window_lines: int = Field(9, ge=7, le=51)
    # 0 = one constant shift per line. 1 = shift varies linearly along
    # each line, which also catches drift within one long line.
    order: int = Field(0, ge=0, le=1)
    # order=1 only: how many along-line segments the per-line correction
    # is estimated over before the straight line is fitted through them.
    n_segments: int = Field(4, ge=2, le=20)
    # Optional clamp (nT) on any single line's correction, so a line that
    # genuinely flew over a strong anomaly isn't flattened into its
    # neighbours. None = no clamp.
    max_shift_nt: Optional[float] = Field(None, gt=0)


class DespikeParams(BaseModel):
    enabled: bool = True
    window_size: int = Field(11, ge=3, le=101)  # samples
    threshold_k: float = Field(4.0, gt=0)  # robust-std multiples before a sample counts as a spike
    # Adaptive Hampel: widen the local median/MAD window on steep
    # along-track gradients so a genuine ramp isn't mistaken for a run of
    # spikes. See processing/despike.py:despike for details.
    adaptive: bool = True
    adaptive_gradient_threshold: float = Field(5.0, gt=0)  # nT/sample
    adaptive_expand_samples: int = Field(4, ge=0, le=50)


class SwayDetectionParams(BaseModel):
    # Flags samples where the IMU (gyroscope/accelerometer) shows the
    # sensor was swinging/rotating abnormally, and excludes them the same
    # way turn/takeoff-landing samples already are - see processing/sway.py.
    # A no-op when the source file has no gyro/accel columns (only the
    # generic/Geometrics MagArrow schema carries them). Off by default - an
    # optional correction the user opts into, not a baseline QC step every
    # survey needs.
    enabled: bool = False
    threshold_k: float = Field(4.0, gt=0)  # robust-z multiples before a sample counts as high-sway


class HeadingEffectCalibrationParams(BaseModel):
    # Applies the Zhang et al. (2022, The Leading Edge) heading-effect
    # compensation: uses a separately-uploaded short calibration flight
    # (see /upload/heading_calibration) to model each sample's magnetometer
    # reading offset as a function of its 3-axis-compass-derived orientation,
    # then subtracts it from the survey data - recovering samples that
    # sway detection (processing/sway.py) would otherwise just exclude.
    # A no-op whenever no calibration flight has been uploaded, or the
    # source format has no Compass columns. Off by default - an optional
    # correction the user opts into, not a baseline QC step every survey
    # needs.
    enabled: bool = False
    # When no dedicated calibration flight was uploaded, auto-build one
    # from the survey's own turn segments instead (see
    # processing/heading_calibration.py:build_turn_based_calibration) -
    # per Geometrics' own MagArrow guidance, turns are usually the best
    # calibration data available. Ignored once a dedicated calibration
    # flight is uploaded (that always takes precedence).
    auto_calibrate_from_turns: bool = True
    # Held-out cross-validation residual standard deviation (nT) above
    # which the calibration data is flagged as likely contaminated (real
    # gradient, drone noise, ...) rather than pure heading effect - see
    # processing/heading_calibration.py:cross_validate_heading_effect_map.
    # Std rather than peak-to-peak: a single held-out point near a sparse
    # region can spike p2p even for genuinely clean data, while std stays
    # stable and separates clean (~1 nT observed) from contaminated
    # (~10 nT observed) calibration data by a wide margin. The compensation
    # is still applied either way; this only affects the reported
    # quality_pass flag, so the user can judge whether to trust it.
    quality_threshold_nt: float = Field(3.0, gt=0)


class NoiseQcParams(BaseModel):
    # Normalised 4th/8th difference noise QC channels (Denisov et al.,
    # 2006), computed per line right after flight-path cleaning - see
    # processing/noise_qc.py. Purely diagnostic (surfaced in the process
    # summary/report), never modifies the data.
    enabled: bool = True


class DuplicateLineParams(BaseModel):
    # Detects lines re-flown over (almost exactly) the same physical track
    # within the main survey itself - e.g. a reflight after a bad first
    # pass, or an accidental repeat - and keeps only the better-quality
    # pass (by the same normalised 4th-difference noise metric as
    # noise_qc.py) over the overlapping stretch, excluding the worse one's
    # points there instead of letting gridding blend or arbitrarily pick
    # between a good pass and a noisy one. See processing/duplicate_lines.py.
    # Off by default - most surveys have no repeat-flown lines, and the
    # detection is deliberately tight (must be near-exactly the same track,
    # not just the next line over) but still a judgment call worth opting
    # into rather than applying silently.
    enabled: bool = False
    # How close (perpendicular to the shared track direction) two lines'
    # centroids must be to count as the same physical track, not just
    # adjacent survey lines - see processing/duplicate_lines.py's module
    # docstring for why this is much tighter than repeatability.py's.
    perp_tolerance_m: float = Field(8.0, gt=0)
    angle_tolerance_deg: float = Field(15.0, gt=0, le=45)


class CrossoverLevelingParams(BaseModel):
    # Off by default - tie lines aren't always flown, and this only does
    # anything useful when perpendicular calibration lines are present in
    # the raw flight (see processing/lines.py:detect_tie_lines).
    enabled: bool = False
    tie_tolerance_deg: float = Field(20.0, ge=1, le=90)
    max_crossover_distance_m: float = Field(15.0, gt=0)
    # Iterative network adjustment (survey <-> tie shifts refined together)
    # instead of treating the tie-line network as a perfect fixed
    # reference - see processing/crossover_leveling.py.
    iterative: bool = True
    # 0 (default) = constant DC shift per line ("zero order" levelling).
    # 1 = polynomial/linear levelling - shift varies along each line
    # instead of being one constant, per the UAV magnetics guidelines'
    # "Polynomial Levelling" method. Needs at least 2 tie-line crossings
    # per survey line to fit a slope; falls back to a constant for lines
    # with fewer. Always uses the single-pass (non-iterative) fit
    # regardless of the `iterative` setting above - see
    # processing/crossover_leveling.py:compute_crossover_leveling.
    leveling_order: int = Field(0, ge=0, le=1)


class NotchFilterParams(BaseModel):
    # Removes specific interference frequencies (e.g. UAV motor rotation
    # ~45-60Hz, or a known cultural noise source) from the raw magnetometer
    # signal before the main low-pass filter - see processing/filters.py:
    # notch_filter and the power-spectrum diagnostic
    # (processing/spectrum.py) used to find candidate frequencies. Empty
    # list (default) = no-op.
    frequencies_hz: list[float] = Field(default_factory=list)
    quality_factor: float = Field(30.0, gt=0)


FilterMethod = Literal["butterworth", "savgol", "moving_average"]


class ProcessParams(BaseModel):
    filter_method: FilterMethod = "butterworth"
    filter_cutoff_hz: float = Field(1.0, gt=0)  # used when filter_method="butterworth"
    filter_window_seconds: float = Field(1.0, gt=0)  # used when filter_method="savgol" or "moving_average"
    filter_polyorder: int = Field(3, ge=1, le=7)  # used when filter_method="savgol"
    # A known constant delay (seconds) between the magnetometer and GPS
    # position streams - some loggers' internal filtering/telemetry path
    # lags the GPS fix by a fraction of a second, which shows up as a
    # small along-track position error. 0.0 = no correction (default).
    gps_mag_lag_seconds: float = 0.0
    # Target EPSG code for the local projected CRS used throughout
    # processing/export (grid, GeoTIFF, XYZ, ...). Left as None (the
    # default), the UTM zone is auto-detected from the data's centroid -
    # set this to override it, e.g. to match a national grid or to keep
    # results consistent with a survey area that straddles a UTM zone
    # boundary. Takes precedence over korea_projection when both are set.
    utm_epsg_override: Optional[int] = None
    # Convenience alternative to utm_epsg_override for Korean surveys:
    # "korea_utm" = EPSG:5179 (KGD2002 Unified CS), "korea2010" =
    # EPSG:5185-5188 (KGD2002 Belt 2010, picked by longitude band), "utm" =
    # Korea-domestic UTM 51N/52N. None (default) = generic auto UTM.
    korea_projection: Optional[Literal["korea_utm", "korea2010", "utm"]] = None
    despike_params: DespikeParams = DespikeParams()
    base_qc_params: BaseQCParams = BaseQCParams()
    sway_detection: SwayDetectionParams = SwayDetectionParams()
    heading_effect_calibration: HeadingEffectCalibrationParams = HeadingEffectCalibrationParams()
    duplicate_line_params: DuplicateLineParams = DuplicateLineParams()
    line_params: LineParams = LineParams()
    diurnal_params: DiurnalParams = DiurnalParams()
    heading_correction: HeadingCorrectionParams = HeadingCorrectionParams()
    crossover_leveling: CrossoverLevelingParams = CrossoverLevelingParams()
    # Runs after the two above: they remove what they can explain (a
    # direction-dependent offset, a tie-line-measured shift), and this
    # takes out whatever per-line level error is left.
    statistical_leveling: StatisticalLevelingParams = StatisticalLevelingParams()
    noise_qc: NoiseQcParams = NoiseQcParams()
    notch_filter: NotchFilterParams = NotchFilterParams()
    # Derive the display boundary from the flown lines at the end of
    # processing (see store.auto_display_boundary and
    # processing/boundary.py) instead of leaving the grid capped only at
    # the convex hull, which over-fills any concave footprint. On by
    # default because that over-fill is the common case and shows up as
    # interpolated surface over unflown ground; a boundary the user drew
    # or imported by hand is replaced on the next processing run, so turn
    # this off to keep one.
    auto_display_boundary: bool = True
    # How far outside the outermost flight line the boundary sits. None
    # (the default) means "one line spacing", resolved per-survey in
    # store.auto_display_boundary - a fixed metre value is either far too
    # tight on a 100 m-spaced survey or too loose on a 5 m-spaced one.
    display_boundary_buffer_m: Optional[float] = Field(None, gt=0)


ValueField = Literal["tmi", "anomaly"]
# 3D inversion only ever accepts "anomaly", never "tmi" - see
# InversionParams.value's comment for why.
InversionValueField = Literal["anomaly"]
TransformName = Literal[
    "rtp", "rte", "1vd", "2vd", "as", "thdr", "tilt", "theta",
    "dx", "dy", "dxx", "dyy", "dxy", "dxz", "dyz",
    "upward_continuation", "detrend", "microlevel",
]
GridMethod = Literal["nearest", "linear", "cubic", "spline", "minimum_curvature", "boxing"]


class HillshadeParams(BaseModel):
    # Geosoft Oasis Montaj-style "color-shaded relief" - the grid values
    # are treated as a pseudo-terrain and illuminated, so subtle
    # gradients read as raised/shadowed texture instead of flat color
    # bands. Common defaults for cartographic hillshading (NW light,
    # 45 deg altitude) work well here too.
    hillshade: bool = False
    hillshade_azimuth_deg: float = Field(315.0, ge=0, le=360)
    hillshade_altitude_deg: float = Field(45.0, ge=1, le=90)
    hillshade_exaggeration: float = Field(3.0, gt=0, le=20)


StretchName = Literal["linear", "equalize", "normal"]


class ContourParams(BaseModel):
    show_contours: bool = False
    # if None, contour_n_levels evenly-spaced levels are picked from the
    # grid's own min/max range instead of a fixed nT interval.
    contour_interval_nt: Optional[float] = Field(None, gt=0)
    contour_n_levels: int = Field(10, ge=2, le=50)


class GridRequest(HillshadeParams, ContourParams):
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    stretch: StretchName = "linear"
    colored: bool = False  # GeoTIFF export only: bake in the on-screen colormap as an RGBA GeoTIFF instead of raw float values
    # Along-line low-pass applied before gridding, to prevent flight-line-parallel
    # "corrugation" ridges (along-line detail no cross-line interpolation can
    # resolve anyway, given lines are typically 5-50x farther apart than the
    # along-line sample spacing). Defaults to on, at a wavelength matched to the
    # estimated line spacing; see processing.gridding._along_line_lowpass.
    along_line_smooth: bool = True
    along_line_smooth_wavelength_m: Optional[float] = Field(None, gt=0)
    # See TransformRequest.microlevel_pre_apply - applies here too so the
    # plain "그리드(원본)" view can show the same pre-leveled surface that
    # derived transforms are computed from, instead of only being
    # reachable via the "마이크로레벨링" transform itself.
    microlevel_pre_apply: bool = False
    microlevel_strength: float = Field(0.8, ge=0, le=1)
    microlevel_angle_tolerance_deg: float = Field(15.0, gt=0, le=45)
    microlevel_wavelength_factor: float = Field(1.5, gt=1)
    # "decorrugation" (default) removes all across-line energy shorter
    # than microlevel_cutoff_factor line spacings - the classic Minty
    # micro-levelling high-pass, and the one that actually catches
    # striping, which spans a harmonic series rather than one wavelength.
    # "notch" is the narrower band-reject around one line spacing;
    # microlevel_wavelength_factor applies to it alone. See
    # processing/microlevel.py.
    microlevel_mode: Literal["decorrugation", "notch"] = "decorrugation"
    microlevel_cutoff_factor: float = Field(4.0, gt=1, le=20)


class TransformRequest(HillshadeParams, ContourParams):
    transform: TransformName
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    stretch: StretchName = "linear"
    colored: bool = False  # GeoTIFF export only: bake in the on-screen colormap as an RGBA GeoTIFF instead of raw float values
    # See GridRequest.along_line_smooth - equally relevant here since every
    # derived transform is computed from this same underlying grid, and
    # derivative-based ones (RTP/1VD/tilt/etc.) amplify corrugation the most.
    along_line_smooth: bool = True
    along_line_smooth_wavelength_m: Optional[float] = Field(None, gt=0)
    # transform="upward_continuation" only: how far to continue the field upward.
    continuation_height_m: Optional[float] = Field(None, gt=0)
    # transform="detrend" only: order of the polynomial regional surface removed (1=plane, 2=quadratic, 3=cubic).
    trend_order: int = Field(1, ge=1, le=3)
    # Low-pass the base grid to its real across-line resolution before
    # any derivative-based transform (see
    # processing/transforms.py:limit_to_line_spacing_resolution). On by
    # default: differentiation multiplies amplitude by the wavenumber, so
    # across-line detail the survey never resolved - invisible in the
    # anomaly map - arrives in AS and the second derivatives eighty times
    # stronger, as the fine hatching those grids show. The filter is
    # directional, so compact targets keep their amplitude.
    derivative_presmooth: bool = True
    # Cutoff wavelength as a fraction of the line spacing. 1.0 is the
    # measured optimum: the artifact bottoms out there and 93% of a
    # compact dipole survives. Higher does not keep helping.
    derivative_presmooth_factor: float = Field(1.0, gt=0, le=3.0)
    # transform="microlevel" only: see processing/microlevel.py for what each controls.
    microlevel_strength: float = Field(0.8, ge=0, le=1)
    microlevel_angle_tolerance_deg: float = Field(15.0, gt=0, le=45)
    microlevel_wavelength_factor: float = Field(1.5, gt=1)
    # "decorrugation" (default) removes all across-line energy shorter
    # than microlevel_cutoff_factor line spacings - the classic Minty
    # micro-levelling high-pass, and the one that actually catches
    # striping, which spans a harmonic series rather than one wavelength.
    # "notch" is the narrower band-reject around one line spacing;
    # microlevel_wavelength_factor applies to it alone. See
    # processing/microlevel.py.
    microlevel_mode: Literal["decorrugation", "notch"] = "decorrugation"
    microlevel_cutoff_factor: float = Field(4.0, gt=1, le=20)
    # Applies microleveling (using the three params above) to the base grid
    # *before* computing whatever transform is requested (RTP/2VD/AS/etc.,
    # or even "none"), instead of microleveling only being its own
    # mutually-exclusive transform choice - lets line-parallel corrugation
    # get cleaned up before a derivative-based transform amplifies it,
    # rather than only after. No-op when transform="microlevel" itself.
    microlevel_pre_apply: bool = False
    # What to do about the band along the edge of the data, where every
    # derived grid is partly measuring its own boundary rather than the
    # ground: "off" (nothing), "outline" (draw the line where the margin
    # ends, hide nothing) or "mask" (blank the band out). Off by default -
    # a target sitting on the survey edge is real data and hiding it
    # without being asked would be worse than the streak. See
    # processing/edge_margin.py for why marking is the remedy and no
    # filter is.
    boundary_margin_mode: Literal["off", "outline", "mask"] = "off"
    # Width of that band. None = the gridding's own extrapolation radius
    # (max(2 cells, 1.2x line spacing)), i.e. exactly the strip whose
    # values were extrapolated outward from the flight lines rather than
    # interpolated between them.
    boundary_margin_m: Optional[float] = Field(None, gt=0)


class PolygonExportRequest(BaseModel):
    polygon: list[list[float]] = Field(..., min_length=3)  # [[lat, lon], ...]


class ManualExcludeRequest(BaseModel):
    mode: Literal["lines", "polygon", "point_ids", "reset"]
    action: Optional[Literal["exclude", "include"]] = None
    line_ids: Optional[list[int]] = None
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...]
    point_ids: Optional[list[int]] = None
    # By default, takeoff/landing ramp points (exclusion_reason
    # takeoff_ramp/landing_ramp - see processing/lines.py) are protected
    # from exclude/include drawing so they don't get accidentally toggled
    # while editing the survey lines around them. Set True to allow them
    # to be affected too (paired with the line editor's "show ramp
    # points" toggle).
    include_ramp: bool = False


class ManualSmoothRequest(BaseModel):
    """Marks points as affected by a localized, non-geological
    disturbance (a building, fence, parked vehicle, ...) so their
    anomaly/TMI value is replaced with a straight-line interpolation from
    the nearest unflagged points on either side along their own line,
    instead of being fabricated from a fitted model. See
    store.py::_apply_manual_smoothing. Points are targeted either by an
    explicit point_id list (e.g. a drag-selected range on a line's time
    series chart), a single lat/lon polygon (e.g. drawn on the map over a
    structure visible in an imported orthophoto reference layer), or a
    batch of polygons at once (mode "polygons" - e.g. every region the
    structure-distortion auto-scan found and the user confirmed; unioned
    into the same single history entry rather than one call per region)."""
    mode: Literal["point_ids", "polygon", "polygons", "reset"]
    point_ids: Optional[list[int]] = None
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...]
    polygons: Optional[list[list[list[float]]]] = None  # [[[lat, lon], ...], ...]


class DisplayBoundaryRequest(BaseModel):
    """Optional user-drawn polygon that further restricts where grid
    interpolation/extrapolation is shown, on top of the automatic
    convex-hull cap in processing/gridding.py::grid_points. The hull cap
    alone still over-fills a concave (e.g. L-shaped) survey footprint's
    notch since the notch is inside the hull; drawing an explicit boundary
    lets the user clip exactly to the real survey outline in that case.
    None/omit clears the boundary (shows the full auto-capped extent)."""
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...] or None to clear


class AutoBoundaryRequest(BaseModel):
    """Regenerate the display boundary from the flown lines - the same
    thing processing does by default (ProcessParams.auto_display_boundary),
    exposed on its own so the buffer can be retuned without reprocessing.
    See processing/boundary.py. buffer_m=None means "one line spacing",
    the same default processing uses."""
    buffer_m: Optional[float] = Field(None, gt=0)


class IntermagnetFetchRequest(BaseModel):
    """Downloads IAGA-2002 data for a public INTERMAGNET observatory to use
    as a stand-in base station when no local base was measured - see
    processing/intermagnet.py. Requires this server's own outbound network
    to reach the data service; if that's blocked, use the IAGA-2002 file
    upload endpoint instead (same parser, no network access needed).

    Leave start_date/end_date both unset to auto-target exactly the
    project's own drone survey flight dates (see Project._survey_dates) -
    the recommended default, since it downloads only the days actually
    needed rather than every day across a survey's full min-to-max span
    (which can be mostly empty padding for a survey flown on a few
    separate days weeks apart). Set both to override with an explicit
    inclusive date range instead (e.g. no drone data uploaded yet, or a
    deliberately different period). Any requested date that comes back
    missing (e.g. the most recent day or two, before definitive data is
    published) is estimated from its own immediate day-before/day-after
    neighbors - see processing/intermagnet.py::fetch_observatory_dates."""
    iaga_code: str
    start_date: str | None = None  # "YYYY-MM-DD"; omit with end_date for survey-date auto-detect
    end_date: str | None = None  # "YYYY-MM-DD", inclusive


class NearestIntermagnetRequest(BaseModel):
    """Auto-selects a directionally spread set of nearby INTERMAGNET
    observatories around the project's survey area (or an explicit
    override point) and combines their data via inverse-distance
    weighting into one substitute base station series - see
    processing/intermagnet.py::select_nearest_observatories /
    estimate_base_from_observatories. Requires this server's own outbound
    network to reach the data service.

    start_date/end_date behave exactly as in IntermagnetFetchRequest -
    leave both unset to auto-target the project's own survey flight
    dates, or set both for an explicit override range.

    max_distance_km caps how far a candidate observatory may be from the
    target point and still be picked (see select_nearest_observatories's
    docstring for why very distant stations - beyond mid-latitude Sq
    phase/amplitude coherence - are more a liability than a help once a
    nearer station has a gap); n_stations then becomes a ceiling rather
    than a guaranteed count if fewer stations exist within the cutoff.
    Set to null to disable the cutoff and reach as far as needed."""
    start_date: str | None = None  # "YYYY-MM-DD"; omit with end_date for survey-date auto-detect
    end_date: str | None = None  # "YYYY-MM-DD", inclusive
    n_stations: int = Field(4, ge=1, le=8)
    max_distance_km: float | None = Field(2000.0, gt=0)
    target_lat: float | None = None
    target_lon: float | None = None


class TileBboxRequest(BaseModel):
    """A bounding box + zoom range to estimate or bulk-download offline
    basemap tiles for - see processing/tile_cache.py. Used for both the
    /tiles/estimate (dry run) and /tiles/download endpoints."""
    source: Literal["osm", "esri"]
    min_lat: float = Field(ge=-90, le=90)
    min_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)
    min_zoom: int = Field(ge=0, le=19)
    max_zoom: int = Field(ge=0, le=19)


class LocalTileFolderRequest(BaseModel):
    """Points the backend at a local folder containing an already-tiled
    {z}/{x}/{y}.<ext> raster pyramid (e.g. produced by QGIS or GDAL's
    gdal2tiles.py) so it can be served as a map layer directly from disk -
    see processing/local_tiles.py. Used for very large orthophotos where
    uploading+re-rendering the raw GeoTIFF (see /overlay-images) is
    impractical."""
    path: str
    label: Optional[str] = None
    scheme: Literal["auto", "xyz", "tms"] = "auto"


class InversionParams(BaseModel):
    # Always "anomaly", never "tmi" - the inversion solve has no free
    # constant/DC-level unknown, only a non-negative susceptibility per
    # cell (see processing/inversion.py::invert). TMI's absolute level
    # (tens of thousands of nT) would either force every cell to the
    # non-negativity clip ceiling or, even under "anomaly", have any
    # leftover regional DC offset misread as a blanket layer of
    # susceptibility - so this is pinned to the one physically sound
    # option rather than left as a user-selectable ValueField.
    value: InversionValueField = "anomaly"
    # Any of these left as None (the default) is auto-estimated from the
    # survey's own line spacing/extent and anomaly spectrum - see
    # processing/inversion_auto.py.
    obs_cell_size_m: Optional[float] = Field(None, gt=0)
    depth_extent_m: Optional[float] = Field(None, gt=0)
    n_layers: Optional[int] = Field(None, ge=1, le=80)
    # Per-layer thickness growth ratio with depth (thickness_k = thickness_0
    # * depth_growth_factor^k) - see processing/inversion.py:build_mesh.
    # None (default) uses processing/inversion_auto.DEFAULT_DEPTH_GROWTH_FACTOR.
    # 1.0 reproduces the old uniform-thickness mesh.
    depth_growth_factor: Optional[float] = Field(None, ge=1.0, le=2.0)
    assumed_agl_m: float = Field(50.0, gt=0)  # used only when no DEM is uploaded
    # Most published DEMs report orthometric height (above the geoid,
    # i.e. mean sea level) while the drone's own GPS records ellipsoidal
    # height (above the WGS84 reference ellipsoid) - the two differ by
    # the local geoid undulation N (h_ellipsoidal = H_orthometric + N),
    # which is tens of meters in many regions (e.g. roughly +25m across
    # South Korea). Left uncorrected, the mesh's DEM-derived ground
    # surface and the drone's own recorded flight altitude sit in two
    # different vertical datums, silently offsetting the active-cell
    # mask and every reported depth by that same amount whenever a DEM
    # is uploaded (the no-DEM path, which derives ground purely from the
    # drone's own ellipsoidal altitude, has no such mismatch since both
    # sides already share one datum). Added here, not applied
    # automatically - the true local N depends on both location and the
    # specific DEM's own datum, which callers must determine themselves
    # (e.g. from the DEM's metadata or a geoid calculator for the
    # survey's coordinates). 0.0 (default) reproduces the original
    # uncorrected behavior. Only used when a DEM is uploaded.
    dem_geoid_offset_m: float = 0.0
    regularization_strength: float = Field(1.0, gt=0)
    n_irls_iterations: int = Field(6, ge=1, le=30)
    # When set, overrides regularization_strength via a discrepancy-
    # principle search that targets this RMS misfit level (nT) instead of
    # using regularization_strength directly - see
    # processing/inversion.py:_select_regularization_strength. Leave None
    # (default) to keep manually setting regularization_strength.
    assumed_noise_nt: Optional[float] = Field(None, gt=0)
    # When the project has user-digitized geology blocks (see
    # GeologyUnitInput / Project.geology_units), regularize the inversion
    # toward their susceptibility values instead of toward zero - see
    # processing/inversion.py:invert's m_ref parameter. True by default so
    # digitizing geology and simply re-running "just works"; set False for
    # a quick side-by-side comparison against the no-prior-information run.
    use_geology_reference: bool = True


class GeologyUnitInput(BaseModel):
    """One user-digitized geological block for the 3D inversion's
    reference model: a polygon (traced over an uploaded geology map
    raster, e.g. the "12. 참조 레이어" GeoTIFF) plus the susceptibility
    value assigned to whatever rock unit it represents. Repeated with
    depth for every mesh column inside the polygon (a "2.5D" assumption -
    the unit is assumed to continue straight down, since early-stage
    geology maps rarely include cross-sections showing dip)."""
    name: str = Field(..., min_length=1, max_length=100)
    susceptibility_si: float = Field(..., ge=0, le=1.0)
    path: list[list[float]] = Field(..., min_length=3)  # [[lat, lon], ...] polygon vertices


class GeologyUnitUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    susceptibility_si: Optional[float] = Field(None, ge=0, le=1.0)


class InversionSliceRequest(HillshadeParams):
    layer_index: int = Field(0, ge=0)
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


SectionProfile = Literal["custom", "ew", "ns"]


class InversionSectionRequest(BaseModel):
    profile: SectionProfile = "custom"
    position_frac: Optional[float] = Field(None, ge=0, le=1)  # required for ew/ns
    path: Optional[list[list[float]]] = Field(None, min_length=2)  # [[lat, lon], ...], required for custom
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    sample_spacing_m: float = Field(10.0, gt=0)


Slice3DOrientation = Literal["custom", "ew", "ns", "azimuth"]


class InversionSlice3DRequest(BaseModel):
    """One interior cutting plane through the 3D inversion mesh, for the
    combined "isosurface volume + several simultaneous section planes"
    3D view - see store.py:get_inversion_slice_3d. "ew"/"ns" cut at an
    arbitrary interior position_frac (0-1) instead of only the mesh
    boundary; "azimuth" cuts through the mesh's own center at an
    arbitrary compass bearing (no map drawing needed - just a slider);
    "custom" follows an arbitrary-direction path (same [[lat, lon], ...]
    convention as InversionSectionRequest.path)."""
    orientation: Slice3DOrientation = "custom"
    position_frac: Optional[float] = Field(None, ge=0, le=1)  # required for ew/ns
    # Compass bearing (degrees, 0=north-south trending line, 90=east-west
    # trending line) of the section line through the mesh center -
    # required for "azimuth". 0-180 covers every distinct line direction
    # (a line at az and az+180 is the same line).
    azimuth_deg: Optional[float] = Field(None, ge=0, lt=180)
    path: Optional[list[list[float]]] = Field(None, min_length=2)  # [[lat, lon], ...], required for custom
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
    sample_spacing_m: float = Field(10.0, gt=0)


class EulerDeconvolutionRequest(BaseModel):
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    # 0 = contact/fault edge, 1 = thin dyke/sill/sheet edge, 2 = pipe/vertical
    # cylinder, 3 = sphere/point dipole - the four textbook structural indices.
    structural_index: float = Field(1.0, ge=0, le=3)
    window_size_m: float = Field(100.0, gt=0)
    max_depth_uncertainty_pct: float = Field(30.0, gt=0, le=200)
    # When set, the drone's own recorded GPS altitude (already gridded and
    # used elsewhere for IGRF/3D-inversion) is used as each observation's
    # real z-coordinate in Euler's homogeneity equation instead of assuming
    # every point was measured on one flat plane - the standard "draped
    # survey" correction. This constant is the flight's height above ground
    # (AGL, e.g. 50 for a terrain-following survey held at 50m) and is used
    # only to relabel the solved depth as "depth below ground" instead of
    # "depth below the local flight point" - see
    # processing/euler_deconvolution.py and terrain.py:estimate_ground_elevation
    # for the same assumed-AGL convention used by the 3D inversion. None
    # (default) keeps the original flat-z=0 assumption unchanged.
    flight_agl_m: Optional[float] = Field(None, gt=0)


class TargetDetectionRequest(BaseModel):
    # Much finer default than the 10m used for regional geology grids -
    # near-surface compact targets (mines, ordnance, vehicles) need a
    # detection grid fine enough to resolve a footprint of a few meters.
    cell_size_m: float = Field(1.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    # If omitted, auto = threshold_k * robust std of the anomaly grid.
    amplitude_threshold_nt: Optional[float] = Field(None, gt=0)
    threshold_k: float = Field(4.0, gt=0)
    min_footprint_m: float = Field(0.5, gt=0)
    max_footprint_m: float = Field(15.0, gt=0)
    fit_window_m: float = Field(8.0, gt=0)
    max_depth_m: float = Field(5.0, gt=0)
    min_fit_quality: float = Field(0.3, ge=0, le=1)


class StructureScanRequest(BaseModel):
    """Auto-detect ground-structure-caused magnetic distortion by
    combining an OpenStreetMap building/road location prior with the same
    compact-anomaly signal detector used for near-surface target
    detection (see TargetDetectionRequest) - see
    processing/osm_structures.py and store.py::scan_structure_distortion.
    Defaults are structure-scale rather than mine/UXO-scale: coarser
    detection grid, larger allowed footprint, more lenient fit quality
    (buildings are messier, less point-like sources than a compact
    target)."""
    cell_size_m: float = Field(2.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    # How far a building's/road's magnetic influence (rebar, buried
    # utilities, guardrails) is assumed to reach beyond its mapped
    # footprint - also how far apart nearby structures' regions get
    # merged into one when applying smoothing.
    building_buffer_m: float = Field(10.0, gt=0)
    road_buffer_m: float = Field(6.0, gt=0)
    # If omitted, auto = threshold_k * robust std of the anomaly grid.
    amplitude_threshold_nt: Optional[float] = Field(None, gt=0)
    threshold_k: float = Field(4.0, gt=0)
    min_footprint_m: float = Field(1.0, gt=0)
    max_footprint_m: float = Field(40.0, gt=0)
    fit_window_m: float = Field(25.0, gt=0)
    max_depth_m: float = Field(10.0, gt=0)
    min_fit_quality: float = Field(0.2, ge=0, le=1)


class MultiscaleEdgeRequest(BaseModel):
    # Multi-scale edge detection ("worming") - see
    # processing/multiscale_edges.py. Traces THDR ridges at a series of
    # upward-continued heights, a quick structural-mapping complement to
    # Euler deconvolution recommended by the UAV magnetics guidelines.
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    heights_m: list[float] = Field(default_factory=lambda: [0.0, 25.0, 50.0, 100.0, 200.0])
    percentile: float = Field(80.0, gt=0, lt=100)


class PowerSpectrumRequest(BaseModel):
    line_id: int
    value: Literal["mag_raw", "mag_filtered"] = "mag_raw"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class InversionVolumeRequest(BaseModel):
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None


class OverlaySampleRequest(BaseModel):
    lat: float
    lon: float


class QcCertificateRequest(BaseModel):
    """Acceptance thresholds for the QC pass/fail certificate (see
    processing/qc_certificate.py) - defaults are reasonable starting
    points, not a formal published standard, since acceptable noise/
    repeatability/gap levels genuinely vary by survey purpose and
    equipment; adjust to match the actual delivery spec."""
    noise_threshold_multiplier: float = Field(2.0, gt=0)
    max_repeatability_1sigma_nt: float = Field(5.0, gt=0)
    max_sampling_gap_pct: float = Field(5.0, gt=0)
    max_excluded_pct: float = Field(30.0, gt=0)


class LineamentRequest(BaseModel):
    """See processing/lineaments.py. `source` selects which derivative of
    the grid ridge-maxima are traced on - THD/analytic-signal/tilt/1VD are
    the standard lineament-extraction inputs (a plain TMI/anomaly grid has
    no sharp ridge over a contact the way its derivatives do)."""
    value: str = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    source: Literal["thd", "as", "tilt", "1vd"] = "thd"
    percentile_threshold: float = Field(90.0, gt=0, le=100)
    min_segment_points: int = Field(4, ge=2)
    min_length_m: float = Field(0.0, ge=0)
    rose_bin_width_deg: float = Field(10.0, gt=0, le=90)
    max_gap_cells: float = Field(2.5, gt=0)


class TiltDepthRequest(BaseModel):
    """See processing/depth_estimation.py:tilt_depth_estimates (Salem et
    al. 2007)."""
    value: str = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    min_depth_m: float = Field(1.0, gt=0)
    max_depth_m: float = Field(500.0, gt=0)


class AnalyticSignalDepthRequest(BaseModel):
    """See processing/depth_estimation.py:analytic_signal_depth_estimates
    (Nabighian 1972 / Roest et al. 1992 half-width method)."""
    value: str = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    percentile_threshold: float = Field(90.0, gt=0, le=100)
    search_radius_cells: int = Field(15, gt=0)


class SpectralDepthRequest(BaseModel):
    """See processing/depth_estimation.py:spectral_depth_diagnostic
    (Spector & Grant 1970)."""
    value: str = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None


class ContactDetectionRequest(BaseModel):
    """Magnetic contact detection - see processing/contacts.py. Reuses
    multiscale_edges.py's per-height THDR ridge points but keeps only
    those that persist across several heights, then vectorizes them into
    contact segments (distinct from LineamentRequest's single-height
    structural-trend extraction)."""
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    heights_m: list[float] = Field(default_factory=lambda: [0.0, 20.0, 40.0, 60.0, 80.0])
    percentile_threshold: float = Field(80.0, gt=0, lt=100)
    min_persistence: float = Field(0.5, gt=0, le=1.0)
    min_segment_points: int = Field(4, ge=2)
    min_length_m: float = Field(0.0, ge=0)
    max_gap_cells: float = Field(3.0, gt=0)


class ProspectivityRequest(BaseModel):
    """Rule-based mineral prospectivity ("target score") mapping - see
    processing/prospectivity.py. Weights a set of normalized magnetic-
    derivative/structural layers into one 0-1 score grid and reports
    ranked, auto-explained local-maximum targets. purpose selects a
    preset weighting ("magnetite_fe" | "skarn" | "ni_cu_pge"); pass
    purpose="custom" with weights to set your own."""
    value: str = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    purpose: Literal["magnetite_fe", "skarn", "ni_cu_pge", "custom"] = "magnetite_fe"
    weights: Optional[dict[str, float]] = None
    decay_length_m: float = Field(200.0, gt=0)
    score_threshold: float = Field(0.6, ge=0, le=1)
    max_targets: int = Field(20, ge=1, le=200)
    min_target_separation_m: Optional[float] = None
    colormap: str = "viridis"


class GridConfidenceRequest(BaseModel):
    """Requests a companion "how much should I trust this cell" layer for
    an already-gridded result - see processing/gridding.py::grid_confidence
    and store.py::get_grid_confidence_overlay."""
    value: str = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    colormap: str = "RdYlGn"
