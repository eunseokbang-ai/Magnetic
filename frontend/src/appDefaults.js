// Starting values of the app's parameter forms. Moved out of App.jsx in
// 2026-11 so the hooks that own each form (src/hooks/) can share them.

export const toggleButtonStyle = {
  width: 36,
  height: 36,
  borderRadius: 8,
  border: "1px solid #ddd0b2",
  background: "white",
  color: "#2c2418",
  fontSize: 16,
  cursor: "pointer",
  boxShadow: "0 1px 4px rgba(0,0,0,0.2)",
};

export const DEFAULT_INVERSION_PARAMS = {
  obs_cell_size_m: 30.0,
  depth_extent_m: 150.0,
  n_layers: 8,
  depth_growth_factor: null,
  regional_removal: "level",
  assumed_agl_m: 50.0,
  dem_geoid_offset_m: 0.0,
  regularization_strength: 1.0,
  n_irls_iterations: 6,
  assumed_noise_nt: null,
};

export const DEFAULT_TARGET_DETECTION_PARAMS = {
  cell_size_m: 1.0,
  amplitude_threshold_nt: null,
  threshold_k: 4.0,
  min_footprint_m: 0.5,
  max_footprint_m: 15.0,
  fit_window_m: 8.0,
  max_depth_m: 5.0,
  min_fit_quality: 0.3,
};

export const DEFAULT_QC_CERTIFICATE_PARAMS = {
  noise_threshold_multiplier: 2.0,
  max_repeatability_1sigma_nt: 5.0,
  max_sampling_gap_pct: 5.0,
  max_excluded_pct: 30.0,
};

export const DEFAULT_LINEAMENT_PARAMS = {
  value: "anomaly",
  cell_size_m: 10.0,
  method: "linear",
  source: "thd",
  percentile_threshold: 90.0,
  min_segment_points: 4,
  min_length_m: 0.0,
  rose_bin_width_deg: 10.0,
  max_gap_cells: 2.5,
};

export const DEFAULT_TILT_DEPTH_PARAMS = { min_depth_m: 1.0, max_depth_m: 500.0 };

export const DEFAULT_AS_DEPTH_PARAMS = { percentile_threshold: 90.0, search_radius_cells: 15 };

export const DEFAULT_CONTACT_PARAMS = {
  value: "anomaly",
  cell_size_m: 10.0,
  method: "linear",
  heights_m: [0.0, 20.0, 40.0, 60.0, 80.0],
  percentile_threshold: 80.0,
  min_persistence: 0.5,
  min_segment_points: 4,
  min_length_m: 0.0,
  max_gap_cells: 3.0,
};

export const DEFAULT_PROSPECTIVITY_PARAMS = {
  value: "anomaly",
  cell_size_m: 10.0,
  method: "linear",
  purpose: "magnetite_fe",
  weights: { asa: 0.3, thd: 0.2, structure: 0.2, contact: 0.2, susceptibility: 0.1 },
  decay_length_m: 200.0,
  score_threshold: 0.6,
  max_targets: 20,
};

export const DEFAULT_STRUCTURE_SCAN_PARAMS = {
  cell_size_m: 2.0,
  building_buffer_m: 10.0,
  road_buffer_m: 6.0,
  threshold_k: 4.0,
  min_footprint_m: 1.0,
  max_footprint_m: 40.0,
  fit_window_m: 25.0,
  max_depth_m: 10.0,
  min_fit_quality: 0.2,
};

// Ranked removal candidates - see processing/anomaly_candidates.py for
// why "as" (analytic signal) is the ranking field and why the region is
// sized from the anomaly's own half-width rather than a fixed radius.
export const DEFAULT_ANOMALY_CANDIDATE_PARAMS = {
  field: "as",
  n_candidates: 10,
  cell_size_m: 5.0,
  max_radius_m: 60.0,
};

export const DEFAULT_TRANSFORM_EXTRA_PARAMS = {
  continuation_height_m: 20.0,
  trend_order: 1,
  microlevel_strength: 0.8,
  microlevel_angle_tolerance_deg: 15.0,
  microlevel_wavelength_factor: 1.5,
  microlevel_mode: "decorrugation",
  microlevel_cutoff_factor: 4.0,
  microlevel_pre_apply: false,
  derivative_presmooth: true,
  derivative_presmooth_factor: 1.0,
  boundary_continuation: true,
  // Off by default: it trades real amplitude for a cleaner picture, which
  // is the user's call. 1.6 (line spacings) is the conservative setting.
  equivalent_source_factor: null,
  // Off: a target sitting on the survey edge is real data, and hiding it
  // without being asked would be worse than the streak it removes.
  boundary_margin_mode: "off",
  boundary_margin_m: null, // null = the gridding's own extrapolation radius
};

export const DEFAULT_PARAMS = {
  filter_method: "butterworth",
  filter_cutoff_hz: 1.0,
  sensor_channel: "1",
  filter_window_seconds: 1.0,
  filter_polyorder: 3,
  gps_mag_lag_seconds: 0.0,
  utm_epsg_override: null,
  korea_projection: null,
  despike_params: {
    enabled: true,
    window_size: 11,
    threshold_k: 4.0,
    adaptive: true,
    adaptive_gradient_threshold: 5.0,
    adaptive_expand_samples: 4,
  },
  base_qc_params: {
    trim_enabled: true,
    trim_window_seconds: 30.0,
    trim_threshold_k: 6.0,
    trim_confirm_seconds: 60.0,
    trim_max_fraction: 0.2,
    despike_enabled: true,
    despike_window_size: 11,
    despike_threshold_k: 5.0,
    segment_mode: "steps",
    segment_gap_minutes: 10.0,
    segment_step_threshold_nt: 20.0,
    segment_gap_suspicious_nt: 30.0,
  },
  sway_detection: {
    enabled: false,
    threshold_k: 4.0,
  },
  heading_effect_calibration: {
    enabled: false,
    auto_calibrate_from_turns: true,
    quality_threshold_nt: 3.0,
  },
  line_params: {
    heading_lag_seconds: 1.0,
    heading_tolerance_deg: 20.0,
    min_speed_mps: 0.1,
    min_line_length_m: 20.0,
    turn_buffer_m: 5.0,
    max_gap_seconds: 1.0,
    direction_method: "heading_histogram",
    merge_continued_lines: true,
    merge_max_offset_m: 25.0,
    merge_max_along_gap_m: 400.0,
    merge_max_overlap_fraction: 0.35,
    remove_transit: true,
    overlap_resolution: "auto",
    overlap_overrides: {},
  },
  diurnal_params: {
    time_offset_seconds: 0.0,
    reference: "mean",
  },
  heading_correction: {
    // On by default - see the matching comment in backend/app/models.py
    // HeadingCorrectionParams for the measured evidence (M350 HaeNam block).
    enabled: false,
    quiet_percentile: 40.0,
    max_match_distance_m: null,
    method: "local_plane",
    neighborhood_radius_factor: 2.0,
  },
  crossover_leveling: {
    enabled: false,
    tie_tolerance_deg: 20.0,
    max_crossover_distance_m: 15.0,
    iterative: true,
    leveling_order: 0,
  },
  statistical_leveling: {
    enabled: true,
    trend_window_lines: 9,
    order: 0,
    n_segments: 4,
    max_shift_nt: null,
  },
  noise_qc: { enabled: true },
  notch_filter: { frequencies_hz: [], quality_factor: 30.0 },
};
