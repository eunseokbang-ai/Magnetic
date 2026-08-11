"""apply_diurnal_correction's extrapolated_mask: np.interp has no
extrapolation mode, so drone samples outside the base station's own
[t0, t1] coverage silently get the correction clamped to the nearest
boundary base reading instead of tracking the (still varying) real
external field - a real mechanism for a day-specific offset. This flags
exactly which samples that happened to, rather than returning them
indistinguishable from properly-covered ones."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.diurnal import apply_diurnal_correction


def test_extrapolated_mask_flags_only_out_of_coverage_samples():
    base_t = pd.date_range("2026-01-01T10:00:00", periods=60, freq="1min")
    base_df = pd.DataFrame({"timestamp": base_t, "mag": 50000.0 + np.sin(np.linspace(0, 3, 60))})

    # drone flies from 09:55 (5 min before base coverage starts) to 11:05
    # (5 min after it ends) - the first/last few samples fall outside
    # [base_t[0], base_t[-1]].
    drone_t = pd.Series(pd.date_range("2026-01-01T09:55:00", "2026-01-01T11:05:00", freq="1min"))
    drone_mag = np.full(len(drone_t), 50000.0)

    result = apply_diurnal_correction(drone_t, drone_mag, base_df)

    within_coverage = (drone_t >= base_t[0]) & (drone_t <= base_t[-1])
    assert np.array_equal(result.extrapolated_mask, ~within_coverage.to_numpy())
    assert result.extrapolated_mask.sum() == int((~within_coverage).sum())
    assert result.extrapolated_mask.sum() > 0  # the test setup actually has out-of-coverage samples


def test_extrapolated_mask_all_false_when_fully_covered():
    base_t = pd.date_range("2026-01-01T09:00:00", periods=120, freq="1min")
    base_df = pd.DataFrame({"timestamp": base_t, "mag": 50000.0})
    drone_t = pd.Series(pd.date_range("2026-01-01T09:30:00", "2026-01-01T10:00:00", freq="1min"))
    drone_mag = np.full(len(drone_t), 50000.0)

    result = apply_diurnal_correction(drone_t, drone_mag, base_df)
    assert not result.extrapolated_mask.any()


if __name__ == "__main__":
    test_extrapolated_mask_flags_only_out_of_coverage_samples()
    test_extrapolated_mask_all_false_when_fully_covered()
    print("ALL CHECKS PASSED")
