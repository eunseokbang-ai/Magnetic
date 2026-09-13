"""Unit tests for the normalised 4th/8th difference noise QC channel."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.noise_qc import compute_difference_qc

RNG = np.random.default_rng(0)


def _make_df(n=200, noise_std=0.0, line_id=0, sample_rate_hz=10.0):
    t = pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(seconds=1.0 / sample_rate_hz))
    smooth = 50000.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, n))  # slow, smooth "geology"
    values = smooth + (RNG.normal(0, noise_std, n) if noise_std > 0 else 0.0)
    return pd.DataFrame({"timestamp": t, "mag_filtered": values, "line_id": line_id})


def test_clean_smooth_signal_has_low_normalized_difference():
    df = _make_df(noise_std=0.0)
    result = compute_difference_qc(df, "mag_filtered")
    assert result["available"]
    assert result["overall_rms_4th_diff_nt"] < 0.05
    assert result["overall_rms_8th_diff_nt"] < 0.05


def test_white_noise_normalized_4th_diff_recovers_input_sigma():
    """For pure white noise, the normalised 4th difference should have
    approximately the same standard deviation as the injected noise
    itself (the point of the sqrt(70) normalisation)."""
    sigma = 2.0
    df = _make_df(n=2000, noise_std=sigma)
    result = compute_difference_qc(df, "mag_filtered")
    assert result["available"]
    # Should recover the injected sigma to within ~20% (finite-sample noise
    # in the RMS estimate plus a small amount of real "geology" bleed-through).
    assert abs(result["overall_rms_4th_diff_nt"] - sigma) < 0.3 * sigma


def test_flags_the_noisier_line_relative_to_others():
    quiet1 = _make_df(n=300, noise_std=0.2, line_id=0)
    quiet2 = _make_df(n=300, noise_std=0.2, line_id=1)
    noisy = _make_df(n=300, noise_std=3.0, line_id=2)
    df = pd.concat([quiet1, quiet2, noisy], ignore_index=True)

    result = compute_difference_qc(df, "mag_filtered")
    assert result["available"]
    flagged_ids = {l["line_id"] for l in result["lines"] if l["flagged"]}
    assert flagged_ids == {2}
    assert result["n_lines_flagged"] == 1


def test_excludes_unassigned_points():
    df = _make_df(n=100, line_id=0)
    df2 = _make_df(n=50, line_id=-1)
    combined = pd.concat([df, df2], ignore_index=True)
    result = compute_difference_qc(combined, "mag_filtered")
    assert result["available"]
    assert all(l["line_id"] >= 0 for l in result["lines"])


def test_unavailable_when_lines_too_short():
    df = _make_df(n=5, line_id=0)
    result = compute_difference_qc(df, "mag_filtered")
    assert result["available"] is False


if __name__ == "__main__":
    test_clean_smooth_signal_has_low_normalized_difference()
    test_white_noise_normalized_4th_diff_recovers_input_sigma()
    test_flags_the_noisier_line_relative_to_others()
    test_excludes_unassigned_points()
    test_unavailable_when_lines_too_short()
    print("ALL CHECKS PASSED")
