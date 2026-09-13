"""_check_file_level_offsets used to crash when the survey had several
files/tiles that each contributed exactly one active point: np.diff on a
length-1 array is length-0, so every per-file diff array was empty, and
np.concatenate([]) (guarded only by "did groupby produce any groups", not
by "did any group produce a non-empty diff") raised ValueError."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from app.store import Project


def test_single_point_per_file_does_not_crash():
    df = pd.DataFrame(
        {
            "line_id": [0, 1, 2, 3],
            "source_file_index": [0, 1, 2, 3],
            "anomaly": [10.0, 12.0, 9.0, 11.0],
        }
    )
    project = Project(id="test")

    result = project._check_file_level_offsets(df)

    assert result["available"] is True


def test_mixed_single_and_multi_point_files_still_flags_the_outlier():
    df = pd.DataFrame(
        {
            "line_id": [0, 1, 1, 1, 2],
            "source_file_index": [0, 1, 1, 1, 2],
            "anomaly": [10.0, 50.0, 51.0, 49.5, 10.5],
        }
    )
    project = Project(id="test")

    result = project._check_file_level_offsets(df)

    assert result["available"] is True
    assert result["flagged_any"] is True
    flagged = {f["source_file_index"] for f in result["files"] if f["flagged"]}
    # files 0 and 2 (single points near 10) sit far from the pooled median
    # (dominated by file 1's cluster near 50) - the single-point diff-array
    # fix must not prevent them from still being correctly flagged.
    assert {0, 2} <= flagged


if __name__ == "__main__":
    test_single_point_per_file_does_not_crash()
    test_mixed_single_and_multi_point_files_still_flags_the_outlier()
    print("ALL CHECKS PASSED")
