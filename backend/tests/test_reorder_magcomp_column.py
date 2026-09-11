"""tools/reorder_magcomp_column.py - moving the whole MagComp column so the
file reads Mag, MagLPF, MagComp.

Like the other repair tool, this rewrites raw survey files, so most of
these tests are about what must survive untouched: every field's exact
text, the line endings, the encoding, and the trailing field this format
carries past the header width.
"""
from __future__ import annotations

import codecs
import pathlib
import sys

import pytest

_TOOLS = pathlib.Path(__file__).resolve().parent.parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

from fix_magcomp_column import split_raw_fields  # noqa: E402
from reorder_magcomp_column import (  # noqa: E402
    main,
    new_column_order,
    reorder_file,
)

# A header shaped like the real one: MagComp sits well after MagLPF, with
# other columns in between that have to shift right by one.
HEADER = ["Date", "Time", "Lat", "Lon", "Mag", "MagLPF", "Alt", "Speed", "MagComp", "Extra"]


def _file(tmp_path, rows, header=None, name="survey.csv", newline="\r\n", encoding="utf-8", trailing_comma=True):
    head = list(header if header is not None else HEADER)
    lines = [",".join(head)]
    for r in rows:
        lines.append(",".join(r) + ("," if trailing_comma else ""))
    p = tmp_path / name
    p.write_bytes(newline.join(lines).encode(encoding) + newline.encode(encoding))
    return p


def _row(**over):
    base = {"Date": "2026-07-24", "Time": "10:00:01", "Lat": "34.4687", "Lon": "126.4008",
            "Mag": "48123.4500", "MagLPF": "48123.400", "Alt": "45.6", "Speed": "3.20",
            "MagComp": "48120.789", "Extra": "x"}
    base.update(over)
    return [base[c] for c in HEADER]


def _cols(line):
    return split_raw_fields(line)


def test_the_new_order_moves_the_column_and_shifts_the_rest():
    # 8 columns, move #6 to just after #2.
    assert new_column_order(8, move_index=6, after_index=2) == [0, 1, 2, 6, 3, 4, 5, 7]


def test_the_new_order_is_right_when_the_column_starts_before_its_target():
    """Removing it first shifts the target's index down, which is the easy
    thing to get wrong by one."""
    assert new_column_order(6, move_index=1, after_index=4) == [0, 2, 3, 4, 1, 5]


def test_the_new_order_is_a_permutation():
    for move, after in ((0, 5), (5, 0), (3, 4), (7, 1)):
        order = new_column_order(8, move, after)
        assert sorted(order) == list(range(8))


def test_magcomp_lands_immediately_after_maglpf(tmp_path):
    src = _file(tmp_path, [_row(), _row(MagComp="48121.001")])

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    out = report.out_path.read_text(encoding="utf-8").splitlines()
    header = [c.strip() for c in _cols(out[0])]
    assert header[:9] == ["Date", "Time", "Lat", "Lon", "Mag", "MagLPF", "MagComp", "Alt", "Speed"]
    # Mag, MagLPF, MagComp consecutive - what was actually asked for.
    assert header.index("MagComp") == header.index("MagLPF") + 1 == header.index("Mag") + 2


def test_every_value_moves_with_its_own_column(tmp_path):
    """The header reordering is meaningless if the data doesn't follow it."""
    src = _file(tmp_path, [_row(Mag="1", MagLPF="2", Alt="3", Speed="4", MagComp="5")])

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)
    out = report.out_path.read_text(encoding="utf-8").splitlines()

    header = [c.strip() for c in _cols(out[0])]
    values = _cols(out[1])
    got = dict(zip(header, values))
    assert got["Mag"] == "1" and got["MagLPF"] == "2" and got["MagComp"] == "5"
    assert got["Alt"] == "3" and got["Speed"] == "4"


def test_no_value_is_reformatted(tmp_path):
    """Trailing zeros and exponent notation must come through untouched -
    a column move is no reason to rewrite numbers."""
    src = _file(tmp_path, [_row(Mag="48123.4500", MagLPF="1.230", Alt="6.02E+23", Speed=" padded ")])

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)
    out = report.out_path.read_text(encoding="utf-8").splitlines()

    assert set(_cols(out[1])) >= {"48123.4500", "1.230", "6.02E+23", " padded "}


def test_the_trailing_field_past_the_header_stays_at_the_end(tmp_path):
    """Data rows in this format end with an extra comma, so they carry one
    more field than the header. Reordering must not drag that into the
    middle of the row."""
    src = _file(tmp_path, [_row()], trailing_comma=True)

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)
    out = report.out_path.read_text(encoding="utf-8").splitlines()

    header_n = len(_cols(out[0]))
    row = _cols(out[1])
    assert len(row) == header_n + 1
    assert row[-1] == "", "the trailing empty field must stay last"


def test_commas_inside_quoted_nmea_sentences_do_not_break_the_columns(tmp_path):
    """These files embed GGA sentences in single quotes, commas and all.
    Counting those as separators would shuffle the wrong columns."""
    header = ["Mag", "MagLPF", "Gga", "MagComp"]
    rows = [["1", "2", "'$GPGGA,101530.00,3428.1,N,12634.5,E,1,08'", "9"]]
    src = _file(tmp_path, rows, header=header)

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)
    out = report.out_path.read_text(encoding="utf-8").splitlines()

    assert [c.strip() for c in _cols(out[0])] == ["Mag", "MagLPF", "MagComp", "Gga"]
    assert _cols(out[1])[:3] == ["1", "2", "9"]
    assert _cols(out[1])[3] == "'$GPGGA,101530.00,3428.1,N,12634.5,E,1,08'"


def test_line_endings_and_cp949_encoding_survive(tmp_path):
    src = _file(tmp_path, [_row(Speed="측정")], newline="\r\n", encoding="cp949")

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert report.encoding == "cp949"
    raw = report.out_path.read_bytes()
    assert b"\r\n" in raw and b"\n\n" not in raw
    raw.decode("cp949")


def test_a_file_without_a_bom_does_not_gain_one(tmp_path):
    src = _file(tmp_path, [_row(Speed="측정")], encoding="utf-8")
    assert not src.read_bytes().startswith(codecs.BOM_UTF8)

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert not report.out_path.read_bytes().startswith(codecs.BOM_UTF8)


def test_the_original_is_left_alone(tmp_path):
    src = _file(tmp_path, [_row()], name="HaeNam_01.csv")
    before = src.read_bytes()

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert report.out_path.name == "HaeNam_01-r.csv"
    assert src.read_bytes() == before


# ---------------------------------------------------------------------------
# cases it must decline or flag rather than guess at
# ---------------------------------------------------------------------------


def test_a_file_already_in_the_right_order_is_left_alone(tmp_path):
    header = ["Date", "Mag", "MagLPF", "MagComp", "Alt"]
    src = _file(tmp_path, [["2026-07-24", "1", "2", "3", "4"]], header=header)

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert report.skipped is not None and "이미" in report.skipped
    assert report.out_path is None


def test_a_missing_column_is_reported_by_name(tmp_path):
    header = ["Date", "Mag", "MagLPF", "Alt"]
    src = _file(tmp_path, [["2026-07-24", "1", "2", "3"]], header=header)

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert report.skipped is not None and "MagComp" in report.skipped
    assert report.out_path is None


def test_rows_too_short_to_place_are_left_and_counted(tmp_path):
    """A row with fewer fields than the header can't be mapped to columns,
    so it stays as-is - and must be reported, because that row alone keeps
    the old order."""
    src = _file(tmp_path, [_row()], trailing_comma=False)
    with open(src, "a", encoding="utf-8", newline="") as f:
        f.write("short,row\r\n")

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert report.n_rows_wrong_width == 1
    assert report.n_rows_reordered == 1


def test_column_names_are_matched_ignoring_case_and_padding(tmp_path):
    header = ["Date", "Mag", " maglpf ", "Alt", "MAGCOMP"]
    src = _file(tmp_path, [["2026-07-24", "1", "2", "3", "4"]], header=header)

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=False, output_dir=None)

    assert report.skipped is None
    out = report.out_path.read_text(encoding="utf-8").splitlines()
    assert [c.strip() for c in _cols(out[0])] == ["Date", "Mag", "maglpf", "MAGCOMP", "Alt"]


def test_dry_run_writes_nothing(tmp_path):
    src = _file(tmp_path, [_row()])

    report = reorder_file(src, "MagComp", "MagLPF", dry_run=True, output_dir=None)

    assert report.n_rows_reordered == 1
    assert not report.out_path.exists()


def test_rerunning_does_not_reprocess_its_own_output(tmp_path, capsys):
    _file(tmp_path, [_row()])

    assert main([str(tmp_path)]) == 0
    assert (tmp_path / "survey-r.csv").exists()
    assert main([str(tmp_path)]) == 0

    assert not (tmp_path / "survey-r-r.csv").exists()
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["survey-r.csv", "survey.csv"]


def test_a_missing_folder_reports_instead_of_raising(capsys):
    assert main(["/no/such/folder"]) == 1
    assert "폴더를 찾을 수 없습니다" in capsys.readouterr().out
