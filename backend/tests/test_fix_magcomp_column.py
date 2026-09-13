"""tools/fix_magcomp_column.py - moving a MagComp value that landed in the
wrong column back to where it belongs.

This tool rewrites the user's raw survey files, so the tests are mostly
about what it must NOT do: everything except the two moved fields has to
survive byte for byte. Number formatting especially - reading these
through pandas and writing them back would quietly turn "1.230" into
"1.23" across the whole file, which is not a thing a repair tool may do
to raw data.
"""
from __future__ import annotations

import codecs
import pathlib
import sys

import pytest

_TOOLS = pathlib.Path(__file__).resolve().parent.parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

from fix_magcomp_column import (  # noqa: E402
    column_letter_to_index,
    fix_file,
    main,
    split_raw_fields,
)

BK = column_letter_to_index("BK")
CK = column_letter_to_index("CK")


def _row(values: dict[int, str], n_fields: int = 95) -> str:
    """A CSV row of n_fields columns, with `values` placed by 0-based index.
    BK and CK start empty so each test states its own case explicitly."""
    fields = [f"c{i}" for i in range(n_fields)]
    fields[BK] = ""
    if n_fields > CK:
        fields[CK] = ""
    for i, v in values.items():
        fields[i] = v
    return ",".join(fields)


def _write(tmp_path: pathlib.Path, name: str, lines: list[str], newline: str = "\r\n", encoding: str = "utf-8") -> pathlib.Path:
    p = tmp_path / name
    p.write_bytes(newline.join(lines).encode(encoding) + newline.encode(encoding))
    return p


def test_column_letters_map_to_the_right_indices():
    assert column_letter_to_index("A") == 0
    assert column_letter_to_index("Z") == 25
    assert column_letter_to_index("AA") == 26
    # The two the user named: BK is the 63rd column, CK the 89th.
    assert BK == 62
    assert CK == 88
    assert CK - BK == 26


def test_bad_column_letters_are_rejected():
    for bad in ("", "  ", "B2", "1", "-"):
        with pytest.raises(ValueError):
            column_letter_to_index(bad)


# ---------------------------------------------------------------------------
# field splitting - the part everything else depends on
# ---------------------------------------------------------------------------


def test_commas_inside_quoted_nmea_sentences_do_not_split_fields():
    """MagArrow CSVs embed GGA/RMC sentences in single quotes, and those
    contain commas. Splitting on every comma would shift every column
    after them - the tool would corrupt exactly what it is meant to fix."""
    line = "1,2,'$GPGGA,123519,4807.038,N,01131.000,E,1,08',9,10"
    fields = split_raw_fields(line)

    assert len(fields) == 5, "the 7 commas inside the quoted sentence must not split it"
    assert fields[2] == "'$GPGGA,123519,4807.038,N,01131.000,E,1,08'"
    assert fields[3] == "9"


def test_rejoining_split_fields_reproduces_the_line_exactly():
    """The tool's whole no-collateral-damage claim rests on this."""
    for line in (
        "a,b,c",
        "1,2,'x,y,z',4",
        "a,,b,,",
        "",
        "'quoted at start',tail",
        "trailing,",
    ):
        assert ",".join(split_raw_fields(line)) == line


# ---------------------------------------------------------------------------
# the fix itself
# ---------------------------------------------------------------------------


def test_shifted_value_moves_to_bk_and_ck_is_emptied(tmp_path):
    src = _write(tmp_path, "survey.csv", [
        _row({BK: "MagComp"}),               # header, already correct
        _row({BK: "48123.456"}),             # fine
        _row({BK: "", CK: "48124.789"}),     # shifted
        _row({BK: "48125.001"}),             # fine
    ])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.n_moved == 1
    assert report.n_already_ok == 3
    out = report.out_path.read_text(encoding="utf-8").splitlines()
    fixed = split_raw_fields(out[2])
    assert fixed[BK] == "48124.789"
    assert fixed[CK] == ""
    # Column count must not change, or every column after CK shifts.
    assert len(fixed) == len(split_raw_fields(out[1]))


def test_untouched_rows_are_passed_through_byte_for_byte(tmp_path):
    """Trailing zeros, exponent notation and odd spacing all have to
    survive - a repair tool that reformats numbers is worse than the bug."""
    delicate = _row({BK: "48123.4500", 3: "1.230", 4: "6.02E+23", 5: " padded "})
    src = _write(tmp_path, "survey.csv", [
        _row({BK: "MagComp"}),
        delicate,
        _row({BK: "", CK: "48124.789"}),
    ])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)
    out = report.out_path.read_text(encoding="utf-8").splitlines()

    assert out[1] == delicate


def test_the_repaired_row_keeps_every_other_field_verbatim(tmp_path):
    shifted = _row({BK: "", CK: "48124.789", 3: "1.230", 10: "'$GPGGA,123519,4807.038,N'"})
    src = _write(tmp_path, "survey.csv", [_row({BK: "MagComp"}), shifted])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)
    out = report.out_path.read_text(encoding="utf-8").splitlines()

    before, after = split_raw_fields(shifted), split_raw_fields(out[1])
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert differing == [BK, CK], "only the two moved columns may differ"


def test_crlf_line_endings_and_encoding_are_preserved(tmp_path):
    # Non-ASCII content, or cp949 and utf-8 are the same bytes and the
    # test would prove nothing about which one was chosen.
    src = _write(tmp_path, "survey.csv",
                 [_row({BK: "MagComp", 2: "측정"}), _row({BK: "", CK: "1.0", 2: "측정"})],
                 newline="\r\n", encoding="cp949")

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.encoding == "cp949"
    raw = report.out_path.read_bytes()
    assert b"\r\n" in raw and b"\n\n" not in raw
    raw.decode("cp949")  # must still be readable as what it was


def test_a_utf8_file_without_a_bom_does_not_gain_one(tmp_path):
    """Writing back through utf-8-sig would prepend three bytes the
    original never had, and plenty of software still trips over a BOM."""
    src = _write(tmp_path, "survey.csv",
                 [_row({BK: "MagComp", 2: "측정"}), _row({BK: "", CK: "1.0", 2: "측정"})],
                 encoding="utf-8")
    assert not src.read_bytes().startswith(codecs.BOM_UTF8)

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert not report.out_path.read_bytes().startswith(codecs.BOM_UTF8)


def test_a_utf8_file_with_a_bom_keeps_it(tmp_path):
    src = tmp_path / "survey.csv"
    body = _row({BK: "MagComp"}) + "\n" + _row({BK: "", CK: "1.0"}) + "\n"
    src.write_bytes(codecs.BOM_UTF8 + body.encode("utf-8"))

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.out_path.read_bytes().startswith(codecs.BOM_UTF8)


def test_output_is_named_with_the_m_suffix_and_the_original_is_untouched(tmp_path):
    src = _write(tmp_path, "HaeNam_01.csv", [_row({BK: "MagComp"}), _row({BK: "", CK: "1.0"})])
    original = src.read_bytes()

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.out_path.name == "HaeNam_01-m.csv"
    assert src.read_bytes() == original


# ---------------------------------------------------------------------------
# rows the tool must leave alone, and say so
# ---------------------------------------------------------------------------


def test_rows_with_both_columns_filled_are_left_alone_and_counted(tmp_path):
    """Ambiguous: moving would destroy whatever is already in BK. The tool
    reports these instead of guessing."""
    src = _write(tmp_path, "survey.csv", [
        _row({BK: "MagComp"}),
        _row({BK: "48123.4", CK: "999.9"}),
    ])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.n_both_filled == 1
    assert report.n_moved == 0
    assert report.out_path is None, "nothing changed, so no -m file"


def test_rows_with_an_unbalanced_quote_are_left_alone_and_counted(tmp_path):
    """An odd number of quotes means the field boundaries can't be trusted,
    so editing by column index would land in the wrong place."""
    src = _write(tmp_path, "survey.csv", [
        _row({BK: "MagComp"}),
        _row({BK: "", CK: "1.0", 10: "'unterminated"}),
    ])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.unbalanced_quote_lines == 1
    assert report.n_moved == 0


def test_short_rows_are_counted_not_crashed_on(tmp_path):
    src = _write(tmp_path, "survey.csv", [_row({BK: "MagComp"}), "a,b,c"])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert report.n_too_short == 1
    assert report.n_moved == 0


def test_varying_field_counts_are_recorded_for_the_report(tmp_path):
    """A row that is 26 fields wider than its neighbours means the whole
    tail shifted, not just MagComp - the user needs to be told, because
    then moving one column is the wrong fix."""
    src = _write(tmp_path, "survey.csv", [
        _row({BK: "MagComp"}, n_fields=95),
        _row({BK: "", CK: "1.0"}, n_fields=95 + 26),
    ])

    report = fix_file(src, BK, CK, dry_run=False, write_unchanged=False, output_dir=None)

    assert set(report.field_counts) == {95, 121}


def test_dry_run_writes_nothing(tmp_path):
    src = _write(tmp_path, "survey.csv", [_row({BK: "MagComp"}), _row({BK: "", CK: "1.0"})])

    report = fix_file(src, BK, CK, dry_run=True, write_unchanged=False, output_dir=None)

    assert report.n_moved == 1
    assert not report.out_path.exists()


# ---------------------------------------------------------------------------
# folder walk / CLI
# ---------------------------------------------------------------------------


def test_rerunning_does_not_reprocess_its_own_output(tmp_path, capsys):
    _write(tmp_path, "survey.csv", [_row({BK: "MagComp"}), _row({BK: "", CK: "1.0"})])

    assert main([str(tmp_path)]) == 0
    assert (tmp_path / "survey-m.csv").exists()
    # Second run: the -m file must not be picked up as input.
    assert main([str(tmp_path)]) == 0

    assert not (tmp_path / "survey-m-m.csv").exists()
    produced = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert produced == ["survey-m.csv", "survey.csv"]


def test_a_missing_folder_reports_instead_of_raising(capsys):
    assert main(["/no/such/folder"]) == 1
    assert "폴더를 찾을 수 없습니다" in capsys.readouterr().out


def test_columns_can_be_overridden_from_the_command_line(tmp_path, capsys):
    a, c = column_letter_to_index("A"), column_letter_to_index("C")
    (tmp_path / "s.csv").write_text("h1,h2,h3\n,b,7\n", encoding="utf-8")

    assert main([str(tmp_path), "--to-column", "A", "--from-column", "C"]) == 0

    assert (tmp_path / "s-m.csv").read_text(encoding="utf-8").splitlines()[1] == "7,b,"
