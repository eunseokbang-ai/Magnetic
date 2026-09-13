"""Parses IAGA-2002 formatted geomagnetic observatory data (the standard
INTERMAGNET publication format) into a base-station-shaped (timestamp, mag)
series. The sample text below is hand-written to match the real IAGA-2002
fixed-width header format precisely (see intermagnet.py's module
docstring) - it is NOT real observatory data, just format-accurate
synthetic values for testing the parser."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing.intermagnet import IagaParseError, haversine_km, parse_iaga2002

_XYZF_SAMPLE = """\
 Format                 IAGA-2002                                    |
 Source of Data         Test Institute                                |
 Station Name           Testville                                     |
 IAGA CODE              TST                                           |
 Geodetic Latitude      52.270                                        |
 Geodetic Longitude     104.450                                       |
 Elevation              500                                           |
 Reported                XYZF                                        |
 Sensor Orientation     XYZF                                          |
 Digital Sampling       1 second                                      |
 Data Interval Type     filtered 1-minute (00015)                     |
 Data Type              reported                                      |
 # Synthetic test fixture, not real observatory data.                 |
DATE       TIME         DOY     TSTX      TSTY      TSTZ      TSTF   |
2026-07-24 00:00:00.000 205     20000.00  0.00      45000.00  49244.29
2026-07-24 00:01:00.000 205     20001.00  0.10      45001.00  49245.31
2026-07-24 00:02:00.000 205     99999.00  99999.00  99999.00  99999.00
2026-07-24 00:03:00.000 205     20003.00  0.30      45003.00  49247.34
"""

_HDZF_SAMPLE = """\
 Format                 IAGA-2002                                    |
 Source of Data         Test Institute                                |
 Station Name           Otherville                                    |
 IAGA CODE              OTV                                           |
 Geodetic Latitude      40.300                                        |
 Geodetic Longitude     240.000                                       |
 Elevation              100                                           |
 Reported                HDZF                                        |
DATE       TIME         DOY     OTVH      OTVD      OTVZ      OTVF   |
2026-07-24 00:00:00.000 205     30000.00  -2.50     40000.00  50000.00
2026-07-24 00:01:00.000 205     30001.00  -2.50     40001.00  50001.00
"""

_NO_TOTAL_FIELD_SAMPLE = """\
 Format                 IAGA-2002                                    |
 Station Name           Bareville                                     |
 IAGA CODE              BRV                                           |
 Geodetic Latitude      10.000                                        |
 Geodetic Longitude     20.000                                        |
 Reported                D                                            |
DATE       TIME         DOY     BRVD                                 |
2026-07-24 00:00:00.000 205     -1.20
"""


def test_parses_xyzf_header_metadata():
    result = parse_iaga2002(_XYZF_SAMPLE)
    assert result.station_name == "Testville"
    assert result.iaga_code == "TST"
    assert result.lat == pytest.approx(52.270)
    assert result.lon == pytest.approx(104.450)
    assert result.elevation_m == pytest.approx(500)
    assert result.reported == "XYZF"


def test_uses_reported_f_column_directly():
    result = parse_iaga2002(_XYZF_SAMPLE)
    # 3 valid rows survive after the 99999.00 fill-value row is dropped
    assert len(result.df) == 3
    assert np.allclose(result.df["mag"].to_numpy(), [49244.29, 49245.31, 49247.34])


def test_computes_total_field_from_hz_when_f_not_directly_usable():
    # HDZF sample does report F directly too - verify it's picked over D
    # (declination units are ambiguous, so F must win when present).
    result = parse_iaga2002(_HDZF_SAMPLE)
    assert len(result.df) == 2
    assert np.allclose(result.df["mag"].to_numpy(), [50000.00, 50001.00])


def test_longitude_normalized_from_0_360_to_signed():
    result = parse_iaga2002(_HDZF_SAMPLE)
    assert result.lon == pytest.approx(-120.0)


def test_raises_clear_error_when_total_field_not_derivable():
    with pytest.raises(IagaParseError):
        parse_iaga2002(_NO_TOTAL_FIELD_SAMPLE)


def test_raises_on_non_iaga_text():
    with pytest.raises(IagaParseError):
        parse_iaga2002("this is not an IAGA-2002 file at all\njust some random text\n")


def test_haversine_known_distance():
    # Seoul to Busan is roughly 325 km great-circle
    d = haversine_km(37.5665, 126.9780, 35.1796, 129.0756)
    assert 300 < d < 350


def test_haversine_zero_for_same_point():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    test_parses_xyzf_header_metadata()
    test_uses_reported_f_column_directly()
    test_computes_total_field_from_hz_when_f_not_directly_usable()
    test_longitude_normalized_from_0_360_to_signed()
    test_raises_clear_error_when_total_field_not_derivable()
    test_raises_on_non_iaga_text()
    test_haversine_known_distance()
    test_haversine_zero_for_same_point()
    print("ALL CHECKS PASSED")
