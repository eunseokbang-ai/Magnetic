"""Bringing a draped survey to one flight height, against a known answer.

The synthetic is the case the feature exists for: a compact source under
a hill and a broad geology, flown at a constant clearance over 60 m of
relief, so neighbouring readings differ in height by tens of metres. The
truth is the same model evaluated on the level reference surface.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing import height_normalization as hn

INC, DEC = 50.0, -8.0
SPACING = 50.0


def _field(x, y, z, f, dipoles):
    out = np.zeros(len(x))
    for sx, sy, sz, m in dipoles:
        rx, ry, rz = x - sx, y - sy, z - sz
        r2 = rx * rx + ry * ry + rz * rz
        fr = f[0] * rx + f[1] * ry + f[2] * rz
        out += 100.0 * m * (3 * fr * fr - r2) / (r2 * r2 * np.sqrt(r2))
    return out


def _survey(relief=30.0, n_lines=40, length=1500.0, step=0.65, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    f = hn._field_direction(INC, DEC)
    xs, ys, zs, gid = [], [], [], []
    for i in range(n_lines):
        y = np.arange(0.0, length, step)
        if i % 2:
            y = y[::-1]
        x = np.full_like(y, i * SPACING)
        ground = relief * np.sin(2 * np.pi * x / 1500.0) * np.cos(2 * np.pi * y / 1200.0)
        xs.append(x)
        ys.append(y)
        zs.append(ground + 50.0)
        gid.append(np.full(len(y), i))
    x, y, z, gid = map(np.concatenate, (xs, ys, zs, gid))
    cx, cy = 1000.0, 750.0
    g_c = relief * np.sin(2 * np.pi * cx / 1500.0) * np.cos(2 * np.pi * cy / 1200.0)
    dipoles = [(cx, cy, g_c - 40.0, 1e6)]                      # 170 nT at the drone
    for gx in np.arange(-700, 2800, 700):
        for gy in np.arange(-700, 2300, 700):
            dipoles.append((gx + rng.uniform(-100, 100), gy + rng.uniform(-100, 100), -300.0, rng.normal(0, 5e7)))
    d = _field(x, y, z, f, dipoles) + rng.normal(0, noise, len(x))
    return x, y, z, gid, d, f, dipoles


@pytest.fixture(scope="module")
def hilly():
    return _survey()


def test_the_interior_error_drops_to_a_fraction_of_what_it_was(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    z_ref = float(np.median(z))
    truth = _field(x, y, np.full(len(x), z_ref), f, dipoles)

    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC, z_ref_m=z_ref)

    inner = (gid >= 3) & (gid <= gid.max() - 3) & (y > 100) & (y < 1400)
    before = np.sqrt(np.mean((d - truth)[inner] ** 2))
    after = np.sqrt(np.mean((d + r.correction_nt - truth)[inner] ** 2))
    assert before > 6.0                                   # the height effect is real
    assert after < before / 3.0                           # and mostly gone
    assert r.converged and r.n_iterations <= hn._MAX_ITER
    assert r.fit_rms_nt < 0.05 * r.data_rms_nt
    assert not r.warnings


def test_the_compact_source_under_the_hill_is_corrected_too(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    z_ref = float(np.median(z))
    truth = _field(x, y, np.full(len(x), z_ref), f, dipoles)
    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC, z_ref_m=z_ref)

    near = np.hypot(x - 1000, y - 750) < 300
    before = np.sqrt(np.mean((d - truth)[near] ** 2))
    after = np.sqrt(np.mean((d + r.correction_nt - truth)[near] ** 2))
    assert before > 15.0
    assert after < before / 4.0


def test_the_outermost_line_is_not_made_worse(hilly):
    """The layer has data on one side only there, so its vertical
    gradient is wrong; the taper keeps that from reaching the data."""
    x, y, z, gid, d, f, dipoles = hilly
    z_ref = float(np.median(z))
    truth = _field(x, y, np.full(len(x), z_ref), f, dipoles)
    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC, z_ref_m=z_ref)

    outer = (gid < 1) | (gid > gid.max() - 1) | (y <= 50) | (y >= 1450)
    before = np.sqrt(np.mean((d - truth)[outer] ** 2))
    after = np.sqrt(np.mean((d + r.correction_nt - truth)[outer] ** 2))
    assert after <= before
    assert 0 < r.n_points_edge_tapered < 0.2 * len(x)


def test_a_reading_at_the_reference_height_is_left_alone(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    z_ref = float(np.median(z))
    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC, z_ref_m=z_ref)

    at_ref = np.abs(z - z_ref) < 0.5
    inner = (gid >= 3) & (gid <= gid.max() - 3) & (y > 100) & (y < 1400)
    assert at_ref.sum() > 100
    assert np.median(np.abs(r.correction_nt[at_ref & inner])) < 0.3
    # and a whole flight at one height: nothing at all
    assert np.allclose(r.correction_at(x[:50], y[:50], np.full(50, z_ref)), 0.0)


def test_a_flat_flight_is_hardly_changed():
    x, y, z, gid, d, f, dipoles = _survey(relief=1.0)

    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC)

    assert r.altitude["std"] < 1.0
    assert np.median(np.abs(r.correction_nt)) < 0.2       # the regional's own 0.2 nT/m, nothing else
    assert any("평면에 가깝" in w for w in r.warnings)


def test_readings_without_a_height_get_no_correction_and_are_counted(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    z = z.copy()
    z[::97] = np.nan

    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC)

    assert r.n_points_uncorrected == int(np.isnan(z).sum())
    assert np.all(r.correction_nt[np.isnan(z)] == 0.0)
    assert np.abs(r.correction_nt[~np.isnan(z)]).max() > 1.0
    assert any("보정하지 않았습니다" in w for w in r.warnings)


def test_only_the_masked_readings_are_fitted(hilly):
    """Readings excluded from the fit still get a correction - the layer
    is evaluated at them - but they did not shape it."""
    x, y, z, gid, d, f, dipoles = hilly
    bad = gid == 20
    spoiled = d.copy()
    spoiled[bad] += 500.0

    r = hn.fit_height_normalization(x, y, z, spoiled, gid, SPACING, INC, DEC, fit_mask=~bad)
    clean = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC)

    assert r.n_blocks_fitted < clean.n_blocks_fitted
    assert np.abs(r.correction_nt[bad]).max() > 0.0
    assert abs(r.fit_rms_nt - clean.fit_rms_nt) < 0.5


def test_no_heights_at_all_is_a_clear_error(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    with pytest.raises(ValueError, match="고도"):
        hn.fit_height_normalization(x, y, np.full(len(x), np.nan), d, gid, SPACING, INC, DEC)


def test_a_reference_below_the_layer_is_refused_and_one_outside_the_flight_is_flagged(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    with pytest.raises(ValueError, match="등가층"):
        hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC, z_ref_m=float(z.min()) - 100.0)

    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC, z_ref_m=float(z.max()) + 20.0)
    assert any("범위" in w for w in r.warnings)


def test_blocks_follow_the_track_and_interpolate_back():
    x = np.concatenate([np.arange(0, 100, 0.5), np.full(200, 50.0)])
    y = np.concatenate([np.zeros(200), np.arange(0, 100, 0.5)])
    z = np.full(400, 10.0)
    group = np.repeat([0, 1], 200)

    blocks = hn.block_tracks(x, y, z, group, block_size_m=10.0)

    assert len(blocks) == 20
    assert set(blocks.group) == {0, 1}
    assert np.allclose(blocks.count, 20)
    back = blocks.to_points(blocks.x + blocks.y)
    assert np.abs(back - (x + y)).max() < 5.0             # within a block of the truth
    assert np.abs(back[10:190] - (x + y)[10:190]).max() < 1e-6   # exact between block centres


def test_saved_and_restored_is_the_same_model(hilly):
    x, y, z, gid, d, f, dipoles = hilly
    r = hn.fit_height_normalization(x, y, z, d, gid, SPACING, INC, DEC)

    back = hn.HeightNormalization.from_saved(r.to_meta(), r.to_arrays())

    assert back.summary() == r.summary()
    assert np.allclose(back.correction_nt, r.correction_nt, atol=1e-3)
    assert np.allclose(back.field_at(x[:20], y[:20], z[:20]), r.field_at(x[:20], y[:20], z[:20]))
