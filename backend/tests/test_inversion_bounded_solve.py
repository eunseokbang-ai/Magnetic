"""The bound-constrained solve at the heart of the 3D inversion.

Until 2026-11 the inversion solved its regularized least-squares problem
without bounds and clipped the negative cells to zero afterwards. That is
not the solution of the bounded problem and not near it: on real surveys
with strong negative anomalies the clipped model missed the data by more
than the data vary (GaSaDo: 436 nT rms against 64 nT; Andong: 708 against
438). magcore.inversion._bounded_solve now solves the bounded problem
itself. Checked here against the optimality conditions, against an
independent solver, and on the survey where the defect was found (skipped
where the field data is absent - point MAGNETIC_GASADO_DIR at it).
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.optimize import lsq_linear

import magcore.inversion as inv
from magcore.inversion import InversionError, _bounded_solve, _FreeGram, _ridge_solve, _select_regularization_strength

CHI_MAX = 1.0


def problem(seed=0, n_obs=40, n_cells=160, signed=True):
    """A small ill-conditioned problem of the inversion's shape: smooth
    kernels, weights spanning orders of magnitude, and (signed=True) data
    that no non-negative model reproduces exactly."""
    rng = np.random.default_rng(seed)
    xo = np.linspace(0, 1, n_obs)[:, None]
    xc = rng.uniform(0, 1, n_cells)[None, :]
    depth = rng.uniform(0.02, 0.4, n_cells)[None, :]
    G = 50.0 * depth / ((xo - xc) ** 2 + depth ** 2) ** 1.5 * 1e-3
    p = (1.0 + 30.0 * depth.ravel()) ** 3
    truth = np.where(rng.uniform(size=n_cells) < 0.15, rng.uniform(0.2, 0.8, n_cells), 0.0)
    if signed:
        truth = truth * rng.choice([1.0, -1.0], n_cells)
    d = G @ truth + rng.normal(0, 0.02, n_obs)
    return G, d, p


def objective(G, d, p, alpha, m, m_ref=0.0):
    return float(np.sum((G @ m - d) ** 2) + alpha * np.sum((m - m_ref) ** 2 / p))


def kkt_error(G, d, p, alpha, m, m_ref=0.0):
    """How far m is from where the optimality conditions put it."""
    lam = (d - G @ m) / alpha
    return float(np.abs(np.clip(m_ref + p * (G.T @ lam), 0.0, CHI_MAX) - m).max())


def reference_solution(G, d, p, alpha, m_ref=None):
    """The same problem by scipy's bounded least squares, on the stacked system."""
    n = G.shape[1]
    m_ref = np.zeros(n) if m_ref is None else m_ref
    w = np.sqrt(alpha / p)
    A = np.vstack([G, np.diag(w)])
    b = np.concatenate([d, w * m_ref])
    return lsq_linear(A, b, bounds=(0.0, CHI_MAX), method="bvls", tol=1e-14, max_iter=5000).x


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("strength", [100.0, 1.0, 0.01])
def test_it_is_the_solution_of_the_bounded_problem(seed, strength):
    G, d, p = problem(seed)
    alpha = strength * float(np.sum(G * G)) / G.shape[1]
    m, converged = _bounded_solve(G, d, p, alpha, CHI_MAX, np.zeros(G.shape[1]))
    assert converged
    assert m.min() >= 0.0 and m.max() <= CHI_MAX
    assert kkt_error(G, d, p, alpha, m) < 1e-7
    ref = reference_solution(G, d, p, alpha)
    assert objective(G, d, p, alpha, m) <= objective(G, d, p, alpha, ref) * (1 + 1e-9)
    assert np.abs(m - ref).max() < 1e-5
    # and the bounds matter in it: cells sit on them
    assert (m == 0.0).sum() > 10


def test_with_a_reference_model_and_cells_at_the_ceiling():
    G, d, p = problem(3, signed=False)
    d = d * 6.0                                    # needs more than the ceiling allows in places
    m_ref = np.full(G.shape[1], 0.3)
    alpha = 0.05 * float(np.sum(G * G)) / G.shape[1]
    m, converged = _bounded_solve(G, d, p, alpha, CHI_MAX, m_ref)
    assert converged and (m == CHI_MAX).sum() > 0 and (m == 0.0).sum() > 0
    assert kkt_error(G, d, p, alpha, m, m_ref) < 1e-7
    ref = reference_solution(G, d, p, alpha, m_ref)
    assert np.abs(m - ref).max() < 1e-5


def test_with_no_cell_at_a_bound_it_is_the_plain_ridge_solution():
    G, d, p = problem(4)
    n = G.shape[1]
    m_ref = np.full(n, 0.5)
    alpha = 1e3 * float(np.sum(G * G)) / n         # held close to the reference, well inside the bounds
    d = G @ m_ref + 0.01 * d
    m, converged = _bounded_solve(G, d, p, alpha, CHI_MAX, m_ref)
    plain = m_ref + p * (G.T @ np.linalg.solve((G * p) @ G.T + alpha * np.eye(len(d)), d - G @ m_ref))
    assert converged and 0.0 < plain.min() and plain.max() < CHI_MAX
    assert np.allclose(m, plain, rtol=0, atol=1e-12)


def _clipped_and_bounded(seed, strength):
    G, d, p = problem(seed)
    alpha = strength * float(np.sum(G * G)) / G.shape[1]
    unconstrained = p * (G.T @ np.linalg.solve((G * p) @ G.T + alpha * np.eye(len(d)), d))
    clipped = np.clip(unconstrained, 0.0, CHI_MAX)            # what the inversion used to return
    m, converged = _bounded_solve(G, d, p, alpha, CHI_MAX, np.zeros(G.shape[1]))
    assert converged

    def rms(model):
        return float(np.sqrt(np.mean((G @ model - d) ** 2)))

    return rms(clipped), rms(m), float(np.sqrt(np.mean(d ** 2))), objective(G, d, p, alpha, m) / objective(G, d, p, alpha, clipped)


def test_clipping_the_unconstrained_answer_was_not_it():
    # data a non-negative model can explain almost entirely: the clipped
    # model misses them ten times worse than the bounded one
    clipped, bounded, data, objective_ratio = _clipped_and_bounded(seed=11, strength=0.1)
    assert bounded < 0.02 * data
    assert clipped > 5 * bounded
    assert objective_ratio < 0.1

    # data dominated by negative anomalies, which no non-negative model
    # explains: the bounded model can do no better than stay near zero and
    # says so in its misfit; the clipped one is worse than no model at all
    clipped, bounded, data, objective_ratio = _clipped_and_bounded(seed=5, strength=1.0)
    assert bounded <= data < clipped
    assert objective_ratio < 0.95


def test_the_answer_does_not_depend_on_where_it_starts_or_on_the_shared_cache():
    G, d, p = problem(6)
    n = G.shape[1]
    scale = float(np.sum(G * G)) / n
    rng = np.random.default_rng(1)
    gram = _FreeGram(G, p)
    previous = rng.uniform(0, 1, n)                # nothing like a solution
    for strength in (10.0, 0.1, 3.0, 0.02):
        cold, ok_cold = _bounded_solve(G, d, p, strength * scale, CHI_MAX, np.zeros(n))
        warm, ok_warm = _bounded_solve(G, d, p, strength * scale, CHI_MAX, np.zeros(n), gram=gram, m_start=previous)
        assert ok_cold and ok_warm
        assert np.abs(cold - warm).max() < 1e-7
        previous = warm


def test_misfit_never_falls_as_regularization_grows():
    G, d, p = problem(7)
    scale = float(np.sum(G * G)) / G.shape[1]
    misfits = []
    for strength in np.geomspace(1e-3, 1e4, 15):
        m, converged = _bounded_solve(G, d, p, strength * scale, CHI_MAX, np.zeros(G.shape[1]))
        assert converged
        misfits.append(float(np.sqrt(np.mean((G @ m - d) ** 2))))
    assert np.all(np.diff(misfits) >= -1e-9)


def _misfit_at(G, d, p, strength):
    m, _ = _bounded_solve(G, d, p, strength * float(np.sum(G * G)) / G.shape[1], CHI_MAX, np.zeros(G.shape[1]))
    return float(np.sqrt(np.mean((G @ m - d) ** 2)))


def test_the_search_lands_on_a_target_that_can_be_reached():
    G, d, p = problem(8, signed=False)
    scale = float(np.sum(G * G)) / G.shape[1]
    floor, ceiling = _misfit_at(G, d, p, 1e-3), _misfit_at(G, d, p, 1e3)
    target = float(np.sqrt(floor * ceiling))
    strength = _select_regularization_strength(G, d, p, scale, CHI_MAX, target, np.zeros(G.shape[1]))
    assert _misfit_at(G, d, p, strength) == pytest.approx(target, rel=0.02)


def test_the_search_does_not_chase_a_target_that_cannot_be_reached():
    """Signed data, a non-negative model: no strength fits to 1 % of the
    data. The search gives up where weakening stops paying, well above
    the weakest strength it is allowed, and fits nearly as well there."""
    G, d, p = problem(9)
    scale = float(np.sum(G * G)) / G.shape[1]
    floor = _misfit_at(G, d, p, inv._SEARCH_WEAKEST)
    target = 0.01 * float(np.sqrt(np.mean(d ** 2)))
    assert floor > 5 * target
    strength = _select_regularization_strength(G, d, p, scale, CHI_MAX, target, np.zeros(G.shape[1]))
    assert strength > 10 * inv._SEARCH_WEAKEST
    assert _misfit_at(G, d, p, strength) < 1.6 * floor


def test_a_solve_that_did_not_converge_is_refused(monkeypatch):
    G, d, p = problem(10)
    monkeypatch.setattr(inv, "_BOUNDED_SOLVE_MAX_STEPS", 2)
    alpha = 1e-3 * float(np.sum(G * G)) / G.shape[1]
    m, converged = _bounded_solve(G, d, p, alpha, CHI_MAX, np.zeros(G.shape[1]))
    assert not converged and m.min() >= 0.0 and m.max() <= CHI_MAX
    with pytest.raises(InversionError):
        _ridge_solve(G, d, p, alpha, CHI_MAX, np.zeros(G.shape[1]))


# ---------------------------------------------------------------------------
# The survey the defect was found on: the 24 flights of 2026-11-03..11 with
# the 5 m terrain model. 436 nT with the clip.
# ---------------------------------------------------------------------------
GASADO = pathlib.Path(os.environ.get("MAGNETIC_GASADO_DIR") or pathlib.Path(__file__).resolve().parents[2] / "GaSaDo")
DTM = GASADO / "dtm" / "Terra_DEM_KOREA_Sea_Islands_5m.tif"


@pytest.mark.skipif(not (list(GASADO.glob("SRVY0-ACQU*.csv")) and DTM.exists()), reason=f"GaSaDo field data or terrain model not present ({GASADO})")
def test_gasado_inversion_fits_the_data():
    from fastapi.testclient import TestClient

    from app.main import app
    from app.store import store

    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    number = lambda f: int(f.name.split("ACQU")[1].split("-")[0])  # noqa: E731
    files = sorted((f for f in GASADO.glob("SRVY0-ACQU*.csv") if 85 <= number(f) <= 108), key=number)
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files])
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/upload/dem", files={"file": (DTM.name, DTM.read_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/inversion", json={"dem_geoid_offset_m": 19.17})
    assert r.status_code == 200, r.text
    summary = r.json()
    result = store.get(pid).inversion_result
    data_std = float(np.std(result.observed_nt))
    assert data_std > 40
    assert summary["rms_misfit_nt"] < 0.15 * data_std
    chi = result.susceptibility[result.mesh.active]
    assert chi.min() >= 0.0 and chi.max() <= 1.0
    assert summary["used_dem"] is True
    # the level these flights sit at is taken out first (see test_inversion_regional.py)
    regional = summary["regional_removal"]
    assert regional["method"] == "level" and -130 < regional["level_nt"] < -60
