"""render_section_png used to write matplotlib.rcParams["font.family"] and
["axes.unicode_minus"] directly, which is process-global, permanent state -
after the first call, every other matplotlib render in the process (or a
concurrently-running one, since FastAPI's sync endpoints run in a real OS
threadpool) would stay stuck on NanumGothic/no-unicode-minus. Wrapping the
render in matplotlib.rc_context(...) scopes the override to just this call
and restores whatever was there before on exit."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

from app.processing.inversion import render_section_png


def test_rcparams_restored_after_render():
    prior_font = matplotlib.rcParams["font.family"]
    prior_minus = matplotlib.rcParams["axes.unicode_minus"]
    assert prior_font != ["NanumGothic"]

    n_layers, n_samples = 5, 8
    section = np.random.default_rng(0).uniform(0.0, 0.1, size=(n_layers, n_samples))
    distance_m = np.linspace(0.0, 70.0, n_samples)
    z_centers = np.linspace(-40.0, -5.0, n_layers)[::-1]
    ground_elevation_m = np.full(n_samples, 0.0)
    path_x = np.linspace(0.0, 70.0, n_samples)
    path_y = np.zeros(n_samples)
    mesh_x_centers = np.linspace(-10.0, 80.0, 20)
    mesh_y_centers = np.linspace(-10.0, 10.0, 10)

    result = render_section_png(
        section, distance_m, z_centers, ground_elevation_m, path_x, path_y,
        mesh_x_centers, mesh_y_centers, cmap_name="viridis",
    )

    assert result["image_data_url"].startswith("data:image/png;base64,")
    assert matplotlib.rcParams["font.family"] == prior_font
    assert matplotlib.rcParams["axes.unicode_minus"] == prior_minus


if __name__ == "__main__":
    test_rcparams_restored_after_render()
    print("ALL CHECKS PASSED")
