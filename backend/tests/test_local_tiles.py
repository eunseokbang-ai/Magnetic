"""Unit tests for processing/local_tiles.py - registering and serving an
already-tiled {z}/{x}/{y} raster pyramid straight off disk (see that
module's docstring for why this exists: very large orthophotos that can't
practically be uploaded+re-rendered through /overlay-images)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.processing import local_tiles


def _make_xyz_folder(root: pathlib.Path, zoom: int, x_range: range, y_range: range, ext: str = "png") -> None:
    for x in x_range:
        for y in y_range:
            d = root / str(zoom) / str(x)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{y}.{ext}").write_bytes(f"tile-{zoom}-{x}-{y}".encode())


@pytest.fixture(autouse=True)
def _reset_registry():
    local_tiles._registry.clear()
    yield
    local_tiles._registry.clear()


@pytest.fixture(autouse=True)
def _never_open_a_real_dialog(monkeypatch):
    """No test may open the native folder picker.

    pick_folder_dialog() opens a real, modal Tk window on whatever machine
    runs the backend - which during a test run is the developer's own
    screen. A test that reaches it does not fail; it blocks, with the rest
    of the suite behind it, until somebody notices the window and dismisses
    it. That is what it looked like when the suite appeared to hang
    part-way through on a machine that has tkinter installed.

    So Tk itself is made unavailable for the duration. A test that wants
    the dialog's behaviour patches pick_folder_dialog directly and never
    gets here; anything else fails fast down the documented "no display"
    path instead of waiting for a human.
    """
    try:
        import tkinter
    except ImportError:
        return

    def refuse(*_args, **_kwargs):
        raise RuntimeError("tests must not open the native folder dialog")

    monkeypatch.setattr(tkinter, "Tk", refuse)


def test_register_rejects_missing_folder(tmp_path):
    with pytest.raises(local_tiles.LocalTileError, match="찾을 수 없습니다"):
        local_tiles.register_local_tile_folder(str(tmp_path / "does_not_exist"))


def test_register_rejects_folder_without_zoom_subdirs(tmp_path):
    (tmp_path / "random_file.txt").write_text("not a tile folder")
    with pytest.raises(local_tiles.LocalTileError, match="타일 폴더 형식"):
        local_tiles.register_local_tile_folder(str(tmp_path))


def test_register_xyz_folder_detects_zoom_ext_and_scheme(tmp_path):
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(500, 503), y_range=range(300, 303))
    _make_xyz_folder(tmp_path, zoom=12, x_range=range(2000, 2004), y_range=range(1200, 1204))

    layer = local_tiles.register_local_tile_folder(str(tmp_path), label="테스트 정사영상")

    assert layer.label == "테스트 정사영상"
    assert layer.min_zoom == 10
    assert layer.max_zoom == 12
    assert layer.ext == "png"
    assert layer.scheme == "xyz"  # no tilemapresource.xml present -> defaults to xyz


def test_register_bounds_roughly_match_expected_tile_extent(tmp_path):
    # zoom 12, a compact 4x4 block of tiles near a known location.
    _make_xyz_folder(tmp_path, zoom=12, x_range=range(3500, 3504), y_range=range(1600, 1604))
    layer = local_tiles.register_local_tile_folder(str(tmp_path))

    south, west, north, east = layer.bounds
    expected_north, expected_west = local_tiles._tile_to_deg(3500, 1600, 12)
    expected_south, expected_east = local_tiles._tile_to_deg(3504, 1604, 12)

    assert south == pytest.approx(expected_south)
    assert west == pytest.approx(expected_west)
    assert north == pytest.approx(expected_north)
    assert east == pytest.approx(expected_east)
    assert north > south
    assert east > west


def test_register_detects_tms_scheme_from_tilemapresource_xml(tmp_path):
    (tmp_path / "tilemapresource.xml").write_text("<TileMap version='1.0.0'></TileMap>")
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(0, 2), y_range=range(0, 2))

    layer = local_tiles.register_local_tile_folder(str(tmp_path))
    assert layer.scheme == "tms"


def test_register_tms_bounds_flip_row_correctly(tmp_path):
    # In TMS, row 0 is the SOUTHERNMOST row (opposite of XYZ's row 0 = north).
    # A tile written at TMS row y=1600 sits further north than one at y=1000
    # for the same zoom - registering must flip these when computing bounds,
    # otherwise north/south would come out swapped or wrong.
    (tmp_path / "tilemapresource.xml").write_text("<TileMap></TileMap>")
    zoom = 12
    _make_xyz_folder(tmp_path, zoom=zoom, x_range=range(100, 102), y_range=range(1000, 1002))

    layer = local_tiles.register_local_tile_folder(str(tmp_path))
    assert layer.scheme == "tms"

    n = 2**zoom
    # TMS rows 1000..1001 correspond to XYZ rows (n-1-1001)..(n-1-1000).
    xyz_min_y, xyz_max_y = n - 1 - 1001, n - 1 - 1000
    expected_north, expected_west = local_tiles._tile_to_deg(100, xyz_min_y, zoom)
    expected_south, expected_east = local_tiles._tile_to_deg(102, xyz_max_y + 1, zoom)

    south, west, north, east = layer.bounds
    assert south == pytest.approx(expected_south)
    assert north == pytest.approx(expected_north)
    assert west == pytest.approx(expected_west)
    assert east == pytest.approx(expected_east)


def test_register_scheme_override_beats_autodetection(tmp_path):
    (tmp_path / "tilemapresource.xml").write_text("<TileMap></TileMap>")
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(0, 2), y_range=range(0, 2))

    layer = local_tiles.register_local_tile_folder(str(tmp_path), scheme_override="xyz")
    assert layer.scheme == "xyz"


def test_register_ignores_non_tile_sidecar_files(tmp_path):
    d = tmp_path / "8" / "5"
    d.mkdir(parents=True)
    (d / "5.png").write_bytes(b"real-tile")
    (d / "5.png.aux.xml").write_text("<PAMDataset></PAMDataset>")  # GDAL sidecar, not a tile

    layer = local_tiles.register_local_tile_folder(str(tmp_path))
    assert layer.ext == "png"


def test_get_tile_serves_bytes_for_xyz_layer(tmp_path):
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(500, 502), y_range=range(300, 302))
    layer = local_tiles.register_local_tile_folder(str(tmp_path))

    content, media_type = local_tiles.get_tile(layer.id, 10, 500, 300)
    assert content == b"tile-10-500-300"
    assert media_type == "image/png"


def test_get_tile_flips_row_for_tms_layer(tmp_path):
    (tmp_path / "tilemapresource.xml").write_text("<TileMap></TileMap>")
    zoom = 10
    tms_row_on_disk = 300
    _make_xyz_folder(tmp_path, zoom=zoom, x_range=range(500, 502), y_range=range(tms_row_on_disk, tms_row_on_disk + 2))
    layer = local_tiles.register_local_tile_folder(str(tmp_path))
    assert layer.scheme == "tms"

    n = 2**zoom
    # The on-disk file is named by its TMS row; a client always asks for the
    # XYZ row, so the XYZ row that should resolve back to this file is:
    xyz_y = n - 1 - tms_row_on_disk
    content, media_type = local_tiles.get_tile(layer.id, zoom, 500, xyz_y)
    assert content == f"tile-{zoom}-500-{tms_row_on_disk}".encode()
    assert media_type == "image/png"


def test_get_tile_returns_none_outside_zoom_range(tmp_path):
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(0, 2), y_range=range(0, 2))
    layer = local_tiles.register_local_tile_folder(str(tmp_path))

    content, _ = local_tiles.get_tile(layer.id, 5, 0, 0)
    assert content is None


def test_get_tile_returns_none_for_missing_file(tmp_path):
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(0, 2), y_range=range(0, 2))
    layer = local_tiles.register_local_tile_folder(str(tmp_path))

    content, _ = local_tiles.get_tile(layer.id, 10, 999, 999)
    assert content is None


def test_get_tile_returns_none_for_unknown_layer_id():
    content, _ = local_tiles.get_tile("no-such-layer", 10, 0, 0)
    assert content is None


def test_pick_folder_dialog_raises_clear_error_without_tkinter(monkeypatch):
    # Exercises the fallback a headless deployment hits, by making the
    # import fail the way it would there. tkinter is missing on some
    # machines and present on others, so the condition is simulated rather
    # than relied on.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tkinter":
            raise ImportError("no tkinter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(local_tiles.LocalTileError, match="tkinter"):
        local_tiles.pick_folder_dialog()


def test_the_dialog_cannot_be_opened_from_a_test():
    """The guard above, pinned - and with it pick_folder_dialog's second
    error path, for a machine that has tkinter but no display to put a
    window on. Whichever of the two applies here, reaching this function
    from a test must raise rather than wait for someone to click."""
    with pytest.raises(local_tiles.LocalTileError):
        local_tiles.pick_folder_dialog()


def test_pick_folder_endpoint_turns_a_dialog_failure_into_a_400(monkeypatch):
    # What this is really about is main.py's generic handler: LocalTileError
    # is a ValueError subclass, and the route has to come back as a 400
    # carrying its Korean message rather than a 500.
    #
    # The failure is injected instead of being provoked by a machine that
    # happens to lack tkinter. Calling the route for real would open the
    # native folder picker and block the suite on a machine that has it -
    # and on a machine that hasn't, the test would be asserting something
    # about the host rather than about the route.
    def no_dialog_here(*_args, **_kwargs):
        raise local_tiles.LocalTileError(
            "이 서버 환경에서는 폴더 선택 창을 지원하지 않습니다 (tkinter 미설치) - 경로를 직접 입력하세요."
        )

    monkeypatch.setattr(local_tiles, "pick_folder_dialog", no_dialog_here)

    r = TestClient(app).post("/api/local-tiles/pick-folder")

    assert r.status_code == 400, r.text
    assert "tkinter" in r.json()["detail"]


def test_unregister_removes_layer(tmp_path):
    _make_xyz_folder(tmp_path, zoom=10, x_range=range(0, 2), y_range=range(0, 2))
    layer = local_tiles.register_local_tile_folder(str(tmp_path))

    local_tiles.unregister_local_tile_folder(layer.id)
    content, _ = local_tiles.get_tile(layer.id, 10, 0, 0)
    assert content is None


def test_tile_to_deg_is_inverse_of_known_tile_math_invariants():
    # zoom 0 spans the whole world: tile (0,0) NW corner should be (~85.05, -180).
    lat, lon = local_tiles._tile_to_deg(0, 0, 0)
    assert lat == pytest.approx(85.0511, abs=0.01)
    assert lon == pytest.approx(-180.0)
    # tile (1,1) at zoom 1 (SE quadrant) NW corner should be the origin (0,0).
    lat2, lon2 = local_tiles._tile_to_deg(1, 1, 1)
    assert lat2 == pytest.approx(0.0, abs=1e-9)
    assert lon2 == pytest.approx(0.0, abs=1e-9)
