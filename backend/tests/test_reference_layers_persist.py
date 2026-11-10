"""Reference layers are part of the project: saved in the bundle with
their order, opacity and visibility, and handed back on load."""
import io
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from app.main import app
from app.store import Project, store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
KML = """<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">
<Placemark><name>BH-1</name><Point><coordinates>128.496,36.528,0</coordinates></Point></Placemark></kml>"""


def _geotiff_bytes(n=40) -> bytes:
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=n, width=n, count=3, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(128.49, 36.53, 0.0001, 0.0001)) as dst:
        dst.write(np.full((3, n, n), 120, dtype="uint8"))
    return buf.getvalue()


def _geotiff_file(path) -> pathlib.Path:
    path.write_bytes(_geotiff_bytes(300))
    return path


def _tile_folder(root) -> pathlib.Path:
    d = root / "tiles" / "15" / "28078"
    d.mkdir(parents=True)
    Image.new("RGB", (256, 256), (10, 200, 10)).save(d / "12650.png")
    return root / "tiles"


def _project(client):
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    return pid


def test_layers_survive_save_and_load_in_order_with_their_settings(tmp_path):
    client = TestClient(app)
    pid = _project(client)
    project = store.get(pid)
    before = project.changed_at

    assert client.post(f"/api/projects/{pid}/reference-layers", files={"file": ("geology.tif", _geotiff_bytes(), "image/tiff")}).status_code == 200
    assert client.post(f"/api/projects/{pid}/reference-layers", files={"file": ("drill.kml", KML.encode(), "text/xml")}).status_code == 200
    ortho = _geotiff_file(tmp_path / "ortho.tif")
    tiles = _tile_folder(tmp_path)
    raster = client.post("/api/local-rasters/register", json={"path": str(ortho), "label": "정사영상"}).json()
    folder = client.post("/api/local-tiles/register", json={"path": str(tiles), "label": "타일", "scheme": "xyz"}).json()
    assert raster["path"] == str(ortho.resolve()) and folder["path"] == str(tiles.resolve())
    assert project.changed_at > before                                  # uploads count as unsaved work

    state = [
        {"type": "raster", "name": "정사영상", "opacity": 1.0, "visible": True, "path": raster["path"]},
        {"type": "image", "name": "geology.tif", "opacity": 0.35, "visible": False},
        {"type": "tiles", "name": "타일", "opacity": 0.9, "visible": True, "path": folder["path"], "scheme": "xyz"},
        {"type": "vector", "name": "drill", "opacity": 1.0, "visible": True},
    ]
    r = client.put(f"/api/projects/{pid}/reference-layers/state", json={"layers": state})
    assert r.status_code == 200 and r.json()["n_layers"] == 4
    marked = project.changed_at
    client.put(f"/api/projects/{pid}/reference-layers/state", json={"layers": state})
    assert project.changed_at == marked                                 # the same list again is not a change

    data = client.get(f"/api/projects/{pid}/save").content if False else project.save_project_bundle()
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "reference/0.tif" in names and "reference_vectors.json" in names
    meta = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("meta.json"))
    assert meta["reference_files"] == {"geology.tif": "reference/0.tif"} and len(meta["reference_layer_state"]) == 4

    fresh = Project(id="fresh")
    result = fresh.load_project_bundle(data)
    layers = result["reference_layers"]
    assert [(l["type"], l["name"]) for l in layers] == [("raster", "정사영상"), ("image", "geology.tif"), ("tiles", "타일"), ("vector", "drill")]
    image = layers[1]
    assert image["opacity"] == 0.35 and image["visible"] is False and image["image_data_url"].startswith("data:image/png;base64,")
    assert layers[3]["features"][0]["name"] == "BH-1" and layers[3]["n_points"] == 1
    # path-based layers are registered again and serve tiles under their new ids
    assert layers[0]["raster_layer_id"] != raster["id"] and layers[0]["path"] == raster["path"]
    assert client.get(f"/api/local-tiles/{layers[2]['tile_layer_id']}/15/28078/12650").status_code == 200

    # removing a layer removes it from the saved list too
    client.delete(f"/api/projects/{pid}/reference-layers/drill")
    assert [s["name"] for s in project.reference_layer_state] == ["정사영상", "geology.tif", "타일"]


def test_a_layer_whose_file_is_gone_comes_back_flagged_not_dropped(tmp_path):
    client = TestClient(app)
    pid = _project(client)
    ortho = _geotiff_file(tmp_path / "gone.tif")
    raster = client.post("/api/local-rasters/register", json={"path": str(ortho)}).json()
    client.put(f"/api/projects/{pid}/reference-layers/state",
               json={"layers": [{"type": "raster", "name": "gone", "path": raster["path"]}]})
    data = store.get(pid).save_project_bundle()
    client.delete(f"/api/local-rasters/{raster['id']}")
    ortho.unlink()
    layers = Project(id="x").load_project_bundle(data)["reference_layers"]
    assert len(layers) == 1 and layers[0]["name"] == "gone" and "찾을 수 없습니다" in layers[0]["error"]


def test_a_bundle_from_before_layers_were_saved_leaves_the_screen_alone():
    client = TestClient(app)
    pid = _project(client)
    project = store.get(pid)
    data = project.save_project_bundle()
    # strip the reference keys, as an older version would have written it
    src = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in src.namelist():
            payload = src.read(name)
            if name == "meta.json":
                meta = json.loads(payload)
                meta.pop("reference_layer_state")
                meta.pop("reference_files")
                payload = json.dumps(meta)
            zf.writestr(name, payload)
    assert Project(id="old").load_project_bundle(buf.getvalue())["reference_layers"] is None
    # and a current bundle with no layers says so explicitly
    assert Project(id="new").load_project_bundle(data)["reference_layers"] == []
