"""Vector reference layers (KML/KMZ/GeoJSON/CSV points, lines, polygons)
through magcore.vector_layer and the two upload endpoints."""
import io
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from magcore.vector_layer import VectorLayerError, load_vector_layer

# the shape Google Earth Pro writes: several points under one placemark
KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Folder>
<Placemark><name>36.5280, 128.4960</name><MultiGeometry>
<Point><coordinates>128.496,36.528,0</coordinates></Point>
<Point><coordinates>128.4849,36.5283,0</coordinates></Point>
</MultiGeometry></Placemark>
<Placemark><name>line</name><LineString><coordinates>128.48,36.52,0 128.49,36.53,0</coordinates></LineString></Placemark>
<Placemark><name>block</name><ExtendedData><Data name="unit"><value>granite</value></Data></ExtendedData>
<Polygon><outerBoundaryIs><LinearRing><coordinates>128.48,36.52 128.49,36.52 128.49,36.53 128.48,36.52</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
</Folder></Document></kml>"""


def test_kml_multigeometry_lines_polygons_and_properties():
    v = load_vector_layer(KML.encode(), "drill.kml")
    assert v["format"] == "kml" and v["name"] == "drill"
    assert (v["n_points"], v["n_lines"], v["n_polygons"]) == (2, 1, 1)
    pts = [f for f in v["features"] if f["geometry"] == "Point"]
    assert pts[0]["coordinates"] == [[36.528, 128.496]] and pts[0]["name"].endswith("#1")      # [lat, lon]
    assert pts[1]["coordinates"] == [[36.5283, 128.4849]]
    poly = next(f for f in v["features"] if f["geometry"] == "Polygon")
    assert poly["properties"] == {"unit": "granite"} and len(poly["coordinates"]) == 4
    assert v["bounds"] == [[36.52, 128.48], [36.53, 128.496]]


def test_kmz_is_the_zipped_kml():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("doc.kml", KML)
    v = load_vector_layer(buf.getvalue(), "drill.kmz")
    assert v["format"] == "kml" and v["n_points"] == 2


def test_geojson_and_csv():
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "BH-1", "depth_m": 120}, "geometry": {"type": "Point", "coordinates": [128.496, 36.528]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPoint", "coordinates": [[128.48, 36.52], [128.49, 36.53]]}},
    ]}
    v = load_vector_layer(json.dumps(gj).encode(), "holes.geojson")
    assert v["n_points"] == 3 and v["features"][0]["name"] == "BH-1" and v["features"][0]["properties"]["depth_m"] == 120
    assert v["features"][1]["name"] == "feature 2 #1"
    csv = "﻿시추공,위도,경도,심도\nBH-1,36.528,128.496,120\nBH-2,36.5283,128.4849,80\n"
    v = load_vector_layer(csv.encode("utf-8"), "holes.csv")
    assert v["n_points"] == 2 and v["features"][1]["name"] == "BH-2" and v["features"][1]["properties"] == {"시추공": "BH-2", "심도": "80"}
    with pytest.raises(VectorLayerError, match="위경도 범위"):
        load_vector_layer(b"name,lat,lon\nA,437333,152642\n", "utm.csv")
    with pytest.raises(VectorLayerError, match="위도/경도 열"):
        load_vector_layer(b"name,x,y\nA,1,2\n", "xy.csv")


def test_upload_endpoints_route_vectors_and_rasters():
    client = TestClient(app)
    r = client.post("/api/vector-layers", files={"file": ("drill.kml", KML.encode(), "application/vnd.google-earth.kml+xml")})
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "vector" and r.json()["n_points"] == 2
    pid = client.post("/api/projects").json()["project_id"]
    r = client.post(f"/api/projects/{pid}/reference-layers", files={"file": ("drill.kml", KML.encode(), "text/xml")})
    assert r.status_code == 200 and r.json()["type"] == "vector"
    bad = client.post("/api/vector-layers", files={"file": ("x.bin", b"\x00\x01", "application/octet-stream")})
    assert bad.status_code == 400 and "벡터" in bad.json()["detail"]
    assert client.delete(f"/api/projects/{pid}/reference-layers/drill").status_code == 200
