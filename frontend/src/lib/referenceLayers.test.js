import { describe, expect, it } from "vitest";
import { layerFromDescriptor, layerSyncState, layersFromDescriptors, moveLayer, rasterLayer, tileFolderLayer, uploadedLayer } from "./referenceLayers";

describe("rows built from what the backend answers", () => {
  it("an uploaded GeoTIFF is the untyped kind, 80 % opaque", () => {
    const row = uploadedLayer({ name: "geology.tif", image_data_url: "data:image/png;base64,AAAA", bounds: [[36, 128], [37, 129]] });
    expect(row).toMatchObject({ name: "geology.tif", opacity: 0.8, visible: true, image_data_url: "data:image/png;base64,AAAA" });
    expect(row.type).toBeUndefined();
    expect(row.id).toBeTruthy();
  });

  it("an uploaded vector file keeps its features and counts", () => {
    const row = uploadedLayer({ type: "vector", name: "drill", features: [{ name: "BH-1" }], bounds: null, n_points: 1, n_lines: 0, n_polygons: 0 });
    expect(row).toMatchObject({ type: "vector", name: "drill", n_points: 1, opacity: 1, visible: true });
    expect(row.features).toEqual([{ name: "BH-1" }]);
  });

  it("path-based layers remember the path they were registered from", () => {
    const tiles = tileFolderLayer({ id: "t1", label: "타일", bounds: [], min_zoom: 10, max_zoom: 18, path: "D:\\tiles", scheme: "xyz" });
    expect(tiles).toMatchObject({ type: "tiles", name: "타일", tile_layer_id: "t1", path: "D:\\tiles", scheme: "xyz" });
    const raster = rasterLayer({ id: "r1", label: "정사영상", bounds: [], min_zoom: 8, max_zoom: 22, path: "D:\\ortho.tif", size_gb: 114.3, has_overviews: true, res_m: 0.02 });
    expect(raster).toMatchObject({ type: "raster", name: "정사영상", raster_layer_id: "r1", path: "D:\\ortho.tif", size_gb: 114.3 });
  });

  it("two rows never share an id", () => {
    const ids = new Set(Array.from({ length: 200 }, () => uploadedLayer({ name: "a" }).id));
    expect(ids.size).toBe(200);
  });
});

describe("layers saved with a project", () => {
  const descriptors = [
    { type: "raster", name: "정사영상", opacity: 1, visible: true, raster_layer_id: "r9", path: "D:\\ortho.tif", bounds: [] },
    { type: "image", name: "geology.tif", opacity: 0.35, visible: false, image_data_url: "data:x", bounds: [] },
    { type: "tiles", name: "gone", opacity: 1, visible: true, path: "E:\\tiles", error: "폴더를 찾을 수 없습니다" },
    { type: "vector", name: "drill", features: [], n_points: 2 },
  ];

  it("come back in the saved order with their settings", () => {
    const { layers } = layersFromDescriptors(descriptors);
    expect(layers.map((l) => [l.type, l.name])).toEqual([["raster", "정사영상"], [undefined, "geology.tif"], ["vector", "drill"]]);
    expect(layers[1]).toMatchObject({ opacity: 0.35, visible: false });
    expect(layers[0].raster_layer_id).toBe("r9"); // the id this server run issued, not the saved one
  });

  it("a layer that could not be reconnected is named, not silently dropped", () => {
    const { layers, message } = layersFromDescriptors(descriptors);
    expect(layers.some((l) => l.name === "gone")).toBe(false);
    expect(message).toContain("gone");
    expect(message).toContain("폴더를 찾을 수 없습니다");
    expect(layersFromDescriptors(descriptors.filter((d) => !d.error)).message).toBeNull();
  });

  it("missing opacity / visibility default to shown and opaque", () => {
    expect(layerFromDescriptor({ type: "vector", name: "v" })).toMatchObject({ opacity: 1, visible: true });
  });
});

describe("what is mirrored back to the project", () => {
  it("is order, opacity, visibility and paths - never the content", () => {
    const layers = [
      { id: "1", name: "geology.tif", image_data_url: "data:huge", opacity: 0.5, visible: true },
      { id: "2", name: "정사영상", type: "raster", raster_layer_id: "r1", path: "D:\\ortho.tif", opacity: 1, visible: false },
      { id: "3", name: "타일", type: "tiles", path: "D:\\tiles", scheme: "tms", opacity: 1, visible: true },
    ];
    expect(layerSyncState(layers)).toEqual([
      { type: "image", name: "geology.tif", opacity: 0.5, visible: true, path: null, scheme: null },
      { type: "raster", name: "정사영상", opacity: 1, visible: false, path: "D:\\ortho.tif", scheme: null },
      { type: "tiles", name: "타일", opacity: 1, visible: true, path: "D:\\tiles", scheme: "tms" },
    ]);
  });
});

describe("reordering", () => {
  const layers = [{ id: "a" }, { id: "b" }, { id: "c" }];

  it("up moves a row toward the end of the list (the top of the map)", () => {
    expect(moveLayer(layers, "a", "up").map((l) => l.id)).toEqual(["b", "a", "c"]);
    expect(moveLayer(layers, "c", "down").map((l) => l.id)).toEqual(["a", "c", "b"]);
  });

  it("does nothing at the ends or for an unknown row, and never mutates", () => {
    expect(moveLayer(layers, "c", "up")).toBe(layers);
    expect(moveLayer(layers, "a", "down")).toBe(layers);
    expect(moveLayer(layers, "zzz", "up")).toBe(layers);
    moveLayer(layers, "a", "up");
    expect(layers.map((l) => l.id)).toEqual(["a", "b", "c"]);
  });
});
