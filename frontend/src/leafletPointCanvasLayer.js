import L from "leaflet";

// Draws all points directly onto one canvas instead of creating one
// L.circleMarker (+ event listener) per point. For a few hundred thousand
// points, per-point Leaflet Layer objects are the actual bottleneck (heavy
// allocation, internal bookkeeping, N event registrations) - a plain draw
// loop is orders of magnitude cheaper, and re-coloring on option changes
// (color scale, value field) only needs a redraw, not teardown+rebuild.
//
// Follows the same technique Leaflet's own L.Canvas renderer uses: the
// canvas is a child of the map's overlayPane, which Leaflet CSS-transforms
// as a whole during a drag, so panning is smooth "for free" without
// per-frame JS work. A full redraw only happens on moveend/zoomend, same as
// L.Canvas's _update().
const PointCanvasLayer = L.Layer.extend({
  initialize(points, options) {
    L.Util.setOptions(this, options);
    this._points = points || [];
    this._colorScale = options && options.colorScale;
    this._onHover = options && options.onHover;
    // onClick(point | null): the point under a click, or null for a click
    // on nothing. clickable() says whether clicks mean anything just now
    // (not while another map tool owns them) - it also decides the cursor.
    this._onClick = options && options.onClick;
    this._clickable = (options && options.clickable) || (() => true);
    this._drawn = [];
    this._padding = 0.15; // overscan fraction of viewport, like L.Canvas
  },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create("canvas", "leaflet-point-canvas-layer");
    const pane = map.getPane(this.options.pane || "overlayPane");
    pane.appendChild(this._canvas);
    this._ctx = this._canvas.getContext("2d");

    // Pointer events are taken from the map, not from this canvas. The
    // canvas is one of several stacked in the overlay pane - Leaflet's own
    // canvas renderer (the boundary outline, markers drawn as paths...)
    // sits on top of it and would swallow every click and hover meant for
    // a point. The map hears them all, and does not report the click that
    // ends a drag.
    this._canvas.style.pointerEvents = "none";
    if (this._onHover || this._onClick) {
      map.on("mousemove", this._handleMouseMove, this);
      map.on("mouseout", this._handleMouseOut, this);
    }
    if (this._onClick) map.on("click", this._handleClick, this);

    map.on("moveend", this._reset, this);
    map.on("zoomend", this._reset, this);
    map.on("resize", this._reset, this);
    this._reset();
    return this;
  },

  onRemove(map) {
    map.off("mousemove", this._handleMouseMove, this);
    map.off("mouseout", this._handleMouseOut, this);
    map.off("click", this._handleClick, this);
    if (this._pointerCursor) map.getContainer().style.cursor = "";
    this._canvas.remove();
    map.off("moveend", this._reset, this);
    map.off("zoomend", this._reset, this);
    map.off("resize", this._reset, this);
    return this;
  },

  setData(points, colorScale) {
    this._points = points || [];
    this._colorScale = colorScale;
    this._draw();
  },

  setColorScale(colorScale) {
    this._colorScale = colorScale;
    this._draw();
  },

  _reset() {
    if (!this._map) return;
    const size = this._map.getSize();
    const pad = size.multiplyBy(this._padding);
    const min = this._map.containerPointToLayerPoint(pad.multiplyBy(-1)).round();
    const fullSize = size.add(pad.multiplyBy(2));

    L.DomUtil.setPosition(this._canvas, min);
    this._canvas.width = fullSize.x;
    this._canvas.height = fullSize.y;
    this._canvas.style.width = `${fullSize.x}px`;
    this._canvas.style.height = `${fullSize.y}px`;
    this._origin = min;
    this._draw();
  },

  _draw() {
    if (!this._ctx || !this._map) return;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    if (!this._points.length || !this._colorScale) {
      this._drawn = [];
      return;
    }

    const map = this._map;
    const origin = this._origin;
    const drawn = new Array(this._points.length);
    let n = 0;

    for (let i = 0; i < this._points.length; i++) {
      const p = this._points[i];
      const lp = map.latLngToLayerPoint([p.lat, p.lon]);
      const x = lp.x - origin.x;
      const y = lp.y - origin.y;
      if (x < -10 || y < -10 || x > this._canvas.width + 10 || y > this._canvas.height + 10) continue;

      const excluded = p.excluded;
      const r = excluded ? 1.2 : 2;
      ctx.globalAlpha = excluded ? 0.35 : 0.9;
      ctx.fillStyle = excluded ? "#9ca3af" : this._colorScale(p.value);
      ctx.fillRect(x - r, y - r, r * 2, r * 2);
      drawn[n++] = [x, y, p];
    }
    drawn.length = n;
    ctx.globalAlpha = 1;
    this._drawn = drawn;
  },

  // the drawn point nearest a map mouse event, within `radius` pixels
  _nearest(e, radius) {
    if (!this._origin) return null;
    const x = e.layerPoint.x - this._origin.x;
    const y = e.layerPoint.y - this._origin.y;
    let closest = null;
    let minDistSq = radius * radius;
    const drawn = this._drawn;
    for (let i = 0; i < drawn.length; i++) {
      const dx = drawn[i][0] - x;
      const dy = drawn[i][1] - y;
      const dSq = dx * dx + dy * dy;
      if (dSq < minDistSq) {
        minDistSq = dSq;
        closest = drawn[i][2];
      }
    }
    return closest;
  },

  _handleMouseMove(e) {
    const closest = this._nearest(e, 8);
    if (this._onClick) {
      // a hand over a point that can be clicked; left alone while another
      // tool has set a cursor of its own
      const pointer = !!closest && this._clickable();
      if (pointer !== !!this._pointerCursor) {
        this._pointerCursor = pointer;
        if (pointer || this._clickable()) this._map.getContainer().style.cursor = pointer ? "pointer" : "";
      }
    }
    if (closest && this._onHover) this._onHover(closest);
  },

  _handleMouseOut() {
    if (this._onHover) this._onHover(null);
  },

  _handleClick(e) {
    if (!this._clickable()) return;
    this._onClick(this._nearest(e, 8));
  },
});

export default function pointCanvasLayer(points, options) {
  return new PointCanvasLayer(points, options);
}
