// Distance/area measurement for the map ruler tool. No geodesy library
// dependency: distance uses the standard haversine great-circle formula
// (matches Leaflet's own L.LatLng.distanceTo), and area projects the ring
// onto a local tangent plane centered on its own latitude (equirectangular,
// degrees -> meters via cos(lat) scaling) before applying the planar
// shoelace formula - accurate to well under 1% at survey scale (a few km),
// which is what an on-map anomaly-zone ruler needs, without requiring true
// geodesic-polygon-area code.
const EARTH_RADIUS_M = 6371008.8;

function toRad(deg) {
  return (deg * Math.PI) / 180;
}

export function haversineDistance(a, b) {
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(h));
}

export function pathLength(latlngs) {
  let total = 0;
  for (let i = 1; i < latlngs.length; i++) {
    total += haversineDistance(latlngs[i - 1], latlngs[i]);
  }
  return total;
}

function projectLocalMeters(latlngs) {
  const lat0 = latlngs.reduce((s, p) => s + p.lat, 0) / latlngs.length;
  const mPerDegLat = (Math.PI / 180) * EARTH_RADIUS_M;
  const mPerDegLng = mPerDegLat * Math.cos(toRad(lat0));
  return latlngs.map((p) => [p.lng * mPerDegLng, p.lat * mPerDegLat]);
}

export function polygonArea(latlngs) {
  if (latlngs.length < 3) return 0;
  const pts = projectLocalMeters(latlngs);
  let sum = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

export function formatDistance(m) {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(1)} m`;
}

export function formatArea(m2) {
  if (m2 >= 1e6) return `${(m2 / 1e6).toFixed(3)} km²`;
  if (m2 >= 1e4) return `${(m2 / 1e4).toFixed(3)} ha`;
  return `${m2.toFixed(1)} m²`;
}

// A small circular ring of [lat, lon] points around a center, radiusM out -
// e.g. to turn a single detected point anomaly (which the manual-smoothing
// polygon mode has no notion of) into a polygon covering its footprint.
export function circlePolygon(lat, lon, radiusM, nPoints = 16) {
  const mPerDegLat = (Math.PI / 180) * EARTH_RADIUS_M;
  const mPerDegLng = mPerDegLat * Math.cos(toRad(lat));
  const ring = [];
  for (let i = 0; i < nPoints; i++) {
    const theta = (2 * Math.PI * i) / nPoints;
    ring.push([lat + (radiusM * Math.sin(theta)) / mPerDegLat, lon + (radiusM * Math.cos(theta)) / mPerDegLng]);
  }
  return ring;
}
