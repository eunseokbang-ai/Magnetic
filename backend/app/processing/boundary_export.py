"""Export a display-boundary polygon as the two formats GIS software
actually takes: a zipped ESRI shapefile and a KML.

Both are written in WGS84 lon/lat, matching the [lat, lon] rings the rest
of the app passes around, so the file needs no reprojection on import and
carries no dependency on whichever UTM zone the project happens to use.
"""
from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

import shapefile as pyshp

# WGS84 geographic, matching the lon/lat coordinates written below.
_WGS84_PRJ = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def _closed_lonlat_rings(rings_latlon: list[list[list[float]]]) -> list[list[list[float]]]:
    """[[[lat, lon], ...], ...] -> [[[lon, lat], ...], ...] with each ring
    explicitly closed. Both formats want the first vertex repeated at the
    end; the app's in-memory rings are not required to be closed."""
    if not rings_latlon:
        raise ValueError("경계 폴리곤이 비어 있습니다.")
    out = []
    for ring_latlon in rings_latlon:
        if len(ring_latlon) < 3:
            raise ValueError("경계 폴리곤은 최소 3개의 좌표가 필요합니다.")
        ring = [[float(lon), float(lat)] for lat, lon in ring_latlon]
        if ring[0] != ring[-1]:
            ring.append(list(ring[0]))
        out.append(ring)
    return out


def boundary_to_shapefile_zip(rings_latlon: list[list[list[float]]], name: str = "boundary") -> bytes:
    """Zipped .shp/.shx/.dbf/.prj polygon layer - one feature, with a
    separate part per ring so a survey split into blocks exports as one
    multi-part polygon rather than losing everything but the first."""
    rings = _closed_lonlat_rings(rings_latlon)

    shp_buf, shx_buf, dbf_buf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    writer = pyshp.Writer(shp=shp_buf, shx=shx_buf, dbf=dbf_buf, shapeType=pyshp.POLYGON)
    writer.field("name", "C", size=64)
    writer.field("n_part", "N")
    writer.field("n_vertex", "N")
    # pyshp orients each outer ring clockwise itself; disjoint outer rings
    # passed as separate parts of one shape is the standard way an ESRI
    # polygon carries a multi-part area.
    writer.poly(rings)
    writer.record(name[:64], len(rings), sum(len(r) for r in rings))
    writer.close()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}.shp", shp_buf.getvalue())
        zf.writestr(f"{name}.shx", shx_buf.getvalue())
        zf.writestr(f"{name}.dbf", dbf_buf.getvalue())
        zf.writestr(f"{name}.prj", _WGS84_PRJ)
    return zip_buf.getvalue()


def boundary_to_kml(rings_latlon: list[list[list[float]]], name: str = "boundary") -> bytes:
    """KML with one placemark, styled as a semi-transparent fill with a
    solid outline so it reads as an area (not just a line) when dropped
    into Google Earth. Several disjoint blocks become a MultiGeometry so
    they stay one named feature."""
    rings = _closed_lonlat_rings(rings_latlon)

    def polygon_xml(ring: list[list[float]], indent: str) -> str:
        # KML coordinates are lon,lat[,alt] triples, whitespace separated.
        coords = " ".join(f"{lon:.8f},{lat:.8f},0" for lon, lat in ring)
        return (
            f"{indent}<Polygon>\n"
            f"{indent}  <tessellate>1</tessellate>\n"
            f"{indent}  <outerBoundaryIs><LinearRing>\n"
            f"{indent}    <coordinates>{coords}</coordinates>\n"
            f"{indent}  </LinearRing></outerBoundaryIs>\n"
            f"{indent}</Polygon>"
        )

    if len(rings) == 1:
        geometry = polygon_xml(rings[0], "      ")
    else:
        parts = "\n".join(polygon_xml(r, "          ") for r in rings)
        geometry = f"      <MultiGeometry>\n{parts}\n      </MultiGeometry>"

    safe_name = escape(name)
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{safe_name}</name>
    <Style id="surveyBoundary">
      <LineStyle><color>ff1f63a9</color><width>2</width></LineStyle>
      <PolyStyle><color>331f63a9</color></PolyStyle>
    </Style>
    <Placemark>
      <name>{safe_name}</name>
      <styleUrl>#surveyBoundary</styleUrl>
{geometry}
    </Placemark>
  </Document>
</kml>
"""
    return kml.encode("utf-8")
