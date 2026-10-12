#!/usr/bin/env python3
"""
Build OSM road-network and POI features for Lima Metro, aggregated to H3 res-8.

Default input is the local Geofabrik/OSM GeoPackage already present in the repo:
data/datasets/peru-260623-free.gpkg.zip

Output: data/silver/h3_features/h3_osm_features.parquet
Schema: h3_index, road_density_km_km2, intersection_count,
        poi_count_{retail,food,finance,transport,healthcare,education,nightlife}

Usage:
    python3 scripts/fetch_osm_h3.py              # roads + POIs from local GPKG
    python3 scripts/fetch_osm_h3.py --skip-pois  # road features only
    python3 scripts/fetch_osm_h3.py --dry-run    # count H3 cells only
"""

import argparse
from collections import Counter
from pathlib import Path

import h3
import pandas as pd
import pyogrio

# Lima Metropolitana bbox (WGS84)
WEST, SOUTH, EAST, NORTH = -77.2, -12.3, -76.7, -11.7
BBOX = (WEST, SOUTH, EAST, NORTH)
H3_RES = 8

ROOT = Path(__file__).resolve().parents[1]
OSM_GPKG = ROOT / "data/datasets/osm/peru.gpkg"
OSM_GPKG_ZIP = ROOT / "data/datasets/peru-260623-free.gpkg.zip"
OUT_DIR = ROOT / "data/silver/h3_features"

POI_CATEGORIES = {
    "retail": {
        "mall",
        "market_place",
        "supermarket",
        "convenience",
        "department_store",
        "clothes",
        "bakery",
        "butcher",
        "greengrocer",
        "kiosk",
        "shop",
    },
    "food": {
        "restaurant",
        "cafe",
        "fast_food",
        "food_court",
        "bar",
        "pub",
    },
    "finance": {
        "bank",
        "atm",
    },
    "transport": {
        "bus_stop",
        "bus_station",
        "taxi",
        "ferry_terminal",
        "parking",
        "fuel",
    },
    "healthcare": {
        "hospital",
        "clinic",
        "doctors",
        "dentist",
        "pharmacy",
        "veterinary",
    },
    "education": {
        "school",
        "university",
        "college",
        "kindergarten",
        "library",
    },
    "nightlife": {
        "nightclub",
        "bar",
        "pub",
        "biergarten",
    },
}

POI_LAYERS = [
    "gis_osm_pois_free",
    "gis_osm_pois_a_free",
    "gis_osm_transport_free",
    "gis_osm_transport_a_free",
    "gis_osm_traffic_free",
    "gis_osm_traffic_a_free",
]


def lima_cells(west: float, south: float, east: float, north: float, res: int) -> set[str]:
    """Return all H3 cells covering the Lima bbox."""
    geojson = {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }
    return set(h3.geo_to_cells(geojson, res))


def latlng_to_h3(lat: float, lng: float, res: int = H3_RES) -> str:
    return h3.latlng_to_cell(lat, lng, res)


def read_layer(layer: str, columns: list[str] | None = None):
    path = OSM_GPKG if OSM_GPKG.exists() else OSM_GPKG_ZIP
    return pyogrio.read_dataframe(path, layer=layer, bbox=BBOX, columns=columns)


def iter_line_endpoints(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        coords = geometry.coords
        if len(coords) >= 2:
            yield coords[0]
            yield coords[-1]
    elif geometry.geom_type == "MultiLineString":
        for part in geometry.geoms:
            yield from iter_line_endpoints(part)


def fetch_road_features(res: int = H3_RES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate local OSM road lines to H3 road density and endpoint-degree intersections."""
    source = OSM_GPKG if OSM_GPKG.exists() else OSM_GPKG_ZIP
    print(f"Reading local OSM roads from {source.name}...")
    roads = read_layer("gis_osm_roads_free", columns=["fclass", "geometry"])
    roads = roads[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    print(f"  Road segments in bbox: {len(roads):,}")

    roads_proj = roads.to_crs("EPSG:32718")
    roads["length_km"] = roads_proj.geometry.length.to_numpy() / 1000.0
    centroids = roads_proj.geometry.centroid.to_crs("EPSG:4326")
    roads["h3_index"] = [latlng_to_h3(pt.y, pt.x, res) for pt in centroids]

    h3_area = h3.average_hexagon_area(res, unit="km^2")
    road_agg = (
        roads.groupby("h3_index", as_index=False)["length_km"]
        .sum()
        .rename(columns={"length_km": "road_length_km"})
    )
    road_agg["road_density_km_km2"] = road_agg["road_length_km"] / h3_area

    endpoint_degree: Counter[tuple[float, float]] = Counter()
    for geom in roads.geometry:
        for lon, lat, *_ in iter_line_endpoints(geom) or ():
            endpoint_degree[(round(lon, 6), round(lat, 6))] += 1

    intersection_rows = [
        {"h3_index": latlng_to_h3(lat, lon, res), "degree": degree}
        for (lon, lat), degree in endpoint_degree.items()
        if degree >= 3
    ]
    if intersection_rows:
        intersections = pd.DataFrame(intersection_rows)
        inter_agg = intersections.groupby("h3_index").size().reset_index(name="intersection_count")
    else:
        inter_agg = pd.DataFrame(columns=["h3_index", "intersection_count"])

    print(f"  H3 with roads: {len(road_agg):,} · with intersections: {len(inter_agg):,}")
    return road_agg[["h3_index", "road_density_km_km2"]], inter_agg


def category_for_fclass(fclass: str | None) -> str | None:
    if not isinstance(fclass, str):
        return None
    for category, values in POI_CATEGORIES.items():
        if fclass in values:
            return category
    return None


def fetch_poi_features(res: int = H3_RES) -> pd.DataFrame:
    """Aggregate local OSM POI point/polygon layers to H3 category counts."""
    poi_frames = []
    for layer in POI_LAYERS:
        try:
            pois = read_layer(layer, columns=["fclass", "geometry"])
        except Exception as exc:
            print(f"  Warning: could not read {layer}: {exc}")
            continue
        pois = pois[~pois.geometry.is_empty & pois.geometry.notna()].copy()
        if pois.empty:
            continue
        pois["category"] = pois["fclass"].map(category_for_fclass)
        pois = pois[pois["category"].notna()].copy()
        if pois.empty:
            continue

        centroids = pois.to_crs("EPSG:32718").geometry.centroid.to_crs("EPSG:4326")
        pois["h3_index"] = [latlng_to_h3(pt.y, pt.x, res) for pt in centroids]
        poi_frames.append(pois[["h3_index", "category"]])
        print(f"  {layer}: {len(pois):,} matched POIs")

    if not poi_frames:
        return pd.DataFrame(columns=["h3_index", *[f"poi_count_{cat}" for cat in POI_CATEGORIES]])

    all_pois = pd.concat(poi_frames, ignore_index=True)
    counts = (
        all_pois.groupby(["h3_index", "category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts = counts.rename(columns={cat: f"poi_count_{cat}" for cat in POI_CATEGORIES})
    for cat in POI_CATEGORIES:
        col = f"poi_count_{cat}"
        if col not in counts.columns:
            counts[col] = 0
    return counts[["h3_index", *[f"poi_count_{cat}" for cat in POI_CATEGORIES]]]


def main():
    parser = argparse.ArgumentParser(description="OSM features -> H3 res-8 (Lima)")
    parser.add_argument("--skip-pois", action="store_true", help="Only road features")
    parser.add_argument("--skip-roads", action="store_true", help="Only POI features")
    parser.add_argument("--dry-run", action="store_true", help="Count H3 cells and exit")
    args = parser.parse_args()

    if not OSM_GPKG.exists() and not OSM_GPKG_ZIP.exists():
        raise FileNotFoundError(f"Missing local OSM GeoPackage: {OSM_GPKG} or {OSM_GPKG_ZIP}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating H3 res-{H3_RES} cells for Lima bbox...")
    cells = lima_cells(WEST, SOUTH, EAST, NORTH, H3_RES)
    print(f"  -> {len(cells):,} H3 cells")
    if args.dry_run:
        return

    result = pd.DataFrame({"h3_index": sorted(cells)})

    if not args.skip_roads:
        road_df, inter_df = fetch_road_features(H3_RES)
        result = result.merge(road_df, on="h3_index", how="left")
        result = result.merge(inter_df, on="h3_index", how="left")
        result["road_density_km_km2"] = result["road_density_km_km2"].fillna(0.0)
        result["intersection_count"] = result["intersection_count"].fillna(0).astype(int)

    if not args.skip_pois:
        print("Reading local OSM POIs...")
        poi_df = fetch_poi_features(H3_RES)
        result = result.merge(poi_df, on="h3_index", how="left")

    for cat in POI_CATEGORIES:
        col = f"poi_count_{cat}"
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0).astype(int)

    out_path = OUT_DIR / "h3_osm_features.parquet"
    result.to_parquet(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"  {len(result):,} rows x {len(result.columns)} columns")
    print(result.describe(include="all").round(2))


if __name__ == "__main__":
    main()
