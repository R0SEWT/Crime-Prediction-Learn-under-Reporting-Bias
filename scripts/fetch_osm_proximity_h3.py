#!/usr/bin/env python3
"""
Build OSM proximity features for Lima H3 res-8 cells.

Output: data/silver/h3_features/h3_osm_proximity.parquet
Schema: h3_index, dist_police_km, dist_metro_km, dist_brt_km

Usage:
    python3 scripts/fetch_osm_proximity_h3.py
    python3 scripts/fetch_osm_proximity_h3.py --dry-run
    python3 scripts/fetch_osm_proximity_h3.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree

WEST, SOUTH, EAST, NORTH = -77.2, -12.3, -76.7, -11.7
BBOX = (WEST, SOUTH, EAST, NORTH)
H3_RES = 8

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data/datasets/osm/peru.gpkg"
GPKG_ZIP = ROOT / "data/datasets/peru-260623-free.gpkg.zip"
OSM_FILE = ROOT / "data/silver/h3_features/h3_osm_features.parquet"
OUT_FILE = ROOT / "data/silver/h3_features/h3_osm_proximity.parquet"

# Flat approximation is adequate at this extent for nearest-neighbor screening.
LAT_KM = 111.0
LNG_KM = 111.0 * np.cos(np.radians(-12.0))

LOGGER = logging.getLogger("fetch_osm_proximity_h3")

POLICE_LAYERS = (
    "pois",
    "points",
    "gis_osm_pois_free",
    "gis_osm_pois_a_free",
)
METRO_LAYERS = (
    "transport",
    "railways",
    "gis_osm_transport_free",
    "gis_osm_transport_a_free",
    "gis_osm_railways_free",
)
BRT_LAYERS = (
    "transport",
    "gis_osm_transport_free",
    "gis_osm_transport_a_free",
)

POLICE_FCLASSES = {"police"}
METRO_FCLASSES = {"railway_station", "subway_entrance", "halt", "railway_halt"}
BRT_FCLASSES = {"bus_stop"}
BRT_NAME_PATTERN = (
    r"metropolitano|alimentador|naranjal|izaguirre|pacifico|pacífico|independencia|"
    r"tomas valle|tomás valle|el milagro|honorio delgado|parque del trabajo|"
    r"caquet[aá]|2 de mayo|quilca|españa|estaci[oó]n central|jir[oó]n de la uni[oó]n|"
    r"tacna|estadio nacional|m[eé]xico|canad[aá]|javier prado|canaval|arambur[uú]|"
    r"domingo oru[eé]|angamos|ricardo palma|benavides|28 de julio|plaza de flores|"
    r"balta|bulevar|estadio uni[oó]n|escuela militar|ter[aá]n|rosario de villa|matellini"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSM proximity features -> H3 res-8 (Lima)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List GPKG layers and count relevant features, but do not write output",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def resolve_gpkg() -> Path:
    if GPKG.exists():
        return GPKG
    if GPKG_ZIP.exists():
        return GPKG_ZIP
    raise FileNotFoundError(f"Missing local OSM GeoPackage: {GPKG} or {GPKG_ZIP}")


def list_layers(path: Path) -> pd.DataFrame:
    layers = pyogrio.list_layers(path)
    return pd.DataFrame(layers, columns=["layer", "geometry_type"])


def available_layers(path: Path) -> set[str]:
    return set(list_layers(path)["layer"].astype(str))


def candidate_layers(layer_names: Iterable[str], existing: set[str]) -> list[str]:
    return [layer for layer in layer_names if layer in existing]


def read_layer(path: Path, layer: str) -> pd.DataFrame:
    info = pyogrio.read_info(path, layer=layer)
    fields = set(info["fields"])
    if "fclass" not in fields:
        LOGGER.warning("Layer %s has no fclass column; skipping", layer)
        return pd.DataFrame()

    columns = ["fclass"]
    if "name" in fields:
        columns.append("name")

    try:
        gdf = pyogrio.read_dataframe(path, layer=layer, bbox=BBOX, columns=columns)
    except Exception as exc:
        LOGGER.warning("Could not read layer %s: %s", layer, exc)
        return pd.DataFrame()

    if gdf.empty:
        return gdf
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def geometries_to_latlon(gdf: pd.DataFrame) -> np.ndarray:
    if gdf.empty:
        return np.empty((0, 2), dtype=np.float64)

    centroids = gdf.to_crs("EPSG:32718").geometry.centroid.to_crs("EPSG:4326")
    latlon = np.column_stack([centroids.y.to_numpy(), centroids.x.to_numpy()])
    return np.unique(np.round(latlon.astype(np.float64), 7), axis=0)


def min_dist_km(origins_latlon: np.ndarray, targets_latlon: np.ndarray) -> np.ndarray:
    if len(targets_latlon) == 0:
        return np.full(len(origins_latlon), np.nan, dtype=np.float32)

    scale = np.array([LAT_KM, LNG_KM], dtype=np.float64)
    origins = origins_latlon.astype(np.float64, copy=False) * scale
    targets = targets_latlon.astype(np.float64, copy=False) * scale
    tree = cKDTree(targets)
    dists, _ = tree.query(origins)
    return dists.astype(np.float32)


def load_h3_grid() -> tuple[list[str], np.ndarray]:
    if not OSM_FILE.exists():
        raise FileNotFoundError(f"Missing H3 grid source: {OSM_FILE}")

    df = pd.read_parquet(OSM_FILE, columns=["h3_index"])
    if df.empty:
        raise ValueError(f"H3 grid source is empty: {OSM_FILE}")

    cells = df["h3_index"].astype(str).drop_duplicates().sort_values().tolist()
    centroids = np.array([h3.cell_to_latlng(cell) for cell in cells], dtype=np.float64)
    return cells, centroids


def collect_targets(
    path: Path,
    existing: set[str],
    label: str,
    layers: Iterable[str],
    fclasses: set[str],
    post_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[np.ndarray, int, int]:
    selected_layers = candidate_layers(layers, existing)
    if not selected_layers:
        LOGGER.warning("%s: none of the candidate layers exists: %s", label, ", ".join(layers))
        return np.empty((0, 2), dtype=np.float64), 0, 0

    matched_frames = []
    raw_count = 0
    for layer in selected_layers:
        gdf = read_layer(path, layer)
        if gdf.empty:
            LOGGER.debug("%s: %s has no features in bbox", label, layer)
            continue

        raw_count += len(gdf)
        matched = gdf[gdf["fclass"].isin(fclasses)].copy()
        if post_filter is not None and not matched.empty:
            matched = post_filter(matched)
        LOGGER.debug("%s: %s matched %s features", label, layer, len(matched))
        if not matched.empty:
            matched_frames.append(matched)

    if not matched_frames:
        LOGGER.warning("%s: no matching fclass found in %s", label, ", ".join(selected_layers))
        return np.empty((0, 2), dtype=np.float64), raw_count, 0

    targets = pd.concat(matched_frames, ignore_index=True)
    return geometries_to_latlon(targets), raw_count, len(targets)


def brt_name_filter(gdf: pd.DataFrame) -> pd.DataFrame:
    if "name" not in gdf.columns:
        LOGGER.warning("BRT: name column not available; using all bus_stop features")
        return gdf

    names = gdf["name"].fillna("").astype(str)
    named = gdf[names.str.strip().ne("")].copy()
    if named.empty:
        LOGGER.warning("BRT: no bus_stop names available; using all bus_stop features")
        return gdf

    filtered = named[names.loc[named.index].str.contains(BRT_NAME_PATTERN, case=False, regex=True, na=False)]
    if filtered.empty:
        LOGGER.warning("BRT: no Metropolitano-like bus_stop names found; using all bus_stop features")
        return gdf

    LOGGER.info("BRT: using %s Metropolitano-like bus_stop features from %s named bus_stop features", len(filtered), len(named))
    return filtered


def count_targets(path: Path, existing: set[str]) -> dict[str, tuple[np.ndarray, int, int]]:
    return {
        "police": collect_targets(path, existing, "police", POLICE_LAYERS, POLICE_FCLASSES),
        "metro": collect_targets(path, existing, "metro", METRO_LAYERS, METRO_FCLASSES),
        "brt": collect_targets(path, existing, "BRT", BRT_LAYERS, BRT_FCLASSES, brt_name_filter),
    }


def dry_run(path: Path, existing: set[str]) -> None:
    LOGGER.info("OSM GeoPackage: %s", path)
    LOGGER.info("Available layers:")
    for row in list_layers(path).itertuples(index=False):
        LOGGER.info("  %s (%s)", row.layer, row.geometry_type)

    if OSM_FILE.exists():
        cells = pd.read_parquet(OSM_FILE, columns=["h3_index"])["h3_index"].astype(str).nunique()
        LOGGER.info("H3 grid source: %s (%s unique cells)", OSM_FILE, f"{cells:,}")
    else:
        LOGGER.warning("H3 grid source missing: %s", OSM_FILE)

    targets = count_targets(path, existing)
    for label, (latlon, raw_count, matched_count) in targets.items():
        LOGGER.info(
            "%s: %s matched features (%s unique target centroids; %s rows scanned)",
            label,
            f"{matched_count:,}",
            f"{len(latlon):,}",
            f"{raw_count:,}",
        )


def build_proximity(path: Path, existing: set[str]) -> pd.DataFrame:
    cells, centroids = load_h3_grid()
    targets = count_targets(path, existing)

    result = pd.DataFrame({"h3_index": cells})
    result["dist_police_km"] = min_dist_km(centroids, targets["police"][0])
    result["dist_metro_km"] = min_dist_km(centroids, targets["metro"][0])
    result["dist_brt_km"] = min_dist_km(centroids, targets["brt"][0])

    for col in ["dist_police_km", "dist_metro_km", "dist_brt_km"]:
        result[col] = result[col].astype("float32")
    return result


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    path = resolve_gpkg()
    existing = available_layers(path)

    if args.dry_run:
        dry_run(path, existing)
        return

    result = build_proximity(path, existing)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_FILE, index=False)

    LOGGER.info("Saved: %s", OUT_FILE)
    LOGGER.info("%s rows x %s columns", f"{len(result):,}", len(result.columns))
    LOGGER.info("\n%s", result.describe(include="all").round(3))


if __name__ == "__main__":
    main()
