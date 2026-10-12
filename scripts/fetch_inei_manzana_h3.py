#!/usr/bin/env python3
"""
INEI Manzana Censal 2017 → H3 res-8 Lima (slr-omc prerequisito).

Fuente: Geoportal COFOPRI — capa espejo INEI Manzana Censal 2017.
  Endpoint: https://geoportal.cofopri.gob.pe/giscofopri/rest/services/Tematicos/INEI/MapServer
  Licencia: datos públicos INEI/COFOPRI.

Motivation: nbi_pct/poverty_decile/avg_household_size en el feature matrix son CONSTANTES
por distrito (1 único valor/ubigeo) → 0 aporte intra-distrital. Los datos a nivel MANZANA
tienen variación real dentro de cada distrito. Este script agrega esa variación a H3 res-8.

Campos clave del servicio (132,755 manzanas Lima):
  pobtotal  - población total por manzana
  pobnbi    - población con al menos 1 NBI
  poph      - población hombres
  pobm      - población mujeres
  vivienda  - total de viviendas
  ubigeo    - código ubigeo del distrito (6 dígitos)
  tipo      - tipo de manzana (URBANA/RURAL/etc)

Salida: data/silver/h3_features/h3_inei_manzana.parquet
Schema: h3_index, ubigeo, mzn_pop_total, mzn_nbi_pct, mzn_nbi_pop,
        mzn_vivienda, mzn_pop_hombres_pct, mzn_n_manzanas

Uso:
    python3 scripts/fetch_inei_manzana_h3.py               # descarga y procesa
    python3 scripts/fetch_inei_manzana_h3.py --raw-file /path/to/manzana.parquet
    python3 scripts/fetch_inei_manzana_h3.py --dry-run     # valida sin escribir

Dependencias (ya en el entorno principal):
    pip install geopandas shapely h3 pandas pyarrow requests tqdm
    O sin geopandas: usar centroide lat/lng de la respuesta REST (más rápido).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MATRIX_FILE = ROOT / "data/silver/h3_feature_matrix.parquet"
RAW_DIR = ROOT / "data/raw/cofopri"
OUT_DIR = ROOT / "data/silver/h3_features"
OUT_FILE = OUT_DIR / "h3_inei_manzana.parquet"

H3_RES = 8

# ArcGIS REST — capa MANZANA CENSAL dentro del MapServer de INEI en COFOPRI
ARCGIS_BASE = (
    "https://geoportal.cofopri.gob.pe/giscofopri/rest/services/Tematicos/INEI/MapServer"
)
# Layers del MapServer (verificado 2026-07-24 vía ?f=json, slr-hcai.3):
#   0=CENTRO POBLADO  1=SITIOS DE INTERES  2=EJE VIAL  3=MANZANA CENSAL  4=CASCO URBANO
#   5=DISTRITO  6=PROVINCIA  7=DEPARTAMENTO
# La capa con pobtotal/pobnbi/poph/pobm/vivienda es la 3 (antes apuntaba erróneamente a 0
# = centro poblado, sin esos campos → 400 "Failed to execute query"; causa del 33% previo).
MANZANA_LAYER_ID = 3

FIELDS = "OBJECTID,pobtotal,pobnbi,poph,pobm,vivienda,ubigeo,tipo"
PAGE_SIZE = 1000

LOGGER = logging.getLogger("fetch_inei_manzana_h3")

# Lima Metropolitana: ubigeos empiezan en '15'
LIMA_UBIGEO_PREFIX = "15"


def _query_layer(layer_id: int, offset: int, page: int) -> list[dict]:
    """Una página de features del ArcGIS REST."""
    url = f"{ARCGIS_BASE}/{layer_id}/query"
    params = {
        "where": f"ubigeo LIKE '{LIMA_UBIGEO_PREFIX}%'",
        "outFields": FIELDS,
        "returnGeometry": "true",
        "geometryType": "esriGeometryPolygon",
        "outSR": "4326",
        "resultOffset": offset,
        "resultRecordCount": page,
        # orden estable por OBJECTID: sin esto la paginación por resultOffset puede
        # repetir/saltar registros cuando exceededTransferLimit=True (causa probable de
        # que la corrida previa cubriera solo ~33% de las celdas). slr-hcai.3.
        "orderByFields": "OBJECTID",
        "f": "json",
    }
    # geoportal.cofopri.gob.pe no envía el certificado intermedio: verificar
    # primero y, solo ante SSLError, reintentar sin verificación (mismo patrón
    # que fetch_renipress_h3.py para portales gob.pe con cadena rota)
    verify = True
    for attempt in range(3):
        try:
            try:
                r = requests.get(url, params=params, timeout=60, verify=verify)
            except requests.exceptions.SSLError:
                LOGGER.warning("cadena TLS rota en %s; reintentando sin verificación", url.split("/query")[0])
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                verify = False
                r = requests.get(url, params=params, timeout=60, verify=verify)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            return data.get("features", [])
        except Exception as exc:
            if attempt == 2:
                raise
            LOGGER.warning("retry %d: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    return []


def _centroid_from_rings(rings: list) -> tuple[float, float]:
    """Centroide simple de un polígono (primer anillo)."""
    ring = rings[0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return float(np.mean(ys)), float(np.mean(xs))  # lat, lng


def download_raw(raw_dir: Path) -> pd.DataFrame:
    """Descarga todas las manzanas de Lima desde ArcGIS REST y devuelve DataFrame."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    offset = 0
    pbar = tqdm(desc="Descargando manzanas Lima", unit=" pág")
    while True:
        feats = _query_layer(MANZANA_LAYER_ID, offset, PAGE_SIZE)
        if not feats:
            break
        for f in feats:
            attr = f.get("attributes", {})
            geom = f.get("geometry", {})
            rings = geom.get("rings", []) if geom else []
            lat, lng = _centroid_from_rings(rings) if rings else (None, None)
            records.append({**attr, "_lat": lat, "_lng": lng})
        offset += len(feats)
        pbar.update(1)
        time.sleep(0.3)  # rate-limit cortés
    pbar.close()
    df = pd.DataFrame(records)
    out = raw_dir / "inei_manzana_lima_raw.parquet"
    df.to_parquet(out, index=False)
    LOGGER.info("Raw guardado: %s (%d manzanas)", out, len(df))
    return df


def load_raw(raw_file: Path) -> pd.DataFrame:
    """Carga raw ya descargado (parquet o jsonl)."""
    if raw_file.suffix == ".parquet":
        return pd.read_parquet(raw_file)
    raise ValueError(f"Formato no soportado: {raw_file.suffix}")


def aggregate_to_h3(df: pd.DataFrame, matrix_cells: set) -> pd.DataFrame:
    """
    Agrega atributos de manzana a H3 res-8.

    Por celda H3:
      mzn_pop_total     - suma pobtotal
      mzn_nbi_pop       - suma pobnbi
      mzn_nbi_pct       - mzn_nbi_pop / mzn_pop_total (ponderado por población)
      mzn_vivienda      - suma viviendas
      mzn_pop_hombres_pct - suma poph / suma pobtotal
      mzn_n_manzanas    - conteo de manzanas en la celda
    """
    # Normalizar nombres de columnas (pueden venir en mayúsculas)
    df.columns = [c.lower() for c in df.columns]

    # Asignar H3 por centroide de manzana
    lat_col = "_lat" if "_lat" in df.columns else "lat"
    lng_col = "_lng" if "_lng" in df.columns else "lng"

    # Filtrar coordenadas válidas
    df = df.dropna(subset=[lat_col, lng_col]).copy()
    df["h3_index"] = [
        h3.latlng_to_cell(lat, lng, H3_RES)
        for lat, lng in zip(df[lat_col], df[lng_col])
    ]

    # Filtrar a celdas del feature matrix
    if matrix_cells:
        df = df[df["h3_index"].isin(matrix_cells)]

    # Coerce numérico. INEI usa 9999999 como sentinel en varios campos.
    for col in ["pobtotal", "pobnbi", "poph", "pobm", "vivienda"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")
        df.loc[df[col] >= 9_999_999, col] = np.nan

    df["pobtotal"] = df["pobtotal"].fillna(0)
    df["poph"] = df["poph"].fillna(0)
    df["pobm"] = df["pobm"].fillna(0)
    df["vivienda"] = df["vivienda"].fillna(0)

    # En la capa COFOPRI/INEI, pobnbi se observa como porcentaje de población
    # con NBI (0..100), no como conteo. Convertimos a población equivalente
    # para poder promediar ponderado por población dentro del H3.
    nbi_pct_raw = df["pobnbi"].where(df["pobnbi"].between(0, 100), np.nan)
    df["_mzn_nbi_pop"] = (df["pobtotal"] * nbi_pct_raw.fillna(0) / 100.0).fillna(0)

    # Usar el ubigeo del primer registro de la manzana por H3 (modal o primero)
    ubigeo_per_h3 = (
        df.groupby("h3_index")["ubigeo"]
        .agg(lambda x: x.mode().iloc[0] if len(x) else "")
        .reset_index()
    )

    agg = (
        df.groupby("h3_index")
        .agg(
            mzn_pop_total=("pobtotal", "sum"),
            mzn_nbi_pop=("_mzn_nbi_pop", "sum"),
            mzn_vivienda=("vivienda", "sum"),
            mzn_pop_hombres=("poph", "sum"),
            mzn_n_manzanas=("pobtotal", "count"),
        )
        .reset_index()
    )
    agg = agg.merge(ubigeo_per_h3, on="h3_index", how="left")

    # Ratios (evitar división por 0)
    total = agg["mzn_pop_total"].replace(0, np.nan)
    agg["mzn_nbi_pct"] = (agg["mzn_nbi_pop"] / total).fillna(0.0).astype("float32")
    agg["mzn_pop_hombres_pct"] = (agg["mzn_pop_hombres"] / total).fillna(0.5).astype("float32")

    # Cast tipos
    agg["mzn_pop_total"] = agg["mzn_pop_total"].astype("int32")
    agg["mzn_nbi_pop"] = agg["mzn_nbi_pop"].round().astype("int32")
    agg["mzn_vivienda"] = agg["mzn_vivienda"].astype("int32")
    agg["mzn_n_manzanas"] = agg["mzn_n_manzanas"].astype("int16")

    return agg[[
        "h3_index", "ubigeo",
        "mzn_pop_total", "mzn_nbi_pct", "mzn_nbi_pop",
        "mzn_vivienda", "mzn_pop_hombres_pct", "mzn_n_manzanas",
    ]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-file", type=Path, default=None,
                    help="Parquet raw ya descargado (salta la descarga)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Valida y muestra estadísticas sin escribir salida")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s: %(message)s")

    # Cargar o descargar raw
    if args.raw_file:
        LOGGER.info("Cargando raw desde %s", args.raw_file)
        raw = load_raw(args.raw_file)
    else:
        default_raw = RAW_DIR / "inei_manzana_lima_raw.parquet"
        if default_raw.exists():
            LOGGER.info("Reutilizando raw existente: %s", default_raw)
            raw = load_raw(default_raw)
        else:
            LOGGER.info("Descargando desde ArcGIS REST COFOPRI…")
            raw = download_raw(RAW_DIR)

    LOGGER.info("Raw: %d filas, columnas: %s", len(raw), list(raw.columns)[:10])

    # Celdas del feature matrix
    matrix_cells: set = set()
    if MATRIX_FILE.exists():
        import pyarrow.parquet as pq
        matrix_cells = set(
            pd.read_parquet(MATRIX_FILE, columns=["h3_index"])["h3_index"].unique()
        )
        LOGGER.info("Feature matrix: %d celdas H3", len(matrix_cells))

    agg = aggregate_to_h3(raw, matrix_cells)
    LOGGER.info("Agregado: %d celdas H3 con datos de manzana", len(agg))

    # Estadísticas de aceptación
    print("\n=== Estadísticas ===")
    print(f"Celdas H3 con datos manzana: {len(agg):,}")
    if matrix_cells:
        cobertura = len(agg) / len(matrix_cells)
        print(f"Cobertura del feature matrix: {cobertura:.1%}")

    print(f"mzn_nbi_pct mediana: {agg['mzn_nbi_pct'].median():.3f}")
    print(f"mzn_nbi_pct std: {agg['mzn_nbi_pct'].std():.3f}")

    # Verificar variación intra-distrital (propósito principal del script)
    nuniq = agg.groupby("ubigeo")["mzn_nbi_pct"].nunique()
    multi = (nuniq > 1).sum()
    print(f"\nTest variación intra-distrital:")
    print(f"  Distritos con >1 valor único de mzn_nbi_pct: {multi}/{len(nuniq)}")
    print(f"  → {'PASA' if multi > 0 else 'FALLA'} (objetivo: romper constancia)")

    if args.dry_run:
        LOGGER.info("DRY-RUN: no se escribe.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_FILE, index=False)
    LOGGER.info("Guardado → %s", OUT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
