#!/usr/bin/env python3
"""
Desagrega la superficie criminal latente de distrito a celdas H3 res-8.

Fuentes:
  data/datasets/silver/latent_surface/latente_distrito_categoria_anio.csv
  data/silver/h3_features/h3_poverty.parquet     (h3_index → ubigeo)
  data/silver/h3_features/h3_population.parquet  (population por celda)

Método (--pattern population, default): distribución proporcional a la población.
  latente_h3 = latente_distrito × (pop_h3 / pop_distrito_total)
  rate_h3    = latente_rate_100k_distrito  (tasa per-cápita constante dentro del distrito)

Si pop_h3 == 0 para todas las celdas del distrito → distribución uniforme.

Método (--pattern geocoded): target HÍBRIDO (slr-us4). El patrón intra-distrital
sale de las denuncias geocodificadas reales (h3_observed_geocoded.parquet) con
suavizado Dirichlet hacia el prior poblacional (pseudo-conteos --smooth M):
  w_h3 = (count_geo_h3 + M × popshare_h3) / (count_geo_distrito + M)
  latente_h3 = latente_distrito × w_h3        (por categoría × año)
Distritos/celdas sin geocodificación caen suavemente al prior poblacional
(count=0 → w = popshare). Esto rompe la circularidad del target population-only:
la variación intra-distrital del label pasa a ser concentración real de crimen.

Output: data/silver/crime_latent_surface.parquet
        data/silver/crime_latent_surface_hybrid.parquet  (--pattern geocoded)
Schema: h3_index, year, crime_cat, latente, latente_ic_low, latente_ic_high,
        latente_rate_100k, r_hat, inestable (int8), ubigeo
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LATENT_CSV = ROOT / "data/datasets/silver/latent_surface/latente_distrito_categoria_anio.csv"
POV_FILE   = ROOT / "data/silver/h3_features/h3_poverty.parquet"
POP_FILE   = ROOT / "data/silver/h3_features/h3_population.parquet"
ORACLE_FILE = ROOT / "data/silver/h3_features/h3_observed_geocoded.parquet"
OUT_FILE   = ROOT / "data/silver/crime_latent_surface.parquet"
OUT_HYBRID = ROOT / "data/silver/crime_latent_surface_hybrid.parquet"

LOGGER = logging.getLogger("build_latent_surface_h3")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", choices=["population", "geocoded"], default="population",
                    help="patrón intra-distrital: población (default) o geocodificado híbrido")
    ap.add_argument("--smooth", type=float, default=10.0,
                    help="pseudo-conteos M del suavizado Dirichlet hacia el prior poblacional")
    args = ap.parse_args()
    out_file = OUT_HYBRID if args.pattern == "geocoded" else OUT_FILE

    # --- cargar insumos ---
    latent = pd.read_csv(LATENT_CSV)
    latent["ubigeo"] = latent["ubigeo"].astype(str).str.zfill(6)
    latent["year"]   = latent["anio"].astype(int)
    LOGGER.info("Latent CSV: %d filas, distritos=%d, años=%s",
                len(latent), latent.ubigeo.nunique(), sorted(latent.year.unique()))

    pov = pd.read_parquet(POV_FILE, columns=["h3_index", "ubigeo"])
    pov = pov[pov["ubigeo"].str.len() == 6].copy()

    pop = pd.read_parquet(POP_FILE, columns=["h3_index", "population"])

    # población por celda H3
    h3_pop = pov.merge(pop, on="h3_index", how="left")
    h3_pop["population"] = pd.to_numeric(h3_pop["population"], errors="coerce").fillna(0)

    # población total por distrito
    dist_pop = h3_pop.groupby("ubigeo")["population"].sum().rename("pop_dist")
    h3_pop = h3_pop.join(dist_pop, on="ubigeo")

    # peso de cada celda dentro de su distrito
    h3_pop["weight"] = np.where(
        h3_pop["pop_dist"] > 0,
        h3_pop["population"] / h3_pop["pop_dist"],
        # fallback uniforme si no hay datos de población
        1.0 / h3_pop.groupby("ubigeo")["h3_index"].transform("count"),
    )

    # --- disaggregate ---
    numeric_cols = ["latente", "latente_ic_low", "latente_ic_high", "observado"]
    rate_cols    = ["latente_rate_100k", "r_hat"]
    flag_cols    = ["inestable"]
    keep = ["ubigeo", "year", "categoria"] + numeric_cols + rate_cols + flag_cols
    latent_clean = latent[[c for c in keep if c in latent.columns]].copy()

    merged = h3_pop[["h3_index", "ubigeo", "weight"]].merge(
        latent_clean, on="ubigeo", how="inner"
    )

    if args.pattern == "geocoded":
        # patrón híbrido: conteos geocodificados reales + suavizado Dirichlet
        # hacia el prior poblacional (weight = popshare dentro del distrito)
        oracle = pd.read_parquet(ORACLE_FILE)
        oracle["ubigeo"] = oracle["ubigeo"].astype(str).str.zfill(6)
        oracle = oracle.rename(columns={"crime_cat": "categoria"})
        # el join incluye ubigeo: una celda fronteriza puede aparecer bajo dos
        # distritos en el oráculo y sin ubigeo el merge duplica filas
        oracle = (oracle.groupby(["h3_index", "ubigeo", "year", "categoria"], as_index=False)
                  ["obs_geo_count"].sum())
        merged = merged.merge(
            oracle,
            on=["h3_index", "ubigeo", "year", "categoria"], how="left",
        )
        merged["obs_geo_count"] = merged["obs_geo_count"].fillna(0.0)
        dist_geo = merged.groupby(["ubigeo", "year", "categoria"])["obs_geo_count"].transform("sum")
        m = args.smooth
        merged["weight"] = (merged["obs_geo_count"] + m * merged["weight"]) / (dist_geo + m)
        n_geo = int((dist_geo > 0).sum())
        LOGGER.info("Patrón geocodificado: %d/%d filas en distrito-cat-año con conteos "
                    "reales (resto cae al prior poblacional), smooth M=%.1f",
                    n_geo, len(merged), m)
        merged = merged.drop(columns=["obs_geo_count"])

    # distribuir cantidades proporcionales a peso
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col] * merged["weight"]

    # tasas per-cápita y flags: constantes dentro del distrito
    result = merged.rename(columns={"categoria": "crime_cat"})
    result = result[["h3_index", "ubigeo", "year", "crime_cat"]
                    + [c for c in numeric_cols if c in merged.columns]
                    + [c for c in rate_cols if c in merged.columns]
                    + [c for c in flag_cols if c in merged.columns]]

    # tipos
    for col in numeric_cols + rate_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
    if "inestable" in result.columns:
        result["inestable"] = result["inestable"].fillna(0).astype("int8")

    LOGGER.info("Output: %d filas, %d celdas H3, %d categorías, años=%s",
                len(result), result.h3_index.nunique(),
                result.crime_cat.nunique(), sorted(result.year.unique()))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_file, index=False)
    LOGGER.info("Guardado → %s", out_file)

    # --- resumen ---
    print("\nCobertura por categoría (celdas H3 con datos, año 2022):")
    y2022 = result[result.year == 2022]
    for cat, g in y2022.groupby("crime_cat"):
        print(f"  {cat:<30} {len(g):>5} celdas  latente_total={g['latente'].sum():>9,.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
