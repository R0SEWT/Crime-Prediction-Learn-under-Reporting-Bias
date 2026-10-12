#!/usr/bin/env python3
"""
Oráculo de evaluación geocodificado H3 (Etapa 2, F0a / slr-xrc).

Construye un target OBSERVADO a celda H3 res-8 desde los PUNTOS reales de denuncia
(no la redistribución poblacional). Esto da la única señal intra-distrital REAL contra
la cual evaluar si el STGNN/priors recuperan la concentración verdadera del crimen.

POR QUÉ EXISTE: `crime_latent_surface.parquet` construye observado y latente a H3 con el
mismo peso poblacional (pop_h3/pop_distrito) → dentro de un distrito el target solo varía
por población. Un prior poblacional ES ese target. Este oráculo rompe esa circularidad
usando las coordenadas reales del hecho. Ver memoria `etapa2-target-h3-es-circular`.

FUENTE (externa, read-only): `INFELIX_LIMA_DENUNCIAS` o fallback local documentado
  3.1M denuncias Lima con lat_hecho/long_hecho/estado_coord/id_dist_hecho/año_hecho.

DECISIONES DE LIMPIEZA:
  - solo_denuncia == 1            (excluye intervenciones policiales / ruteo)
  - estado_coord == CON COORDENADA (descarta imputadas marcadas SIN COORDENADA)
  - ANTI-CENTROIDE: descartar pile-ups en coords exactas (estas son denuncias geocodificadas
    a la COMISARÍA, no al hecho: ej (-12.04637,-77.04275)=CPNP Alfonso Ugarte con 69,561
    denuncias en el milímetro exacto). A 5 decimales (~1 m) ninguna dirección real comparte
    coordenada con decenas → grupos exactos > --max-per-coord se eliminan. ~43% de los
    "CON COORDENADA" son este artefacto.
  - distrito vía id_dist_hecho     (6 díg confiable; NO ubigeo_hecho_delito, corrupto ~37%)
  - coords dentro del bbox de Lima Metropolitana
  - solo celdas presentes en el feature matrix (4,172)

CROSSWALK LIMA.parquet → 5 categorías de la tesis (tipo_hecho / subtipo_hecho / modalidad):
  robo_hurto_callejero      PATRIMONIO + subtipo ∈ {ROBO, HURTO}
  estafa                    PATRIMONIO + ESTAFA Y OTRAS DEFRAUDACIONES
  extorsion                 PATRIMONIO + EXTORSION
  secuestro                 modalidad contiene SECUESTRO
  violencia_familiar_sexual LEY VIOLENCIA CONTRA LA MUJER ∪ (LIBERTAD + VIOL. LIBERTAD SEXUAL)

SESGO DE SELECCIÓN (limitación declarada): qué denuncias reciben coordenada real NO es
aleatorio. Se reporta `real_coord_ratio` por distrito como diagnóstico — celdas en distritos
con baja cobertura geocodificada tienen un oráculo más ruidoso.

EVAL-ONLY: este artefacto JAMÁS entra al entrenamiento.

Output: data/silver/h3_features/h3_observed_geocoded.parquet
        (--monthly → h3_observed_geocoded_monthly.parquet, con columna month)
Schema: h3_index, ubigeo, year[, month], crime_cat, obs_geo_count, real_coord_ratio

Uso:
    python3 scripts/build_observed_h3_points.py
    python3 scripts/build_observed_h3_points.py --dry-run
    python3 scripts/build_observed_h3_points.py --monthly
    python3 scripts/build_observed_h3_points.py --source /ruta/a/LIMA.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research_paths import LIMA_DENUNCIAS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = LIMA_DENUNCIAS
MATRIX_FILE = ROOT / "data/silver/h3_feature_matrix.parquet"
OUT_FILE = ROOT / "data/silver/h3_features/h3_observed_geocoded.parquet"

H3_RES = 8
YEAR_MIN, YEAR_MAX = 2018, 2024  # 2018 para lag histórico, 2024 para holdout temporal
# bbox Lima Metropolitana (de las celdas del matrix, con margen)
LAT_MIN, LAT_MAX = -12.45, -11.55
LNG_MIN, LNG_MAX = -77.30, -76.60

LOGGER = logging.getLogger("build_observed_h3_points")

SOURCE_COLS = [
    "solo_denuncia", "estado_coord", "id_dist_hecho",
    "lat_hecho", "long_hecho", "año_hecho", "mes_hecho",
    "tipo_hecho", "subtipo_hecho", "modalidad_hecho",
]


def map_category(tipo: pd.Series, subtipo: pd.Series, modalidad: pd.Series) -> pd.Series:
    """Mapea (tipo_hecho, subtipo_hecho, modalidad_hecho) a una de las 5 categorías o NaN."""
    t = tipo.fillna("").str.upper()
    s = subtipo.fillna("").str.upper()
    m = modalidad.fillna("").str.upper()

    cat = pd.Series(np.nan, index=tipo.index, dtype=object)

    patrimonio = t == "PATRIMONIO (DELITO)"
    cat[patrimonio & s.isin(["ROBO", "HURTO"])] = "robo_hurto_callejero"
    cat[patrimonio & (s == "ESTAFA Y OTRAS DEFRAUDACIONES")] = "estafa"
    cat[patrimonio & (s == "EXTORSION")] = "extorsion"

    # violencia familiar/sexual: ley de violencia contra la mujer ∪ violación libertad sexual
    ley_mujer = t == "LEY DE VIOLENCIA CONTRA LA MUJER Y GRUPOS VULNERABLES"
    viol_sexual = (t == "LIBERTAD (DELITO)") & (s == "VIOLACION DE LA LIBERTAD SEXUAL")
    cat[ley_mujer | viol_sexual] = "violencia_familiar_sexual"

    # secuestro: modalidad contiene SECUESTRO (prioridad alta, sobre-escribe lo anterior)
    cat[m.str.contains("SECUESTRO", na=False)] = "secuestro"

    return cat


def load_clean_points(source: Path, max_per_coord: int = 30,
                      with_month: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Puntos de denuncia geocodificados limpios (point-level) + real_coord_ratio por distrito.

    Limpieza idéntica al oráculo: solo_denuncia, CON COORDENADA, bbox Lima, anti-centroide
    (pile-ups en coord exacta = geocodificación a comisaría), distrito vía id_dist_hecho,
    restringido a celdas del feature matrix. with_month añade columna `month` (1..12).
    """
    import h3
    if not source.exists():
        raise FileNotFoundError(f"Fuente no encontrada: {source}")
    if not MATRIX_FILE.exists():
        raise FileNotFoundError(f"Feature matrix no encontrado: {MATRIX_FILE}")

    df = pd.read_parquet(source, columns=SOURCE_COLS)
    df = df[df["solo_denuncia"] == 1].copy()
    df["year"] = pd.to_numeric(df["año_hecho"], errors="coerce")
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df["year"] = df["year"].astype(int)
    df["ubigeo"] = df["id_dist_hecho"].astype(str).str.zfill(6)
    if with_month:
        df["month"] = pd.to_numeric(df["mes_hecho"], errors="coerce").astype("Int64")
    df["crime_cat"] = map_category(df["tipo_hecho"], df["subtipo_hecho"], df["modalidad_hecho"])

    df_cat = df[df["crime_cat"].notna()].copy()
    con = df_cat["estado_coord"].astype(str).str.upper() == "CON COORDENADA"
    cov = (df_cat.assign(_con=con.astype(float))
                 .groupby("ubigeo")["_con"].mean().rename("real_coord_ratio"))

    geo = df_cat[con].copy()
    geo["lat"] = pd.to_numeric(geo["lat_hecho"], errors="coerce")
    geo["lng"] = pd.to_numeric(geo["long_hecho"], errors="coerce")
    geo = geo[geo["lat"].between(LAT_MIN, LAT_MAX) & geo["lng"].between(LNG_MIN, LNG_MAX)]

    pile = geo.groupby(["lat", "lng"]).size()
    bad_coords = set(pile[pile > max_per_coord].index)
    if bad_coords:
        key = list(zip(geo["lat"], geo["lng"]))
        keep = np.array([k not in bad_coords for k in key])
        geo = geo[keep]

    geo["h3_index"] = [h3.latlng_to_cell(lat, lng, H3_RES)
                       for lat, lng in zip(geo["lat"].to_numpy(), geo["lng"].to_numpy())]
    matrix_cells = set(pd.read_parquet(MATRIX_FILE, columns=["h3_index"])["h3_index"].unique())
    geo = geo[geo["h3_index"].isin(matrix_cells)]
    if with_month:
        geo = geo[geo["month"].between(1, 12)]
        cols = ["h3_index", "ubigeo", "year", "month", "crime_cat"]
    else:
        cols = ["h3_index", "ubigeo", "year", "crime_cat"]
    return geo[cols].reset_index(drop=True), cov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--max-per-coord", type=int, default=30,
                    help="Descartar coords exactas con más de N denuncias (anti-centroide comisaría)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--monthly", action="store_true",
                    help="Cadencia mensual: añade month, salida *_monthly.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        geo, cov = load_clean_points(args.source, args.max_per_coord, with_month=args.monthly)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("puntos limpios para el oráculo: %d", len(geo))

    if args.dry_run:
        LOGGER.info("DRY-RUN: no se escribe.")
        print(geo.groupby("crime_cat").size().to_string())
        return 0

    group_keys = ["h3_index", "ubigeo", "year", "crime_cat"]
    out_file = OUT_FILE
    if args.monthly:
        group_keys = ["h3_index", "ubigeo", "year", "month", "crime_cat"]
        out_file = OUT_FILE.with_name("h3_observed_geocoded_monthly.parquet")

    agg = geo.groupby(group_keys).size().rename("obs_geo_count").reset_index()
    agg = agg.merge(cov, on="ubigeo", how="left")
    agg["obs_geo_count"] = agg["obs_geo_count"].astype("int32")
    agg["real_coord_ratio"] = agg["real_coord_ratio"].astype("float32")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_file, index=False)
    LOGGER.info("Guardado → %s (%d filas)", out_file, len(agg))

    # ── resumen + verificación de aceptación ──
    print("\n=== Cobertura por categoría (celdas H3 con ≥1 punto) ===")
    for cat, g in agg.groupby("crime_cat"):
        print(f"  {cat:<28} {g['h3_index'].nunique():>5} celdas  total={g['obs_geo_count'].sum():>8,}")

    print(f"\nreal_coord_ratio mediana (Lima): {cov.median():.3f}  "
          f"(criterio aceptación: > 0.25 → {'OK' if cov.median() > 0.25 else 'FALLA'})")

    # test inverso: dentro de un distrito, las celdas del oráculo varían en algo
    # distinto a población (a diferencia del target circular)
    y2022 = agg[(agg.year == 2022) & (agg.crime_cat == "robo_hurto_callejero")]
    big = y2022.groupby("ubigeo").filter(lambda x: len(x) >= 5)
    if len(big):
        ub = big["ubigeo"].value_counts().index[0]
        sub = big[big.ubigeo == ub]
        cv = sub["obs_geo_count"].std() / max(sub["obs_geo_count"].mean(), 1e-9)
        print(f"Test inverso: distrito {ub}, CV intra-distrital del conteo = {cv:.2f} "
              f"(>0 → hay señal real intra-distrital, no solo población)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
