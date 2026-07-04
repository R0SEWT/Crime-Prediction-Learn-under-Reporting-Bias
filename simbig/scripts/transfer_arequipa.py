#!/usr/bin/env python3
"""
Test de transferencia espacial Lima -> Arequipa (slr-211).

Entrena el bloque tabular TRANSFERIBLE (población WorldPop + manzana censal
INEI + OSM sin features Lima-específicas) sobre el panel de Lima (2019-2022,
target = oráculo geocodificado) y evalúa en Arequipa 2023 contra su propio
oráculo geocodificado (85.5% de denuncias con coordenada real en 2019-2023).

Baselines en Arequipa: persistencia (conteos 2022), prior poblacional, y
modelo LOCAL (mismas features entrenadas en Arequipa 2019-2022).

Etapas cacheadas en data/silver/transfer_arequipa/:
    python3 scripts/transfer_arequipa.py --stage oracle     # corpus nacional -> oráculo AQP
    python3 scripts/transfer_arequipa.py --stage features   # WorldPop + manzana + OSM -> features AQP
    python3 scripts/transfer_arequipa.py --stage eval       # entrena Lima, evalúa AQP
    python3 scripts/transfer_arequipa.py                    # all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

NATIONAL_CORPUS = Path.home() / "Code/ElComercio/MapaDelictivo/data/mininter_api/data_denuncias"
WORLDPOP_TIF = ROOT / "data/raw/worldpop/per_ppp_2020_UNadj_constrained.tif"

# Ciudades evaluables (>=80% coords reales 2019-2023, ver etapa2_transfer_feasibility.md)
# bbox WGS84 (west, south, east, north) cubriendo el área metropolitana
CITIES = {
    "arequipa": {"prov": "0401", "bbox": (-71.70, -16.55, -71.40, -16.25),
                 "label": "Arequipa"},
    "piura":    {"prov": "2001", "bbox": (-80.75, -5.35, -80.50, -5.05),
                 "label": "Piura"},
    "cusco":    {"prov": "0801", "bbox": (-72.10, -13.65, -71.80, -13.45),
                 "label": "Cusco"},
}

# config activa (se fija en main según --city; defaults = Arequipa)
CITY = "arequipa"
AQP_WEST, AQP_SOUTH, AQP_EAST, AQP_NORTH = CITIES["arequipa"]["bbox"]
AQP_BBOX = CITIES["arequipa"]["bbox"]
AQP_PROV = CITIES["arequipa"]["prov"]
OUT_DIR = ROOT / "data/silver/transfer_arequipa"
ORACLE_AQP = OUT_DIR / "aqp_oracle.parquet"
FEATURES_AQP = OUT_DIR / "aqp_features.parquet"
REPORT_MD = ROOT / "analysis/etapa2_transfer_arequipa.md"
H3_RES = 8


def set_city(city: str) -> None:
    """Fija la configuración global de ciudad (paths, bbox, provincia)."""
    global CITY, AQP_WEST, AQP_SOUTH, AQP_EAST, AQP_NORTH, AQP_BBOX, AQP_PROV
    global OUT_DIR, ORACLE_AQP, FEATURES_AQP, REPORT_MD
    cfg = CITIES[city]
    CITY = city
    AQP_WEST, AQP_SOUTH, AQP_EAST, AQP_NORTH = cfg["bbox"]
    AQP_BBOX = cfg["bbox"]
    AQP_PROV = cfg["prov"]
    prefix = "aqp" if city == "arequipa" else city   # compat con artefactos AQP existentes
    OUT_DIR = ROOT / f"data/silver/transfer_{city}"
    ORACLE_AQP = OUT_DIR / f"{prefix}_oracle.parquet"
    FEATURES_AQP = OUT_DIR / f"{prefix}_features.parquet"
    REPORT_MD = ROOT / f"analysis/etapa2_transfer_{city}.md"

TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023
MIN_CELLS = 5

# Bloque transferible: existe y se computa IGUAL en ambas ciudades.
# Excluidos: nbi_pct/poverty_decile/avg_household_size (REDATAM solo-Lima),
# dist_metro_km/dist_brt_km (infraestructura solo-Lima), Meta/VIIRS/S2/SV.
FEAT_POP = ["population", "pop_density_km2"]
FEAT_MZN = ["mzn_pop_total", "mzn_nbi_pct", "mzn_nbi_pop", "mzn_vivienda",
            "mzn_pop_hombres_pct", "mzn_n_manzanas"]
FEAT_OSM = ["road_density_km_km2", "intersection_count",
            "poi_count_retail", "poi_count_food", "poi_count_finance",
            "poi_count_transport", "poi_count_healthcare", "poi_count_education",
            "poi_count_nightlife", "dist_police_km"]
TRANSFER_FEATS = FEAT_POP + FEAT_MZN + FEAT_OSM

LOGGER = logging.getLogger("transfer_arequipa")


def aqp_cells() -> list[str]:
    import h3
    geojson = {"type": "Polygon", "coordinates": [[
        [AQP_WEST, AQP_SOUTH], [AQP_EAST, AQP_SOUTH], [AQP_EAST, AQP_NORTH],
        [AQP_WEST, AQP_NORTH], [AQP_WEST, AQP_SOUTH]]]}
    return sorted(h3.geo_to_cells(geojson, H3_RES))


# ── Etapa 1: oráculo geocodificado de Arequipa ───────────────────────────────

def build_oracle() -> pd.DataFrame:
    import h3
    import pyarrow.dataset as pads
    import pyarrow.compute as pc
    from build_observed_h3_points import map_category

    LOGGER.info("Escaneando corpus nacional (prov %s)…", AQP_PROV)
    d = pads.dataset(str(NATIONAL_CORPUS), format="parquet")
    filt = ((pc.field("id_prov_hecho") == AQP_PROV)
            & (pc.field("estado_coord") == "CON COORDENADA")
            & (pc.field("solo_denuncia") == 1)
            & ~pc.field("es_delito_x").isin(["2.Faltas", "3. Niños y adolescentes"]))
    cols = ["año_hecho", "id_dist_hecho", "lat_hecho", "long_hecho",
            "tipo_hecho", "subtipo_hecho", "modalidad_hecho"]
    df = d.to_table(columns=cols, filter=filt).to_pandas()
    df["year"] = pd.to_numeric(df["año_hecho"], errors="coerce")
    df = df[df["year"].between(2018, TEST_YEAR)].copy()
    df["lat"] = pd.to_numeric(df["lat_hecho"], errors="coerce")
    df["lng"] = pd.to_numeric(df["long_hecho"], errors="coerce")
    df = df[df["lat"].between(AQP_SOUTH, AQP_NORTH) & df["lng"].between(AQP_WEST, AQP_EAST)]
    LOGGER.info("Puntos geocodificados en bbox 2018-%d: %d", TEST_YEAR, len(df))

    df["crime_cat"] = map_category(df["tipo_hecho"], df["subtipo_hecho"], df["modalidad_hecho"])
    df = df[df["crime_cat"].notna()].copy()
    df["h3_index"] = [h3.latlng_to_cell(la, ln, H3_RES) for la, ln in zip(df["lat"], df["lng"])]
    df["ubigeo"] = df["id_dist_hecho"].astype(str).str.zfill(6)
    df["year"] = df["year"].astype(int)

    agg = (df.groupby(["h3_index", "ubigeo", "year", "crime_cat"])
           .size().rename("obs_geo_count").reset_index())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(ORACLE_AQP, index=False)
    LOGGER.info("Oráculo AQP: %d filas, %d celdas, %d distritos → %s",
                len(agg), agg.h3_index.nunique(), agg.ubigeo.nunique(), ORACLE_AQP)
    print(agg.groupby(["year", "crime_cat"]).obs_geo_count.sum().unstack(fill_value=0).to_string())
    return agg


# ── Etapa 2: features transferibles de Arequipa ──────────────────────────────

def build_features() -> pd.DataFrame:
    import h3
    cells = aqp_cells()
    LOGGER.info("Grid AQP: %d celdas H3 res-%d", len(cells), H3_RES)
    feats = pd.DataFrame({"h3_index": cells})

    # población WorldPop (mismo raster/método que Lima: suma de pixeles por celda)
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(WORLDPOP_TIF) as src:
        win = from_bounds(AQP_WEST, AQP_SOUTH, AQP_EAST, AQP_NORTH, src.transform)
        arr = src.read(1, window=win)
        tr = src.window_transform(win)
        nodata = src.nodata
    rows, cols_idx = np.where(arr > 0 if nodata is None else (arr != nodata) & (arr > 0))
    xs, ys = rasterio.transform.xy(tr, rows, cols_idx)
    pix = pd.DataFrame({"pop": arr[rows, cols_idx],
                        "h3_index": [h3.latlng_to_cell(y, x, H3_RES) for x, y in zip(xs, ys)]})
    pop = pix.groupby("h3_index")["pop"].sum().rename("population").reset_index()
    area = h3.average_hexagon_area(H3_RES, unit="km^2")
    pop["pop_density_km2"] = pop["population"] / area
    feats = feats.merge(pop, on="h3_index", how="left")
    LOGGER.info("WorldPop: %d celdas con población, total %.0f",
                (feats.population > 0).sum(), feats.population.sum())

    # manzana censal INEI (capa nacional COFOPRI, mismo pipeline que Lima)
    import fetch_inei_manzana_h3 as mzn_mod
    mzn_mod.LIMA_UBIGEO_PREFIX = AQP_PROV
    # en el MapServer actual la manzana censal es la capa 3 (la 0 es CENTRO
    # POBLADO) y no expone campos X,Y — el centroide sale de la geometría
    mzn_mod.MANZANA_LAYER_ID = 3
    mzn_mod.FIELDS = "OBJECTID,pobtotal,pobnbi,poph,pobm,vivienda,ubigeo,tipo"
    raw_path = OUT_DIR / f"{CITY}_manzana_raw.parquet" if CITY != "arequipa" else OUT_DIR / "aqp_manzana_raw.parquet"
    if raw_path.exists():
        raw = mzn_mod.load_raw(raw_path)
    else:
        raw = mzn_mod.download_raw(OUT_DIR)
        # download_raw escribe con nombre Lima; renombrar al cache AQP
        default = OUT_DIR / "inei_manzana_lima_raw.parquet"
        if default.exists():
            default.rename(raw_path)
    agg = mzn_mod.aggregate_to_h3(raw, set(cells))
    feats = feats.merge(agg, on="h3_index", how="left")
    LOGGER.info("Manzana: %d celdas con datos", feats.mzn_n_manzanas.notna().sum())

    # OSM (mismo gpkg Geofabrik Perú, mismas funciones con bbox parcheado)
    import fetch_osm_h3 as osm_mod
    osm_mod.BBOX = AQP_BBOX
    road_df, inter_df = osm_mod.fetch_road_features(H3_RES)
    poi_df = osm_mod.fetch_poi_features(H3_RES)
    feats = feats.merge(road_df, on="h3_index", how="left")
    feats = feats.merge(inter_df, on="h3_index", how="left")
    feats = feats.merge(poi_df, on="h3_index", how="left")
    feats["road_density_km_km2"] = feats["road_density_km_km2"].fillna(0.0)
    feats["intersection_count"] = feats["intersection_count"].fillna(0)
    for c in [c for c in feats.columns if c.startswith("poi_count_")]:
        feats[c] = feats[c].fillna(0)

    # dist_police_km desde el mismo gpkg (fclass=police, haversine al centroide)
    import fetch_osm_proximity_h3 as prox_mod
    prox_mod.BBOX = AQP_BBOX
    path = prox_mod.resolve_gpkg()
    existing = prox_mod.available_layers(path)
    police_latlon, _, _ = prox_mod.collect_targets(
        path, existing, "police", prox_mod.POLICE_LAYERS, prox_mod.POLICE_FCLASSES)
    centers = np.array([h3.cell_to_latlng(c) for c in feats["h3_index"]])
    feats["dist_police_km"] = prox_mod.min_dist_km(centers, police_latlon)

    feats.to_parquet(FEATURES_AQP, index=False)
    LOGGER.info("Features AQP: %s → %s", feats.shape, FEATURES_AQP)
    return feats


# ── Etapa 3: entrenar en Lima, evaluar en Arequipa ───────────────────────────

def intra_stats(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (cat, ub), g in df.groupby(["crime_cat", "ubigeo"]):
        if len(g) < MIN_CELLS or g["target"].nunique() < 2 or g[pred_col].nunique() < 2:
            continue
        rho, _ = spearmanr(g[pred_col], g["target"])
        rows.append({"crime_cat": cat, "ubigeo": ub, "rho": rho})
    return pd.DataFrame(rows)


def macro(st: pd.DataFrame) -> float:
    return float(st.groupby("crime_cat").rho.mean().mean()) if len(st) else float("nan")


def run_eval() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from eval_modality_ladder import CATS, load_panel, available

    oracle = pd.read_parquet(ORACLE_AQP)
    feats = pd.read_parquet(FEATURES_AQP)

    # panel AQP largo: grid × años × categorías, target = conteo geocodificado
    cells = feats["h3_index"].tolist()
    # ubigeo dominante por celda (para agrupación intra-distrital)
    cell_ub = (oracle.groupby(["h3_index", "ubigeo"]).obs_geo_count.sum().reset_index()
               .sort_values("obs_geo_count").drop_duplicates("h3_index", keep="last")
               [["h3_index", "ubigeo"]])
    years = TRAIN_YEARS + [TEST_YEAR]
    base = (pd.MultiIndex.from_product([cells, years, CATS],
                                       names=["h3_index", "year", "crime_cat"])
            .to_frame(index=False))
    base = base.merge(cell_ub, on="h3_index", how="left")
    base = base.merge(oracle[["h3_index", "year", "crime_cat", "obs_geo_count"]],
                      on=["h3_index", "year", "crime_cat"], how="left")
    base["target"] = base["obs_geo_count"].fillna(0.0)
    lag = oracle.copy(); lag["year"] = lag["year"] + 1
    base = base.merge(lag[["h3_index", "year", "crime_cat", "obs_geo_count"]]
                      .rename(columns={"obs_geo_count": "persist"}),
                      on=["h3_index", "year", "crime_cat"], how="left")
    base["persist"] = base["persist"].fillna(0.0)
    aqp = base.merge(feats, on="h3_index", how="left", suffixes=("", "_mzn"))
    # distrito por celda: oráculo (dominante) con fallback al ubigeo de manzana
    if "ubigeo_mzn" in aqp.columns:
        aqp["ubigeo"] = aqp["ubigeo"].fillna(aqp["ubigeo_mzn"].astype(str).str.zfill(6))
        aqp = aqp.drop(columns=["ubigeo_mzn"])
    # solo celdas con distrito conocido o población
    aqp = aqp[aqp["ubigeo"].notna() | (aqp["population"].fillna(0) > 0)].copy()
    aqp["ubigeo"] = aqp["ubigeo"].fillna("0401XX")

    # panel Lima (mismas features transferibles)
    panel_lima, _ = load_panel()
    feats_avail = [f for f in TRANSFER_FEATS if f in panel_lima.columns and f in aqp.columns]
    LOGGER.info("Features transferibles disponibles: %d/%d → %s",
                len(feats_avail), len(TRANSFER_FEATS), feats_avail)

    test = aqp[aqp["year"] == TEST_YEAR].copy()
    results = {}
    for cat in CATS:
        tr_lima = panel_lima[(panel_lima["crime_cat"] == cat)
                             & panel_lima["year"].isin(TRAIN_YEARS)]
        tr_aqp = aqp[(aqp["crime_cat"] == cat) & aqp["year"].isin(TRAIN_YEARS)]
        te = test[test["crime_cat"] == cat].copy()

        def fit_hgb(tr):
            med = tr[feats_avail].median(numeric_only=True)
            m = HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=15,
                                              min_samples_leaf=20, learning_rate=0.06,
                                              l2_regularization=0.05, random_state=42)
            m.fit(tr[feats_avail].fillna(med).to_numpy(),
                  np.log1p(tr["target"].to_numpy(float)))
            return m, med

        m_t, med_t = fit_hgb(tr_lima)
        te["pred_transfer"] = np.expm1(np.maximum(
            m_t.predict(te[feats_avail].fillna(med_t).to_numpy()), 0.0))
        m_l, med_l = fit_hgb(tr_aqp)
        te["pred_local"] = np.expm1(np.maximum(
            m_l.predict(te[feats_avail].fillna(med_l).to_numpy()), 0.0))
        te["pred_persist"] = te["persist"]
        te["pred_pop"] = te["population"].fillna(0.0)
        results[cat] = te

    allte = pd.concat(results.values(), ignore_index=True)
    models = ["pred_transfer", "pred_local", "pred_persist", "pred_pop"]
    stats = {m: intra_stats(allte, m) for m in models}

    # bootstrap pareado sobre distritos (B=2000, mismo método que
    # eval_metric_robustness.py) para los deltas que sostienen el claim
    rng = np.random.default_rng(42)
    B = 2000

    def boot_delta(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, float]:
        m = a.merge(b, on=["crime_cat", "ubigeo"], suffixes=("_a", "_b"))
        by_cat = {c: g[["rho_a", "rho_b"]].dropna().to_numpy()
                  for c, g in m.groupby("crime_cat")}
        draws = np.empty(B)
        for i in range(B):
            deltas = []
            for v in by_cat.values():
                if not len(v):
                    continue
                idx = rng.integers(0, len(v), size=len(v))
                deltas.append(v[idx, 1].mean() - v[idx, 0].mean())
            draws[i] = np.mean(deltas)
        point = float(np.mean([v[:, 1].mean() - v[:, 0].mean()
                               for v in by_cat.values() if len(v)]))
        return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    lines = [
        f"# Transferencia espacial Lima → {CITIES[CITY]['label']} (slr-211/slr-vkj)",
        "",
        f"**Protocolo:** modelo HGB (params del ladder) con el bloque tabular "
        f"TRANSFERIBLE ({len(feats_avail)} features: WorldPop, manzana censal INEI, "
        f"OSM sin features Lima-específicas). Train Lima 2019-2022 → test "
        f"**{CITIES[CITY]['label']} {TEST_YEAR}** vs oráculo geocodificado propio. Spearman "
        f"intra-distrital macro (distritos con ≥{MIN_CELLS} celdas).",
        "",
        "| categoría | transfer (Lima→AQP) | local (AQP→AQP) | persistencia | prior pob. |",
        "|:--|--:|--:|--:|--:|",
    ]
    from eval_modality_ladder import CATS as _CATS
    for cat in _CATS:
        vals = []
        for m in models:
            g = stats[m][stats[m]["crime_cat"] == cat]
            vals.append(f"{g.rho.mean():.3f}" if len(g) else "nan")
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
    lines.append("| **MACRO** | " + " | ".join(f"**{macro(stats[m]):.3f}**" for m in models) + " |")
    n_d = stats["pred_transfer"].ubigeo.nunique() if len(stats["pred_transfer"]) else 0
    lines += ["", f"Distritos evaluables: {n_d}. Celdas test: {test.h3_index.nunique()}.",
              "", "## Deltas pareados (bootstrap sobre distritos, B=2000)", "",
              "| comparación | Δ macro ρ [CI 95%] | ¿CI excluye 0? |", "|:--|:--|:--|"]
    for la, lb in [("pred_pop", "pred_transfer"), ("pred_persist", "pred_transfer"),
                   ("pred_transfer", "pred_local")]:
        d, dl, dh = boot_delta(stats[la], stats[lb])
        sig = "SÍ" if (dl > 0 or dh < 0) else "no"
        names = {"pred_transfer": "transfer", "pred_local": "local",
                 "pred_persist": "persistencia", "pred_pop": "prior pob."}
        lines.append(f"| {names[lb]} − {names[la]} | {d:+.4f} [{dl:+.4f}, {dh:+.4f}] | {sig} |")
    lines += ["", "Referencias Lima (mismo protocolo): bloque completo M4c 0.464; "
              "prior histórico 0.394; per cápita 0.349."]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGuardado → {REPORT_MD}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["oracle", "features", "eval", "all"], default="all")
    ap.add_argument("--city", choices=sorted(CITIES), default="arequipa")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s: %(message)s")
    set_city(args.city)
    LOGGER.info("Ciudad: %s (prov %s, bbox %s)", CITIES[CITY]["label"], AQP_PROV, AQP_BBOX)
    global h3
    import h3  # noqa: F811

    if args.stage in ("oracle", "all") and not (args.stage == "all" and ORACLE_AQP.exists()):
        build_oracle()
    if args.stage in ("features", "all") and not (args.stage == "all" and FEATURES_AQP.exists()):
        build_features()
    if args.stage in ("eval", "all"):
        run_eval()
    return 0


if __name__ == "__main__":
    sys.exit(main())
