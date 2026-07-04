#!/usr/bin/env python3
"""
Q4 Etapa 2: ladder de modalidades M0-M6.

Mide que bloque de senal aporta ganancia marginal al predecir la concentracion
real geocodificada del crimen a H3. El foco no es "modelo final alto", sino
delta por modalidad contra el oraculo.

Split temporal:
  train = 2019-2022
  test  = 2023

Metricas principales:
  macro intra-district Spearman rho, Recall@10, NDCG, PAI.
"""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from etapa2_metrics import evaluate, macro_average
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from etapa2_metrics import evaluate, macro_average

ROOT = Path(__file__).resolve().parents[1]
MATRIX_FILE = ROOT / "data/silver/h3_feature_matrix.parquet"
ORACLE_FILE = ROOT / "data/silver/h3_features/h3_observed_geocoded.parquet"
PRIOR_FILE = ROOT / "data/silver/predictions/prior_ladder.parquet"
STGNN_FILE = ROOT / "data/silver/predictions/real_target_stgnn.parquet"
EDGES_FILE = ROOT / "data/silver/h3_graph_edges.parquet"
OUT_REPORT = ROOT / "analysis/etapa2_q4_modalidad_ladder.md"
OUT_METRICS = ROOT / "data/silver/predictions/q4_modality_ladder_metrics.csv"

CATS = [
    "robo_hurto_callejero",
    "extorsion",
    "estafa",
    "violencia_familiar_sexual",
    "secuestro",
]

DEMOGRAPHIC = [
    "population",
    "pop_density_km2",
    "nbi_pct",
    "poverty_decile",
    "avg_household_size",
]
MANZANA = [
    "mzn_pop_total",
    "mzn_nbi_pct",
    "mzn_nbi_pop",
    "mzn_vivienda",
    "mzn_pop_hombres_pct",
    "mzn_n_manzanas",
]
OSM = [
    "road_density_km_km2",
    "intersection_count",
    "poi_count_retail",
    "poi_count_food",
    "poi_count_finance",
    "poi_count_transport",
    "poi_count_healthcare",
    "poi_count_education",
    "poi_count_nightlife",
    "dist_police_km",
    "dist_metro_km",
    "dist_brt_km",
]
OSM_CENTRALITY = [
    "road_node_count",
    "road_edge_count",
    "road_dead_end_count",
    "road_dead_end_ratio",
    "road_articulation_count",
    "road_articulation_ratio",
    "road_node_degree_mean",
    "road_node_degree_max",
    "road_weighted_degree_km_mean",
    "road_pagerank_sum",
    "road_pagerank_mean",
    "road_pagerank_max",
    "road_betweenness_approx_mean",
    "road_betweenness_approx_max",
    "road_major_edge_count",
    "road_major_edge_share",
    "commercial_poi_count",
    "commercial_corridor_index",
]
COFOPRI = [
    "pcdpi_lotes_count",
    "pcdpi_niveles_median",
    "pcdpi_fai",
    "pcdpi_pct_agua",
    "pcdpi_pct_luz",
    "pcdpi_pct_desague",
    "pcdpi_pct_comercial",
    "pcdpi_pct_residencial",
    "pcdpi_unidades_sum",
    "cofopri_formalizacion_lotes_count",
]
LICENCIAS = [
    "lic_comercial_total",
    "lic_restaurantes",
    "lic_bares",
    "lic_bancos_financieras",
    "lic_mercados",
    "lic_nocturno",
    "lic_talleres",
    "lic_farmacias",
]
MERCADOS = [
    "mercado_count",
    "mercado_puestos_fijos_sum",
    "mercado_puestos_func_sum",
    "mercado_socios_sum",
    "mercado_puestos_total_sum",
    "mercado_abarrotes_puestos_sum",
    "mercado_comidas_puestos_sum",
    "mercado_mayorista_count",
    "mercado_mixto_count",
    "mercado_licencia_count",
    "dist_mercado_km",
]
EDUCACION = [
    "edu_service_count",
    "edu_local_count",
    "edu_inicial_count",
    "edu_primaria_count",
    "edu_secundaria_count",
    "edu_basica_alt_count",
    "edu_especial_count",
    "edu_superior_count",
    "edu_cetpro_count",
    "edu_publica_count",
    "edu_privada_count",
    "edu_escolarizada_count",
    "edu_urbana_count",
    "dist_edu_km",
    "dist_edu_basica_km",
    "dist_edu_superior_km",
]
REMOTE = [
    "elevation_m",
    "slope_deg",
    "elevation_std",
    "lc_builtup",
    "lc_bare",
    "lc_water",
    "lc_trees",
    "lc_grass",
    "s2_ndvi_median",
    "s2_ndbi_median",
    "s2_valid_pixel_ratio",
    "dnb_mean",
    "dnb_std",
]
META = [
    "pop_meta_2020",
    "pop_meta_density_km2",
]
SALUD = [
    "salud_count",
    "salud_hospital_count",
    "salud_sin_intern_count",
    "salud_nivel1_count",
    "salud_nivel2_count",
    "salud_nivel3_count",
    "salud_publico_count",
    "salud_privado_count",
    "dist_salud_km",
    "dist_hospital_km",
]
SEGURIDAD = [
    "seg_comisaria_count",
    "seg_camara_count",
    "seg_serenazgo_count",
    "seg_total_count",
    "dist_comisaria_km",
    "dist_camara_km",
]
TRANSPORTE = [
    "trans_paradero_count",
    "trans_estacion_count",
    "trans_metro_count",
    "trans_total_count",
    "dist_paradero_km",
    "dist_estacion_km",
]

_FULL_URBAN = DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI + LICENCIAS + MERCADOS + EDUCACION + REMOTE + META

# Historial geocodificado como FEATURE (no solo baseline M5): t-1 crudo,
# lag espacial k-ring1 (hotspot suavizado) y share intra-distrital (la
# concentración relativa es exactamente la métrica intra). Se construyen en
# load_panel desde el oráculo, no viven en el feature matrix.
HISTORIAL = ["persist", "persist_ring1", "persist_share"]

FEATURE_SETS = {
    "M1_demografia": DEMOGRAPHIC,
    "M1b_manzana": DEMOGRAPHIC + MANZANA,
    "M2_osm": DEMOGRAPHIC + MANZANA + OSM,
    "M2a_osm_centrality": DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY,
    "M2b_catastro": DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI,
    "M2c_licencias": DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI + LICENCIAS,
    "M2d_mercados": DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI + LICENCIAS + MERCADOS,
    "M2e_educacion": DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI + LICENCIAS + MERCADOS + EDUCACION,
    "M3_remote": DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI + LICENCIAS + MERCADOS + EDUCACION + REMOTE,
    "M4_meta": _FULL_URBAN,
    "M4a_salud": _FULL_URBAN + SALUD,
    "M4b_seguridad": _FULL_URBAN + SALUD + SEGURIDAD,
    "M4c_transporte": _FULL_URBAN + SALUD + SEGURIDAD + TRANSPORTE,
    "M7a_persist": _FULL_URBAN + SALUD + SEGURIDAD + TRANSPORTE + ["persist"],
    "M7b_historial_esp": _FULL_URBAN + SALUD + SEGURIDAD + TRANSPORTE + HISTORIAL,
}

KEYS = ("intra_spearman", "intra_recall10", "intra_pai", "recall10", "ndcg")
LOGGER = logging.getLogger("eval_modality_ladder")


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_panel() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    feat_cols = sorted(set(DEMOGRAPHIC + MANZANA + OSM + OSM_CENTRALITY + COFOPRI + LICENCIAS + MERCADOS + EDUCACION + REMOTE + META + SALUD + SEGURIDAD + TRANSPORTE))
    schema_cols = pq.read_schema(MATRIX_FILE).names
    feat_cols = [c for c in feat_cols if c in schema_cols]
    feat = pd.read_parquet(MATRIX_FILE, columns=["h3_index", "year", "ubigeo"] + feat_cols)
    feat = feat[feat["ubigeo"].notna() & (feat["ubigeo"].astype(str) != "")].copy()
    feat["ubigeo"] = feat["ubigeo"].astype(str).str.zfill(6)
    base = feat[["h3_index", "year", "ubigeo"] + feat_cols].copy()

    oracle = pd.read_parquet(ORACLE_FILE, columns=["h3_index", "year", "crime_cat", "obs_geo_count"])

    panels = []
    for cat in CATS:
        oc = oracle[oracle["crime_cat"] == cat]
        b = base.merge(oc[["h3_index", "year", "obs_geo_count"]], on=["h3_index", "year"], how="left")
        b["target"] = b["obs_geo_count"].fillna(0.0)
        lag = oc.copy()
        lag["year"] = lag["year"] + 1
        b = b.merge(
            lag[["h3_index", "year", "obs_geo_count"]].rename(columns={"obs_geo_count": "persist"}),
            on=["h3_index", "year"],
            how="left",
        )
        b["persist"] = b["persist"].fillna(0.0)
        # lag espacial: media del persist en el k-ring 1 (vecinos H3)
        if EDGES_FILE.exists():
            edges = pd.read_parquet(EDGES_FILE)
            nb = edges.merge(
                b[["h3_index", "year", "persist"]],
                left_on="dst_h3", right_on="h3_index", how="inner",
            )
            ring = (nb.groupby(["src_h3", "year"])["persist"].mean()
                    .rename("persist_ring1").reset_index()
                    .rename(columns={"src_h3": "h3_index"}))
            b = b.merge(ring, on=["h3_index", "year"], how="left")
            b["persist_ring1"] = b["persist_ring1"].fillna(0.0)
        else:
            b["persist_ring1"] = 0.0
        # share intra-distrital del persist (concentración relativa t-1)
        dist_sum = b.groupby(["ubigeo", "year"])["persist"].transform("sum")
        b["persist_share"] = np.where(dist_sum > 0, b["persist"] / dist_sum, 0.0)
        b["crime_cat"] = cat
        panels.append(b.drop(columns=["obs_geo_count"]))

    sample = pd.concat(panels[:1], ignore_index=True)
    feature_sets = {}
    for name, cols in FEATURE_SETS.items():
        feats = available(cols, sample)
        if name == "M2c_licencias" and not available(LICENCIAS, sample):
            LOGGER.warning("M2c_licencias omitido: no hay columnas lic_* en el feature matrix")
            continue
        if name == "M2d_mercados" and not available(MERCADOS, sample):
            LOGGER.warning("M2d_mercados omitido: no hay columnas mercado_* en el feature matrix")
            continue
        if name == "M2e_educacion" and not available(EDUCACION, sample):
            LOGGER.warning("M2e_educacion omitido: no hay columnas edu_* en el feature matrix")
            continue
        if name == "M4a_salud" and not available(SALUD, sample):
            LOGGER.warning("M4a_salud omitido: no hay columnas salud_* en el feature matrix")
            continue
        if name == "M4b_seguridad" and not available(SEGURIDAD, sample):
            LOGGER.warning("M4b_seguridad omitido: no hay columnas seg_* en el feature matrix")
            continue
        if name == "M4c_transporte" and not available(TRANSPORTE, sample):
            LOGGER.warning("M4c_transporte omitido: no hay columnas trans_* en el feature matrix")
            continue
        feature_sets[name] = feats
    return pd.concat(panels, ignore_index=True), feature_sets


def evaluate_prediction(df: pd.DataFrame, pred_col: str) -> tuple[dict[str, dict], dict]:
    per_cat = {}
    for cat in CATS:
        d = df[df["crime_cat"] == cat].copy()
        if d.empty or d["target"].sum() <= 0:
            continue
        per_cat[cat] = evaluate(d, pred_col, "target", district_col="ubigeo", boot=False)
    return per_cat, macro_average(per_cat, KEYS)


def train_models(panel: pd.DataFrame, feature_sets: dict[str, list[str]], test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_years = [y for y in [2019, 2020, 2021, 2022] if y < test_year]
    metric_rows = []

    test_base = panel[panel["year"] == test_year][["h3_index", "year", "ubigeo", "crime_cat", "target", "persist"]].copy()

    for model_name, feats in feature_sets.items():
        LOGGER.info("%s: %d features", model_name, len(feats))
        pred_col = f"pred_{model_name}"
        model_parts = []
        for cat in CATS:
            d = panel[panel["crime_cat"] == cat]
            tr = d[d["year"].isin(train_years)].copy()
            te = d[d["year"] == test_year].copy()
            if te["target"].sum() <= 0:
                continue

            med = tr[feats].median(numeric_only=True)
            Xtr = tr[feats].fillna(med).to_numpy()
            ytr = np.log1p(tr["target"].to_numpy(dtype=float))
            Xte = te[feats].fillna(med).to_numpy()

            model = HistGradientBoostingRegressor(
                max_iter=140,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                learning_rate=0.06,
                l2_regularization=0.05,
                random_state=42,
            )
            model.fit(Xtr, ytr)
            te[pred_col] = np.expm1(np.maximum(model.predict(Xte), 0.0))
            model_parts.append(te[["h3_index", "year", "ubigeo", "crime_cat", "target", "persist", pred_col]])
            del model, Xtr, Xte, ytr
            gc.collect()

        pred_df = pd.concat(model_parts, ignore_index=True)
        per_cat, macro = evaluate_prediction(pred_df, pred_col)
        for cat, res in per_cat.items():
            metric_rows.append({"model": model_name, "crime_cat": cat, **res})
        metric_rows.append({"model": model_name, "crime_cat": "__macro__", **macro})
        del pred_df, model_parts
        gc.collect()

    return test_base, pd.DataFrame(metric_rows)


def add_m0_prior(test_df: pd.DataFrame, metrics: list[dict], test_year: int) -> pd.DataFrame:
    if not PRIOR_FILE.exists():
        LOGGER.warning("No existe %s; ejecutando M0 sin prior_uniform", PRIOR_FILE)
        return test_df
    prior = pd.read_parquet(PRIOR_FILE)
    prior = prior[prior["year"] == test_year][["h3_index", "year", "crime_cat", "prior_uniform", "prior_population", "prior_osm", "prior_historical"]]
    out = test_df.merge(prior, on=["h3_index", "year", "crime_cat"], how="left")
    for col, name in [
        ("prior_uniform", "M0_prior_uniform"),
        ("prior_population", "prior_population_ref"),
        ("prior_osm", "prior_osm_ref"),
        ("prior_historical", "prior_historical_ref"),
    ]:
        per_cat, macro = evaluate_prediction(out.rename(columns={col: "pred_tmp"}), "pred_tmp")
        for cat, res in per_cat.items():
            metrics.append({"model": name, "crime_cat": cat, **res})
        metrics.append({"model": name, "crime_cat": "__macro__", **macro})
    return out


def add_history_direct(test_df: pd.DataFrame, metrics: list[dict]) -> pd.DataFrame:
    per_cat, macro = evaluate_prediction(test_df.rename(columns={"persist": "pred_tmp"}), "pred_tmp")
    for cat, res in per_cat.items():
        metrics.append({"model": "M5_historial", "crime_cat": cat, **res})
    metrics.append({"model": "M5_historial", "crime_cat": "__macro__", **macro})
    return test_df


def add_stgnn_reference(test_df: pd.DataFrame, metrics: list[dict], test_year: int) -> pd.DataFrame:
    if not STGNN_FILE.exists():
        metrics.append({"model": "M6_stgnn_ref", "crime_cat": "__macro__", "missing": True})
        return test_df
    st = pd.read_parquet(STGNN_FILE)
    if "year" not in st.columns:
        st["year"] = test_year
    if "pred" in st.columns:
        st = st[st["year"] == test_year][["h3_index", "year", "crime_cat", "pred"]].rename(columns={"pred": "pred_M6_stgnn_ref"})
    else:
        rows = []
        for cat in CATS:
            pred_col = f"pred_{cat}"
            if pred_col not in st.columns:
                continue
            d = st[["h3_index", "year", pred_col]].copy()
            d["crime_cat"] = cat
            d = d.rename(columns={pred_col: "pred_M6_stgnn_ref"})
            rows.append(d)
        st = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if st.empty:
        metrics.append({"model": "M6_stgnn_ref", "crime_cat": "__macro__", "missing": True})
        return test_df
    st = (
        st.groupby(["h3_index", "year", "crime_cat"], as_index=False)["pred_M6_stgnn_ref"]
        .mean()
    )

    out = test_df.merge(st, on=["h3_index", "year", "crime_cat"], how="left")
    per_cat, macro = evaluate_prediction(out, "pred_M6_stgnn_ref")
    for cat, res in per_cat.items():
        metrics.append({"model": "M6_stgnn_ref", "crime_cat": cat, **res})
    metrics.append({"model": "M6_stgnn_ref", "crime_cat": "__macro__", **macro})
    return out


def macro_table(metrics: pd.DataFrame) -> pd.DataFrame:
    macro = metrics[metrics["crime_cat"] == "__macro__"].copy()
    rows = []
    order = [
        "M0_prior_uniform",
        "M1_demografia",
        "M1b_manzana",
        "M2_osm",
        "M2a_osm_centrality",
        "M2b_catastro",
        "M2c_licencias",
        "M2d_mercados",
        "M2e_educacion",
        "M3_remote",
        "M4_meta",
        "M4a_salud",
        "M4b_seguridad",
        "M4c_transporte",
        "M7a_persist",
        "M7b_historial_esp",
        "M5_historial",
        "M6_stgnn_ref",
        "prior_population_ref",
        "prior_osm_ref",
        "prior_historical_ref",
    ]
    macro["model"] = pd.Categorical(macro["model"], categories=order, ordered=True)
    macro = macro.sort_values("model")
    prev = None
    marginal_ladder = {
        "M1_demografia",
        "M1b_manzana",
        "M2_osm",
        "M2a_osm_centrality",
        "M2b_catastro",
        "M2c_licencias",
        "M2d_mercados",
        "M2e_educacion",
        "M3_remote",
        "M4_meta",
    }
    for _, r in macro.iterrows():
        rho = r.get("macro_intra_spearman", np.nan)
        rec = r.get("macro_intra_recall10", np.nan)
        ndcg = r.get("macro_ndcg", np.nan)
        pai = r.get("macro_intra_pai", np.nan)
        is_ladder = str(r["model"]) in marginal_ladder
        rows.append({
            "model": str(r["model"]),
            "macro_intra_rho": rho,
            "delta_rho": (
                np.nan
                if (not is_ladder) or prev is None or not np.isfinite(rho) or not np.isfinite(prev)
                else rho - prev
            ),
            "macro_intra_recall10": rec,
            "macro_ndcg": ndcg,
            "macro_intra_pai": pai,
        })
        if is_ladder:
            prev = rho
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c != "model":
            out[c] = out[c].map(lambda v: round(float(v), 3) if pd.notna(v) and np.isfinite(float(v)) else "NaN")
    return out


def write_report(metrics: pd.DataFrame, feature_sets: dict[str, list[str]], test_year: int) -> None:
    tbl = macro_table(metrics)
    fmt = format_table(tbl)
    rows = []
    for name, feats in feature_sets.items():
        rows.append({"model": name, "n_features": len(feats), "features": ", ".join(feats)})
    rows.append({"model": "M5_historial", "n_features": 1, "features": "persistencia geocodificada t-1 directa"})
    feat_tbl = pd.DataFrame(rows)

    main = tbl[tbl["model"].isin(["M0_prior_uniform", "M1_demografia", "M1b_manzana", "M2_osm", "M2a_osm_centrality", "M2b_catastro", "M2c_licencias", "M2d_mercados", "M2e_educacion", "M3_remote", "M4_meta", "M4a_salud", "M4b_seguridad", "M4c_transporte", "M7a_persist", "M7b_historial_esp", "M5_historial", "M6_stgnn_ref"])].copy()
    best = main.loc[main["macro_intra_rho"].idxmax()] if main["macro_intra_rho"].notna().any() else None
    m5 = main[main["model"] == "M5_historial"]
    m4 = main[main["model"] == "M4_meta"]
    hist_gain = np.nan
    if not m5.empty and not m4.empty:
        hist_gain = float(m5["macro_intra_rho"].iloc[0] - m4["macro_intra_rho"].iloc[0])

    lines = [
        "# Q4 — Ladder de modalidades M0-M6",
        "",
        f"**Train:** 2019-2022. **Test:** {test_year}. **Target/oraculo:** crimen geocodificado real H3.",
        "",
        "## Resultado principal",
        "",
    ]
    if best is not None:
        lines.append(f"- Mejor modelo ladder: **{best['model']}** con macro intra-rho={best['macro_intra_rho']:.3f}.")
    if np.isfinite(hist_gain):
        lines.append(f"- Ganancia de historial geocodificado sobre M4: Δrho={hist_gain:+.3f}.")
    comparisons = [
        ("M1b_manzana", "M1_demografia", "Manzana censal sobre demografia distrital"),
        ("M2_osm", "M1b_manzana", "OSM sobre manzana censal"),
        ("M2a_osm_centrality", "M2_osm", "Centralidad OSM sobre OSM basico"),
        ("M2b_catastro", "M2a_osm_centrality", "COFOPRI sobre OSM+centralidad"),
        ("M2c_licencias", "M2b_catastro", "Licencias sobre catastro"),
        ("M2d_mercados", "M2c_licencias", "Mercados/CENAMA sobre licencias"),
        ("M2d_mercados", "M2b_catastro", "Mercados/CENAMA sobre catastro"),
        ("M2e_educacion", "M2d_mercados", "ESCALE/MINEDU sobre mercados"),
    ]
    for hi, lo, label in comparisons:
        hi_row = main[main["model"] == hi]
        lo_row = main[main["model"] == lo]
        if not hi_row.empty and not lo_row.empty:
            hi_rho = float(hi_row["macro_intra_rho"].iloc[0])
            lo_rho = float(lo_row["macro_intra_rho"].iloc[0])
            if np.isfinite(hi_rho) and np.isfinite(lo_rho):
                lines.append(f"- {label}: Δrho={hi_rho - lo_rho:+.3f}.")
    m2 = main[main["model"] == "M2_osm"]
    m6 = main[main["model"] == "M6_stgnn_ref"]
    if not m2.empty and not m6.empty:
        m2_rho = float(m2["macro_intra_rho"].iloc[0])
        m6_rho = float(m6["macro_intra_rho"].iloc[0])
        if np.isfinite(m2_rho) and np.isfinite(m6_rho):
            lines.append(f"- STGNN ref vs M2_osm: Δrho={m6_rho - m2_rho:+.3f}.")
    lines.extend([
        "- Las filas `prior_*_ref` son referencias directas del prior ladder, no modelos entrenados.",
        "- `M0_prior_uniform` tiene rho=NaN porque una prediccion uniforme no induce ranking intra-distrital.",
        "- `M1b_manzana`, `M2a_osm_centrality`, `M2b_catastro`, `M2c_licencias`, `M2d_mercados` y `M2e_educacion` separan las fuentes finas nuevas para medir su aporte marginal.",
        "",
        "## Macro y ganancia marginal",
        "",
        fmt.to_markdown(index=False),
        "",
        "## Feature sets",
        "",
        feat_tbl.to_markdown(index=False),
        "",
        "## Veredicto Q4",
        "",
        "La lectura debe hacerse por ganancia marginal: primero demografia distrital, luego manzana censal, OSM basico, centralidad vial OSM, catastro COFOPRI, licencias municipales, mercados/CENAMA, ESCALE/MINEDU, sensores remotos y Meta. El aporte arquitectonico DL se evalua aparte contra la mejor modalidad tabular.",
        "",
        f"Metricas completas: `{OUT_METRICS.relative_to(ROOT)}`.",
    ])
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-year", type=int, default=2023)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    panel, feature_sets = load_panel()
    preds, model_metrics = train_models(panel, feature_sets, args.test_year)
    metric_rows = model_metrics.to_dict("records")
    preds = add_m0_prior(preds, metric_rows, args.test_year)
    preds = add_history_direct(preds, metric_rows)
    preds = add_stgnn_reference(preds, metric_rows, args.test_year)
    metrics = pd.DataFrame(metric_rows)

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    write_report(metrics, feature_sets, args.test_year)

    print(f"metrics -> {OUT_METRICS}")
    print(f"report  -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
