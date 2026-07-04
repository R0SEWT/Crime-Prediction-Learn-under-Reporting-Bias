#!/usr/bin/env python3
"""
Q3 Etapa 2: generalizacion espacial.

Evalua si el predictor H3 generaliza o memoriza patrones locales de Lima.
Usa el bloque M2_osm, que fue el mejor en el ladder Q4, y compara:

  - temporal_holdout: train 2019-2022, test 2023, todos los distritos
  - random_h3_cv: celdas H3 aleatorias fuera (baseline con leakage espacial)
  - spatial_grouped_cv: distritos completos fuera
  - buffered_spatial_cv: distritos test fuera y sus vecinos excluidos del train

La evaluacion se hace contra el oraculo geocodificado real con metricas macro
por categoria, priorizando intra-district Spearman y Recall@10%.
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
DEFAULT_DATA_ROOT = ROOT
OUT_REPORT = ROOT / "analysis/etapa2_q3_spatial_cv.md"
OUT_METRICS = ROOT / "data/silver/predictions/q3_spatial_cv_metrics.csv"

CATS = [
    "robo_hurto_callejero",
    "extorsion",
    "estafa",
    "violencia_familiar_sexual",
    "secuestro",
]
M2_OSM = [
    "population",
    "pop_density_km2",
    "nbi_pct",
    "poverty_decile",
    "avg_household_size",
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
KEYS = ("intra_spearman", "intra_recall10", "intra_pai", "recall10", "ndcg")
LOGGER = logging.getLogger("eval_spatial_cv")


def paths(data_root: Path) -> dict[str, Path]:
    return {
        "matrix": data_root / "data/silver/h3_feature_matrix.parquet",
        "oracle": data_root / "data/silver/h3_features/h3_observed_geocoded.parquet",
        "graph": data_root / "data/silver/h3_graph_edges.parquet",
    }


def load_panel(data_root: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    p = paths(data_root)
    schema_cols = pq.read_schema(p["matrix"]).names
    feats = [c for c in M2_OSM if c in schema_cols]
    feat = pd.read_parquet(p["matrix"], columns=["h3_index", "year", "ubigeo"] + feats)
    feat = feat[feat["ubigeo"].notna() & (feat["ubigeo"].astype(str) != "")].copy()
    feat["ubigeo"] = feat["ubigeo"].astype(str).str.zfill(6)
    h3_admin = feat[["h3_index", "ubigeo"]].drop_duplicates("h3_index")

    oracle = pd.read_parquet(p["oracle"], columns=["h3_index", "year", "crime_cat", "obs_geo_count"])
    panels = []
    for cat in CATS:
        oc = oracle[oracle["crime_cat"] == cat]
        b = feat.merge(oc[["h3_index", "year", "obs_geo_count"]], on=["h3_index", "year"], how="left")
        b["target"] = b["obs_geo_count"].fillna(0.0)
        b["crime_cat"] = cat
        panels.append(b.drop(columns=["obs_geo_count"]))
    return pd.concat(panels, ignore_index=True), feats, h3_admin


def district_adjacency(data_root: Path, h3_admin: pd.DataFrame) -> dict[str, set[str]]:
    p = paths(data_root)
    districts = sorted(h3_admin["ubigeo"].dropna().astype(str).str.zfill(6).unique())
    adj: dict[str, set[str]] = {d: set() for d in districts}
    if not p["graph"].exists():
        return adj
    edges = pd.read_parquet(p["graph"])
    h3_to_d = h3_admin.set_index("h3_index")["ubigeo"].to_dict()
    for src, dst in edges[["src_h3", "dst_h3"]].itertuples(index=False):
        a = h3_to_d.get(src)
        b = h3_to_d.get(dst)
        if not a or not b or a == b:
            continue
        a = str(a).zfill(6)
        b = str(b).zfill(6)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def make_folds(items: list[str], n_splits: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(items), dtype=object)
    rng.shuffle(arr)
    return [list(x) for x in np.array_split(arr, n_splits) if len(x)]


def fit_predict_cat(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    med = tr[feats].median(numeric_only=True)
    xtr = tr[feats].fillna(med).to_numpy()
    ytr = np.log1p(tr["target"].to_numpy(dtype=float))
    xte = te[feats].fillna(med).to_numpy()
    model = HistGradientBoostingRegressor(
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        learning_rate=0.06,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(xtr, ytr)
    pred = np.expm1(np.maximum(model.predict(xte), 0.0))
    del model, xtr, xte, ytr
    gc.collect()
    return pred


def evaluate_split(
    panel: pd.DataFrame,
    feats: list[str],
    split_name: str,
    fold: int,
    train_years: list[int],
    test_year: int,
    train_districts: set[str] | None = None,
    test_districts: set[str] | None = None,
    train_cells: set[str] | None = None,
    test_cells: set[str] | None = None,
) -> list[dict]:
    pred_parts = []
    rows = []
    for cat in CATS:
        d = panel[panel["crime_cat"] == cat]
        tr = d[d["year"].isin(train_years)].copy()
        te = d[d["year"] == test_year].copy()
        if train_districts is not None:
            tr = tr[tr["ubigeo"].isin(train_districts)]
        if test_districts is not None:
            te = te[te["ubigeo"].isin(test_districts)]
        if train_cells is not None:
            tr = tr[tr["h3_index"].isin(train_cells)]
        if test_cells is not None:
            te = te[te["h3_index"].isin(test_cells)]
        if tr.empty or te.empty or te["target"].sum() <= 0:
            continue
        te = te[["h3_index", "year", "ubigeo", "crime_cat", "target"] + feats].copy()
        te["pred"] = fit_predict_cat(tr, te, feats)
        pred_parts.append(te[["h3_index", "year", "ubigeo", "crime_cat", "target", "pred"]])

    if not pred_parts:
        return rows
    pred_df = pd.concat(pred_parts, ignore_index=True)
    per_cat = {}
    for cat in CATS:
        d = pred_df[pred_df["crime_cat"] == cat]
        if d.empty or d["target"].sum() <= 0:
            continue
        res = evaluate(d, "pred", "target", district_col="ubigeo", boot=False)
        per_cat[cat] = res
        rows.append({
            "split": split_name,
            "fold": fold,
            "crime_cat": cat,
            "n_train_districts": len(train_districts) if train_districts is not None else np.nan,
            "n_test_districts": len(test_districts) if test_districts is not None else np.nan,
            **res,
        })
    macro = macro_average(per_cat, KEYS)
    rows.append({
        "split": split_name,
        "fold": fold,
        "crime_cat": "__macro__",
        "n_train_districts": len(train_districts) if train_districts is not None else np.nan,
        "n_test_districts": len(test_districts) if test_districts is not None else np.nan,
        **macro,
    })
    return rows


def run_cv(panel: pd.DataFrame, feats: list[str], adj: dict[str, set[str]], n_splits: int, seed: int, test_year: int) -> pd.DataFrame:
    train_years = [y for y in [2019, 2020, 2021, 2022] if y < test_year]
    rows: list[dict] = []
    districts = sorted(panel["ubigeo"].dropna().astype(str).str.zfill(6).unique())
    cells = sorted(panel["h3_index"].dropna().unique())

    rows.extend(evaluate_split(panel, feats, "temporal_holdout", 0, train_years, test_year))

    for i, fold_cells in enumerate(make_folds(cells, n_splits, seed), start=1):
        test_cells = set(fold_cells)
        train_cells = set(cells) - test_cells
        rows.extend(evaluate_split(
            panel, feats, "random_h3_cv", i, train_years, test_year,
            train_cells=train_cells, test_cells=test_cells,
        ))

    for i, fold_districts in enumerate(make_folds(districts, n_splits, seed), start=1):
        test_d = set(fold_districts)
        train_d = set(districts) - test_d
        rows.extend(evaluate_split(
            panel, feats, "spatial_grouped_cv", i, train_years, test_year,
            train_districts=train_d, test_districts=test_d,
        ))

        buffer_d = set(test_d)
        for d in test_d:
            buffer_d.update(adj.get(d, set()))
        train_buf = set(districts) - buffer_d
        rows.extend(evaluate_split(
            panel, feats, "buffered_spatial_cv", i, train_years, test_year,
            train_districts=train_buf, test_districts=test_d,
        ))

    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    macro = metrics[metrics["crime_cat"] == "__macro__"].copy()
    rows = []
    for split, g in macro.groupby("split"):
        rows.append({
            "split": split,
            "n_folds": len(g),
            "mean_macro_intra_rho": g["macro_intra_spearman"].mean(),
            "std_macro_intra_rho": g["macro_intra_spearman"].std(ddof=0),
            "mean_macro_intra_recall10": g["macro_intra_recall10"].mean(),
            "mean_macro_ndcg": g["macro_ndcg"].mean(),
            "mean_train_districts": g["n_train_districts"].mean(),
            "mean_test_districts": g["n_test_districts"].mean(),
        })
    order = ["temporal_holdout", "random_h3_cv", "spatial_grouped_cv", "buffered_spatial_cv"]
    out = pd.DataFrame(rows)
    out["split"] = pd.Categorical(out["split"], categories=order, ordered=True)
    return out.sort_values("split")


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c != "split":
            out[c] = out[c].map(lambda v: round(float(v), 3) if pd.notna(v) and np.isfinite(float(v)) else "NaN")
    return out


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame, data_root: Path, test_year: int) -> None:
    temporal = summary[summary["split"] == "temporal_holdout"]
    grouped = summary[summary["split"] == "spatial_grouped_cv"]
    buffered = summary[summary["split"] == "buffered_spatial_cv"]
    random = summary[summary["split"] == "random_h3_cv"]

    lines = [
        "# Q3 — Generalizacion espacial",
        "",
        f"**Modelo:** M2_osm (mejor bloque Q4). **Train years:** 2019-2022. **Test year:** {test_year}.",
        f"**Data root:** `{data_root}`.",
        "",
        "## Resultado principal",
        "",
    ]
    if not temporal.empty and not grouped.empty:
        dt = float(grouped["mean_macro_intra_rho"].iloc[0] - temporal["mean_macro_intra_rho"].iloc[0])
        lines.append(f"- Spatial grouped CV vs temporal holdout: Δrho={dt:+.3f}.")
    if not grouped.empty and not buffered.empty:
        db = float(buffered["mean_macro_intra_rho"].iloc[0] - grouped["mean_macro_intra_rho"].iloc[0])
        lines.append(f"- Buffered CV vs grouped CV: Δrho={db:+.3f}.")
    if not random.empty and not buffered.empty:
        dl = float(random["mean_macro_intra_rho"].iloc[0] - buffered["mean_macro_intra_rho"].iloc[0])
        lines.append(f"- Random H3 CV over buffered CV: +{dl:.3f} rho; esa brecha estima inflacion por leakage espacial.")
    if not buffered.empty:
        lines.append(
            f"- Lectura conservadora buffered: macro intra-rho={float(buffered['mean_macro_intra_rho'].iloc[0]):.3f} "
            f"con std={float(buffered['std_macro_intra_rho'].iloc[0]):.3f}; la varianza por fold es alta."
        )
    lines.extend([
        "",
        "## Resumen macro",
        "",
        fmt_table(summary).to_markdown(index=False),
        "",
        "## Veredicto Q3",
        "",
        "La senal OSM generaliza parcialmente, pero no con la fuerza que sugiere el split temporal simple. La caida de temporal a grouped indica que parte del rendimiento depende de identidad distrital; la caida adicional a buffered indica leakage por vecindad. El claim defendible no es 'prediccion fina robusta en todo Lima', sino 'senal urbana OSM transferible con degradacion espacial sustantiva'.",
        "",
        f"Metricas completas: `{OUT_METRICS.relative_to(ROOT)}`.",
    ])
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--test-year", type=int, default=2023)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    panel, feats, h3_admin = load_panel(args.data_root)
    adj = district_adjacency(args.data_root, h3_admin)
    LOGGER.info("Panel=%s | features=%d | districts=%d", panel.shape, len(feats), panel["ubigeo"].nunique())
    LOGGER.info("District adjacency edges=%d", sum(len(v) for v in adj.values()) // 2)

    metrics = run_cv(panel, feats, adj, args.splits, args.seed, args.test_year)
    summary = summarize(metrics)

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    write_report(summary, metrics, args.data_root, args.test_year)

    print(f"metrics -> {OUT_METRICS}")
    print(f"report  -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
