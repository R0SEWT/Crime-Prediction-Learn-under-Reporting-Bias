#!/usr/bin/env python3
"""
Q1 Etapa 2: observado vs latente.

Pregunta: el target latente sirve mas que el target observado?

Este experimento no entrena modelos. Compara las dos superficies H3 disponibles:

  - observado: denuncias MININTER redistribuidas a H3
  - latente: observado / r_hat redistribuido a H3

contra el oraculo geocodificado de denuncias reales. La prueba clave es si el
latente cambia la lectura de riesgo frente al observado, y en que nivel:
intra-distrital, global, magnitud o composicion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa2_metrics import evaluate, macro_average

ROOT = Path(__file__).resolve().parents[1]
SURFACE_FILE = ROOT / "data/silver/crime_latent_surface.parquet"
ORACLE_FILE = ROOT / "data/silver/h3_features/h3_observed_geocoded.parquet"
VALIDATION_FILE = ROOT / "data/datasets/silver/latent_surface/validacion.json"
COMPOSITION_FILE = ROOT / "data/silver/analysis/composition_excl_dv.json"
OUT_REPORT = ROOT / "analysis/etapa2_q1_observado_vs_latente.md"
OUT_METRICS = ROOT / "data/silver/predictions/q1_observed_vs_latent_metrics.csv"

CATS = [
    "robo_hurto_callejero",
    "extorsion",
    "estafa",
    "violencia_familiar_sexual",
    "secuestro",
]
PREDICTORS = ("observado", "latente")
PRIMARY_YEAR = 2023


def load_panel() -> pd.DataFrame:
    surface = pd.read_parquet(
        SURFACE_FILE,
        columns=["h3_index", "ubigeo", "year", "crime_cat", "observado", "latente", "r_hat", "inestable"],
    )
    oracle = pd.read_parquet(
        ORACLE_FILE,
        columns=["h3_index", "ubigeo", "year", "crime_cat", "obs_geo_count"],
    )
    panel = surface.merge(
        oracle,
        on=["h3_index", "year", "crime_cat"],
        how="outer",
        suffixes=("_surface", "_oracle"),
    )
    panel["ubigeo"] = panel["ubigeo_surface"].combine_first(panel["ubigeo_oracle"])
    panel["ubigeo"] = panel["ubigeo"].astype(str).str.zfill(6)
    for col in ("observado", "latente", "obs_geo_count"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    panel["inestable"] = pd.to_numeric(panel["inestable"], errors="coerce").fillna(0).astype(int)
    return panel[panel["crime_cat"].isin(CATS)].copy()


def evaluate_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    macro_rows = []
    for year in sorted(panel["year"].dropna().unique()):
        year = int(year)
        for pred in PREDICTORS:
            per_cat = {}
            for cat in CATS:
                d = panel[(panel["year"] == year) & (panel["crime_cat"] == cat)].copy()
                if d.empty or d["obs_geo_count"].sum() <= 0:
                    continue
                res = evaluate(d, pred, "obs_geo_count", district_col="ubigeo", boot=False)
                per_cat[cat] = res
                rows.append({
                    "year": year,
                    "predictor": pred,
                    "crime_cat": cat,
                    "n_cells": res["n_cells"],
                    "oracle_sum": float(d["obs_geo_count"].sum()),
                    "pred_sum": float(d[pred].sum()),
                    "spearman_global": res["spearman_global"],
                    "recall10": res["recall10"],
                    "ndcg": res["ndcg"],
                    "nmae": res["nmae"],
                    "intra_spearman": res["intra_spearman"],
                    "intra_recall10": res["intra_recall10"],
                    "intra_pai": res["intra_pai"],
                    "intra_n_districts": res["intra_n_districts"],
                })
            if per_cat:
                macro = macro_average(
                    per_cat,
                    ("spearman_global", "recall10", "ndcg", "nmae", "intra_spearman", "intra_recall10", "intra_pai"),
                )
                macro_rows.append({"year": year, "predictor": pred, **macro})
    return pd.DataFrame(rows), pd.DataFrame(macro_rows)


def proportionality(surface: pd.DataFrame) -> dict:
    pos = surface[(surface["observado"] > 0) & (surface["latente"] > 0)].copy()
    pos["factor"] = pos["latente"] / pos["observado"]
    g = pos.groupby(["year", "crime_cat", "ubigeo"])["factor"].agg(["count", "std", "nunique"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    tol = 1e-5
    return {
        "groups": int(len(g)),
        "tolerance": tol,
        "share_constant": float((g["std"].abs() < tol).mean()) if len(g) else np.nan,
        "max_std": float(g["std"].max()) if len(g) else np.nan,
        "max_nunique": int(g["nunique"].max()) if len(g) else 0,
    }


def composition(surface: pd.DataFrame, year: int) -> pd.DataFrame:
    d = surface[surface["year"] == year].groupby("crime_cat")[["observado", "latente"]].sum().reindex(CATS).fillna(0.0)
    for col in ("observado", "latente"):
        total = d[col].sum()
        d[f"share_{col}"] = d[col] / total if total > 0 else np.nan
    d["delta_share_latente_minus_obs"] = d["share_latente"] - d["share_observado"]
    return d.reset_index()


def fmt(x: float, nd: int = 3) -> str:
    return "NaN" if not np.isfinite(x) else f"{x:.{nd}f}"


def write_report(panel: pd.DataFrame, metrics: pd.DataFrame, macro: pd.DataFrame) -> None:
    surface = panel[["h3_index", "ubigeo", "year", "crime_cat", "observado", "latente", "r_hat", "inestable"]].copy()
    prop = proportionality(surface)

    primary_macro = macro[macro["year"] == PRIMARY_YEAR].copy()
    primary_rows = metrics[metrics["year"] == PRIMARY_YEAR].copy()

    wide_macro = primary_macro[[
        "predictor",
        "macro_intra_spearman",
        "macro_spearman_global",
        "macro_intra_recall10",
        "macro_recall10",
        "macro_ndcg",
        "macro_nmae",
    ]].copy()
    for col in wide_macro.columns:
        if col != "predictor":
            wide_macro[col] = wide_macro[col].map(lambda v: round(v, 3) if np.isfinite(v) else "NaN")

    by_cat = primary_rows[[
        "predictor",
        "crime_cat",
        "intra_spearman",
        "spearman_global",
        "intra_recall10",
        "recall10",
        "ndcg",
        "nmae",
    ]].copy()
    for col in by_cat.columns:
        if col not in ("predictor", "crime_cat"):
            by_cat[col] = by_cat[col].map(lambda v: round(v, 3) if np.isfinite(v) else "NaN")

    comp = composition(surface, PRIMARY_YEAR)
    comp_tbl = comp[["crime_cat", "share_observado", "share_latente", "delta_share_latente_minus_obs"]].copy()
    for col in comp_tbl.columns:
        if col != "crime_cat":
            comp_tbl[col] = comp_tbl[col].map(lambda v: round(v, 3) if np.isfinite(v) else "NaN")

    validation_note = ""
    if VALIDATION_FILE.exists():
        validation = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
        # L1 sale del artefacto canonico (build_composition_canon.py), no de validacion.json:
        # requiere excluir violencia familiar y renormalizar sobre las 4 categorias comparables.
        # La clave vieja 'composicion_L1' traia el titular falso 0.99->0.11 (B1, slr-x50g).
        rho_obs = validation["espacial_total_sin_estafa"]["rho_obs"]
        rho_lat = validation["espacial_total_sin_estafa"]["rho_lat"]
        lines = ["## Validacion Etapa 1 contra ENAPRES", ""]
        if COMPOSITION_FILE.exists():
            comp_canon = json.loads(COMPOSITION_FILE.read_text(encoding="utf-8"))
            base = comp_canon["canonical_base"]
            l1 = comp_canon["l1"][base]
            lines.append(
                f"- Composicion L1 vs ENAPRES ({base}, 4 cats excl. violencia familiar): "
                f"observado={l1['observed']:.3f}, latente={l1['latent']:.3f} "
                f"({100 * (l1['latent'] / l1['observed'] - 1):+.0f}%)."
            )
        else:
            lines.append("- Composicion L1: falta composition_excl_dv.json "
                         "(correr scripts/build_composition_canon.py).")
        lines += [
            f"- Ranking distrital total sin estafa: observado rho={rho_obs:.3f}, latente rho={rho_lat:.3f}.",
            "",
            "Lectura: el de-sesgo corrige MAGNITUD, pero ALEJA la composicion por tipo del "
            "benchmark de encuesta (la estafa se sobre-corrige por mismatch de universo) y "
            "tampoco mejora el ranking espacial distrital.",
            "",
        ]
        validation_note = "\n".join(lines)

    obs_intra = float(primary_macro[primary_macro["predictor"] == "observado"]["macro_intra_spearman"].iloc[0])
    lat_intra = float(primary_macro[primary_macro["predictor"] == "latente"]["macro_intra_spearman"].iloc[0])
    obs_global = float(primary_macro[primary_macro["predictor"] == "observado"]["macro_spearman_global"].iloc[0])
    lat_global = float(primary_macro[primary_macro["predictor"] == "latente"]["macro_spearman_global"].iloc[0])

    lines = [
        "# Q1 — Observado vs latente",
        "",
        f"**Anio principal:** {PRIMARY_YEAR}. **Oraculo:** denuncias con coordenada real agregadas a H3.",
        "",
        "## Resultado principal",
        "",
        f"- Intra-distrital: observado macro rho={fmt(obs_intra)}, latente macro rho={fmt(lat_intra)}.",
        f"- Global contra oraculo geocodificado: observado macro rho={fmt(obs_global)}, latente macro rho={fmt(lat_global)}.",
        f"- Proporcionalidad H3 dentro de distrito/categoria/anio: {prop['share_constant']:.1%} de grupos con factor constante en tolerancia {prop['tolerance']:.0e}; max std={prop['max_std']:.3g}.",
        "",
        "Interpretacion: el target latente cambia escala y composicion, pero la superficie H3 actual no agrega nueva senal intra-distrital porque observado y latente usan el mismo peso de redistribucion dentro de cada distrito.",
        "",
        "## Macro 2023",
        "",
        wide_macro.to_markdown(index=False),
        "",
        "## Detalle por categoria 2023",
        "",
        by_cat.to_markdown(index=False),
        "",
        "## Composicion H3 2023",
        "",
        comp_tbl.to_markdown(index=False),
        "",
        validation_note,
        "## Veredicto Q1",
        "",
        "El target latente si sirve para la pregunta de crimen no reportado: corrige magnitud y composicion frente al observado. No sirve, tal como esta construido a H3, para demostrar resolucion fina intra-distrital. Para una STGNN, esto implica que entrenar sobre la superficie latente poblacional-distribuida no prueba aprendizaje de riesgo latente intra-H3; solo prueba aprendizaje del prior de redistribucion.",
        "",
        f"Metricas completas: `{OUT_METRICS.relative_to(ROOT)}`.",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    panel = load_panel()
    metrics, macro = evaluate_panel(panel)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    write_report(panel, metrics, macro)
    print(f"metrics -> {OUT_METRICS}")
    print(f"report  -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
