#!/usr/bin/env python3
"""
Robustez temporal con test 2022 para los contrastes SIMBig (slr-nvc).

Repite train 2019-2021 / test 2022 y compara contra test 2023 los contrastes:
ladder tabular M1->M1b->M4c, transferencia Lima->Arequipa y techo M4c.
Todos los deltas inferenciales usan bootstrap pareado sobre distritos.
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

from eval_modality_ladder import CATS, FEATURE_SETS, available, load_panel  # noqa: E402
from exp_learning_curve import build_city_panel  # noqa: E402
from transfer_arequipa import TRANSFER_FEATS  # noqa: E402

OUT_MD = ROOT / "analysis/etapa2_temporal_robustness.md"
OUT_CSV = ROOT / "data/silver/predictions/temporal_robustness_2022.csv"
MIN_CELLS = 5
B = 2000
LOGGER = logging.getLogger("exp_temporal_robustness")


def train_years_for(test_year: int) -> list[int]:
    return [y for y in [2019, 2020, 2021, 2022] if y < test_year]


def fit_predict_hgb(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    med = tr[feats].median(numeric_only=True)
    model = HistGradientBoostingRegressor(
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        learning_rate=0.06,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(tr[feats].fillna(med).to_numpy(), np.log1p(tr["target"].to_numpy(float)))
    return np.expm1(np.maximum(model.predict(te[feats].fillna(med).to_numpy()), 0.0))


def intra_stats(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (cat, ub), g in df.groupby(["crime_cat", "ubigeo"]):
        if len(g) < MIN_CELLS or g["target"].nunique() < 2 or g[pred_col].nunique() < 2:
            continue
        rho, _ = spearmanr(g[pred_col], g["target"])
        if np.isfinite(rho):
            rows.append({"crime_cat": cat, "ubigeo": str(ub).zfill(6), "rho": float(rho)})
    return pd.DataFrame(rows)


def macro(st: pd.DataFrame) -> float:
    return float(st.groupby("crime_cat").rho.mean().mean()) if len(st) else float("nan")


def boot_paired_delta(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    m = a.merge(b, on=["crime_cat", "ubigeo"], suffixes=("_a", "_b"))
    by_cat = {c: g[["rho_a", "rho_b"]].dropna().to_numpy() for c, g in m.groupby("crime_cat")}
    draws = np.empty(B)
    for i in range(B):
        deltas = []
        for v in by_cat.values():
            if not len(v):
                continue
            idx = rng.integers(0, len(v), size=len(v))
            deltas.append(v[idx, 1].mean() - v[idx, 0].mean())
        draws[i] = np.mean(deltas) if deltas else np.nan
    point = float(np.mean([v[:, 1].mean() - v[:, 0].mean() for v in by_cat.values() if len(v)]))
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def signif(ci_low: float, ci_high: float) -> str:
    return "si" if (ci_low > 0 or ci_high < 0) else "no"


def ladder_stats(panel: pd.DataFrame, test_year: int) -> dict[str, pd.DataFrame]:
    steps = ["M1_demografia", "M1b_manzana", "M4c_transporte"]
    out = {}
    yrs = train_years_for(test_year)
    for step in steps:
        feats = available(FEATURE_SETS[step], panel)
        LOGGER.info("Ladder %s test=%s: %d features", step, test_year, len(feats))
        parts = []
        for cat in CATS:
            d = panel[panel["crime_cat"] == cat]
            tr = d[d["year"].isin(yrs)]
            te = d[d["year"] == test_year].copy()
            te[f"pred_{step}"] = fit_predict_hgb(tr, te, feats)
            parts.append(te[["h3_index", "ubigeo", "crime_cat", "target", f"pred_{step}"]])
        pred = pd.concat(parts, ignore_index=True)
        out[step] = intra_stats(pred, f"pred_{step}")
    return out


def add_city_persist(city: pd.DataFrame) -> pd.DataFrame:
    lag = city[["h3_index", "year", "crime_cat", "target"]].copy()
    lag["year"] = lag["year"] + 1
    return city.merge(
        lag.rename(columns={"target": "persist"}),
        on=["h3_index", "year", "crime_cat"],
        how="left",
    ).assign(persist=lambda d: d["persist"].fillna(0.0))


def transfer_stats(lima: pd.DataFrame, city: pd.DataFrame, test_year: int) -> dict[str, pd.DataFrame]:
    city = add_city_persist(city)
    feats = [f for f in TRANSFER_FEATS if f in lima.columns and f in city.columns]
    yrs = train_years_for(test_year)
    LOGGER.info("Transfer AQP test=%s: %d features, train=%s", test_year, len(feats), yrs)
    test = city[city["year"] == test_year].copy()
    parts = []
    for cat in CATS:
        tr = lima[(lima["crime_cat"] == cat) & lima["year"].isin(yrs)]
        te = test[test["crime_cat"] == cat].copy()
        te["pred_transfer"] = fit_predict_hgb(tr, te, feats)
        te["pred_pop"] = te["population"].fillna(0.0)
        te["pred_persist"] = te["persist"].fillna(0.0)
        parts.append(te[[
            "h3_index", "ubigeo", "crime_cat", "target",
            "pred_transfer", "pred_pop", "pred_persist",
        ]])
    pred = pd.concat(parts, ignore_index=True)
    return {
        "transfer": intra_stats(pred, "pred_transfer"),
        "prior_pob": intra_stats(pred, "pred_pop"),
        "persistencia": intra_stats(pred, "pred_persist"),
    }


def collect_rows(panel: pd.DataFrame, city: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    rng = np.random.default_rng(42)
    ladder_by_year = {year: ladder_stats(panel, year) for year in [2022, 2023]}
    transfer_by_year = {year: transfer_stats(panel, city, year) for year in [2022, 2023]}

    for year, stats in ladder_by_year.items():
        for model, st in stats.items():
            rows.append({
                "family": "ladder",
                "test_year": year,
                "comparison": model,
                "metric": "macro_rho",
                "point": macro(st),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "significant": "",
                "replica": "",
            })
        for lo, hi, label in [
            ("M1_demografia", "M1b_manzana", "M1b_manzana - M1_demografia"),
            ("M1b_manzana", "M4c_transporte", "M4c_transporte - M1b_manzana"),
        ]:
            d, dl, dh = boot_paired_delta(stats[lo], stats[hi], rng)
            expected = "positive" if label.startswith("M1b") else "ns"
            replica = (dl > 0 and expected == "positive") or (dl <= 0 <= dh and expected == "ns")
            rows.append({
                "family": "ladder",
                "test_year": year,
                "comparison": label,
                "metric": "delta_macro_rho",
                "point": d,
                "ci_low": dl,
                "ci_high": dh,
                "significant": signif(dl, dh),
                "replica": "si" if replica else "no",
            })

    for year, stats in transfer_by_year.items():
        for model, st in stats.items():
            rows.append({
                "family": "transfer_aqp",
                "test_year": year,
                "comparison": model,
                "metric": "macro_rho",
                "point": macro(st),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "significant": "",
                "replica": "",
            })
        for lo, hi, label in [
            ("prior_pob", "transfer", "transfer - prior_pob"),
            ("persistencia", "transfer", "transfer - persistencia"),
        ]:
            d, dl, dh = boot_paired_delta(stats[lo], stats[hi], rng)
            expected = "positive" if label == "transfer - prior_pob" else "ns_or_negative"
            replica = (dl > 0 and expected == "positive") or (dl <= 0 and expected == "ns_or_negative")
            rows.append({
                "family": "transfer_aqp",
                "test_year": year,
                "comparison": label,
                "metric": "delta_macro_rho",
                "point": d,
                "ci_low": dl,
                "ci_high": dh,
                "significant": signif(dl, dh),
                "replica": "si" if replica else "no",
            })

    m4_2022 = macro(ladder_by_year[2022]["M4c_transporte"])
    m4_2023 = macro(ladder_by_year[2023]["M4c_transporte"])
    notes.append(f"M4c test 2022={m4_2022:.4f}; test 2023={m4_2023:.4f}; delta={m4_2022 - m4_2023:+.4f}.")
    return pd.DataFrame(rows), notes


def fmt_num(v: float) -> str:
    return "" if pd.isna(v) else f"{float(v):.4f}"


def write_report(rows: pd.DataFrame, notes: list[str]) -> None:
    infer = rows[rows["metric"] == "delta_macro_rho"].copy()
    macro_rows = rows[rows["metric"] == "macro_rho"].copy()

    def table(df: pd.DataFrame) -> str:
        d = df.copy()
        for c in ["point", "ci_low", "ci_high"]:
            if c in d.columns:
                d[c] = d[c].map(fmt_num)
        return d.to_markdown(index=False)

    reps_2022 = infer[infer["test_year"] == 2022]["replica"].eq("si").sum()
    total_2022 = len(infer[infer["test_year"] == 2022])
    verdict = "REPLICA PARCIAL" if reps_2022 < total_2022 else "REPLICA"
    if reps_2022 == 0:
        verdict = "NO REPLICA"

    lines = [
        "# Robustez temporal con test 2022 (slr-nvc)",
        "",
        "**Protocolo:** se recalculan los contrastes con train 2019-2021 / test 2022 "
        "y se comparan con train 2019-2022 / test 2023. Modelo tabular HGB con los "
        "mismos parametros del ladder; deltas con bootstrap pareado sobre distritos "
        f"(B={B}).",
        "",
        "## Macros por modelo",
        "",
        table(macro_rows[["family", "test_year", "comparison", "metric", "point"]]),
        "",
        "## Deltas inferenciales",
        "",
        table(infer[[
            "family", "test_year", "comparison", "point", "ci_low",
            "ci_high", "significant", "replica",
        ]]),
        "",
        "## Techo M4c",
        "",
        *[f"- {n}" for n in notes],
        "",
        "## Veredicto global",
        "",
        f"**{verdict}.** En test 2022 replican {reps_2022}/{total_2022} "
        "contrastes inferenciales bajo los criterios preregistrados en este script.",
        "",
        f"Datos tabulares: `{OUT_CSV.relative_to(ROOT)}`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Reporte -> %s", OUT_MD)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    panel, _ = load_panel()
    city = build_city_panel("arequipa")
    rows, notes = collect_rows(panel, city)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
