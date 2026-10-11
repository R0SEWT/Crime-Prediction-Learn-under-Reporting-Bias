#!/usr/bin/env python3
"""
Degradación sintética de geocodificación: curva de evaluabilidad (slr-dj7).

Convierte el hallazgo Trujillo (24.9% coords -> inevaluable) en experimento:
adelgaza binomialmente los conteos del oráculo de Lima para simular tasas de
geocodificación menores (la base 2019-2023 es ~83.4%), re-entrena el modelo
tabular techo (M4c) sobre los targets degradados, y mide a cada nivel:

  - skill MEDIBLE: rho intra vs oráculo degradado (lo que vería un analista
    en una ciudad con ese registro),
  - skill REAL: rho intra vs oráculo completo (lo que de verdad aprendió),
  - estabilidad del ranking features-vs-persistencia entre seeds.

El adelgazamiento binomial (cada evento retiene su coordenada con prob. p)
asume pérdida ALEATORIA de geocodificación — cota optimista: la pérdida real
(Trujillo) es probablemente selectiva por comisaría, y peor.

Uso: python3 scripts/exp_geocoding_degradation.py
Salida: analysis/etapa2_geocoding_degradation.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_modality_ladder import CATS, FEATURE_SETS, available, load_panel  # noqa: E402

OUT_MD = ROOT / "analysis/etapa2_geocoding_degradation.md"
BASE_RATE = 0.834          # tasa real de coords 2019-2023 en Lima (filtro paper)
RATES = [0.834, 0.70, 0.50, 0.25, 0.10]   # 0.25 ~ nivel Trujillo
N_SEEDS = 10
TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023
MIN_CELLS = 5
LOGGER = logging.getLogger("exp_geocoding_degradation")


def thin(counts: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(counts.astype(int), p).astype(float)


def intra_macro(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    per_cat = []
    for cat, dcat in df.groupby("crime_cat"):
        rhos = []
        for _, g in dcat.groupby("ubigeo"):
            if len(g) < MIN_CELLS or g[target_col].nunique() < 2 or g[pred_col].nunique() < 2:
                continue
            r, _ = spearmanr(g[pred_col], g[target_col])
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            per_cat.append(np.mean(rhos))
    return float(np.mean(per_cat)) if per_cat else float("nan")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from sklearn.ensemble import HistGradientBoostingRegressor

    panel, _ = load_panel()
    feats = available(FEATURE_SETS["M4c_transporte"], panel)
    LOGGER.info("Features M4c: %d", len(feats))
    panel = panel[["h3_index", "ubigeo", "year", "crime_cat", "target", "persist"] + feats].copy()

    rows = []
    for rate in RATES:
        p = min(1.0, rate / BASE_RATE)
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 * seed + int(rate * 100))
            d = panel.copy()
            if p < 1.0:
                d["target_thin"] = thin(d["target"].to_numpy(), p, rng)
                d["persist_thin"] = thin(d["persist"].to_numpy(), p, rng)
            else:
                d["target_thin"] = d["target"]
                d["persist_thin"] = d["persist"]

            te_parts = []
            for cat in CATS:
                dc = d[d["crime_cat"] == cat]
                tr = dc[dc["year"].isin(TRAIN_YEARS)]
                te = dc[dc["year"] == TEST_YEAR].copy()
                med = tr[feats].median(numeric_only=True)
                m = HistGradientBoostingRegressor(
                    max_iter=140, max_leaf_nodes=15, min_samples_leaf=20,
                    learning_rate=0.06, l2_regularization=0.05, random_state=42)
                m.fit(tr[feats].fillna(med).to_numpy(),
                      np.log1p(tr["target_thin"].to_numpy(float)))
                te["pred_feat"] = np.expm1(np.maximum(
                    m.predict(te[feats].fillna(med).to_numpy()), 0.0))
                te_parts.append(te)
            te_all = pd.concat(te_parts, ignore_index=True)

            medible = intra_macro(te_all, "pred_feat", "target_thin")
            real = intra_macro(te_all, "pred_feat", "target")
            persist_med = intra_macro(te_all, "persist_thin", "target_thin")
            rows.append({"rate": rate, "seed": seed, "medible": medible,
                         "real": real, "persist": persist_med,
                         "feat_gana": medible > persist_med})
            if seed == 0:
                LOGGER.info("rate=%.2f seed=0: medible=%.3f real=%.3f persist=%.3f",
                            rate, medible, real, persist_med)

    res = pd.DataFrame(rows)
    agg = res.groupby("rate").agg(
        medible=("medible", "mean"), medible_sd=("medible", "std"),
        real=("real", "mean"), real_sd=("real", "std"),
        persist=("persist", "mean"), feat_gana_pct=("feat_gana", "mean"),
    ).reset_index().sort_values("rate", ascending=False)

    lines = [
        "# Degradación sintética de geocodificación: curva de evaluabilidad (slr-dj7)",
        "",
        f"**Diseño:** adelgazamiento binomial de los conteos del oráculo de Lima "
        f"(base {BASE_RATE:.1%}) hacia tasas menores, {N_SEEDS} seeds por nivel. El modelo "
        f"tabular techo (M4c) se RE-ENTRENA sobre los targets degradados; la persistencia "
        f"usa el historial degradado. 'Medible' = rho intra contra el oráculo degradado "
        f"(lo que vería el analista); 'real' = contra el oráculo completo (skill verdadero). "
        f"Supuesto declarado: pérdida aleatoria — cota optimista frente a la pérdida "
        f"selectiva real (Trujillo).",
        "",
        "| tasa geocod. | ρ medible (±sd) | ρ real (±sd) | ρ persistencia | % seeds features>persist |",
        "|--:|:--|:--|--:|--:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['rate']:.0%} | {r['medible']:.3f} (±{r['medible_sd']:.3f}) "
            f"| {r['real']:.3f} (±{r['real_sd']:.3f}) | {r['persist']:.3f} "
            f"| {r['feat_gana_pct']:.0%} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    res.to_csv(ROOT / "data/silver/predictions/geocoding_degradation.csv", index=False)
    print(f"\nGuardado → {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
