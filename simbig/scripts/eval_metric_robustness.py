#!/usr/bin/env python3
"""
Robustez de las métricas de evaluación intra-distrital (slr-0vh).

Cuatro análisis sobre el protocolo del ladder Q4 (train 2019-2022, test 2023,
oráculo geocodificado):

1. Bootstrap sobre DISTRITOS (la unidad de réplica de la métrica intra) para
   los escalones clave M1b/M2/M4c: CI del macro-rho y CI PAREADO de los deltas
   entre escalones (mismos índices de resampleo en ambos modelos).
2. Captura@10% intra-distrital (share de eventos del distrito en el top-10% de
   celdas predichas) con CI — co-primaria defendible para categorías raras
   donde Spearman se degrada por empates masivos en cero.
3. Sensibilidad al sesgo del oráculo: macro-rho de M4c en la mitad de distritos
   con ALTA tasa de geocodificación real vs la mitad BAJA (real_coord_ratio).
4. Variante per cápita: rho intra de pred/pop vs target/pop (¿el ranking
   sobrevive al quitar el volumen poblacional?).

Uso: python3 scripts/eval_metric_robustness.py
Salida: analysis/etapa2_metric_robustness.md
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
from eval_modality_ladder import CATS, FEATURE_SETS, ORACLE_FILE, available, load_panel  # noqa: E402

OUT_MD = ROOT / "analysis/etapa2_metric_robustness.md"
STEPS = ["M1_demografia", "M1b_manzana", "M2_osm", "M4c_transporte"]
TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023
MIN_CELLS = 5
B = 2000
RNG = np.random.default_rng(42)

LOGGER = logging.getLogger("eval_metric_robustness")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fit_predict(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor
    parts = []
    for cat in CATS:
        d = panel[panel["crime_cat"] == cat]
        tr = d[d["year"].isin(TRAIN_YEARS)]
        te = d[d["year"] == TEST_YEAR].copy()
        med = tr[feats].median(numeric_only=True)
        m = HistGradientBoostingRegressor(
            max_iter=140, max_leaf_nodes=15, min_samples_leaf=20,
            learning_rate=0.06, l2_regularization=0.05, random_state=42,
        )
        m.fit(tr[feats].fillna(med).to_numpy(), np.log1p(tr["target"].to_numpy(float)))
        te["pred"] = np.expm1(np.maximum(m.predict(te[feats].fillna(med).to_numpy()), 0.0))
        parts.append(te[["h3_index", "ubigeo", "crime_cat", "target", "pred", "population"]])
    return pd.concat(parts, ignore_index=True)


def per_district_stats(te: pd.DataFrame, per_capita: bool = False) -> pd.DataFrame:
    """Por (cat, distrito): rho intra y captura@10% (share de eventos en top-10% pred)."""
    rows = []
    for (cat, ub), g in te.groupby(["crime_cat", "ubigeo"]):
        if len(g) < MIN_CELLS or g["target"].nunique() < 2 or g["pred"].nunique() < 2:
            continue
        t, p = g["target"].to_numpy(float), g["pred"].to_numpy(float)
        if per_capita:
            pop = g["population"].to_numpy(float)
            ok = pop > 0
            if ok.sum() < MIN_CELLS:
                continue
            t, p = t[ok] / pop[ok], p[ok] / pop[ok]
            if len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
                continue
        rho, _ = spearmanr(p, t)
        k = max(1, int(round(0.10 * len(t))))
        top = np.argsort(p)[-k:]
        cap = float(t[top].sum() / t.sum()) if t.sum() > 0 else np.nan
        rows.append({"crime_cat": cat, "ubigeo": ub, "rho": rho, "captura10": cap})
    return pd.DataFrame(rows)


def macro_from_stats(st: pd.DataFrame, col: str) -> float:
    return float(st.groupby("crime_cat")[col].mean().mean())


def boot_macro(st: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Bootstrap sobre distritos dentro de cada categoría; CI del macro."""
    by_cat = {c: g[col].dropna().to_numpy() for c, g in st.groupby("crime_cat")}
    draws = np.empty(B)
    for b in range(B):
        means = [RNG.choice(v, size=len(v), replace=True).mean() for v in by_cat.values() if len(v)]
        draws[b] = np.mean(means)
    return macro_from_stats(st, col), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_paired_delta(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """CI pareado del delta macro (mismos distritos resampleados en ambos modelos)."""
    m = a.merge(b, on=["crime_cat", "ubigeo"], suffixes=("_a", "_b"))
    by_cat = {c: g[[f"{col}_a", f"{col}_b"]].dropna().to_numpy() for c, g in m.groupby("crime_cat")}
    draws = np.empty(B)
    for i in range(B):
        deltas = []
        for v in by_cat.values():
            if not len(v):
                continue
            idx = RNG.integers(0, len(v), size=len(v))
            deltas.append(v[idx, 1].mean() - v[idx, 0].mean())
        draws[i] = np.mean(deltas)
    point = float(np.mean([v[:, 1].mean() - v[:, 0].mean() for v in by_cat.values() if len(v)]))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    panel, _ = load_panel()
    stats, preds = {}, {}
    for step in STEPS:
        feats = available(FEATURE_SETS[step], panel)
        LOGGER.info("%s: %d features", step, len(feats))
        preds[step] = fit_predict(panel, feats)
        stats[step] = per_district_stats(preds[step])

    lines = [
        "# Robustez de métricas intra-distritales (slr-0vh)",
        "",
        f"**Protocolo:** train 2019-2022, test {TEST_YEAR}, oráculo geocodificado. "
        f"Bootstrap B={B} sobre **distritos** (la unidad de réplica de la métrica intra), "
        "percentiles 2.5/97.5. HGB con params default del ladder.",
        "",
        "## 1. CI por escalón (macro sobre categorías)",
        "",
        "| escalón | macro ρ [CI 95%] | captura@10% [CI 95%] |",
        "|:--|:--|:--|",
    ]
    for step in STEPS:
        r, rl, rh = boot_macro(stats[step], "rho")
        c, cl, ch = boot_macro(stats[step], "captura10")
        lines.append(f"| {step} | {r:.3f} [{rl:.3f}, {rh:.3f}] | {c:.3f} [{cl:.3f}, {ch:.3f}] |")

    lines += ["", "## 2. Deltas PAREADOS entre escalones (¿significativos?)", "",
              "| comparación | Δ macro ρ [CI 95%] | ¿CI excluye 0? |", "|:--|:--|:--|"]
    for a, b in [("M1_demografia", "M1b_manzana"), ("M1b_manzana", "M2_osm"),
                 ("M2_osm", "M4c_transporte"), ("M1b_manzana", "M4c_transporte")]:
        d, dl, dh = boot_paired_delta(stats[a], stats[b], "rho")
        sig = "SÍ" if (dl > 0 or dh < 0) else "no"
        lines.append(f"| {b} − {a} | {d:+.4f} [{dl:+.4f}, {dh:+.4f}] | {sig} |")

    lines += ["", "## 3. Por categoría (M4c): ρ vs captura@10% con CI", "",
              "| categoría | ρ [CI] | captura@10% [CI] | n distritos |", "|:--|:--|:--|:--|"]
    st4 = stats["M4c_transporte"]
    for cat in CATS:
        g = st4[st4["crime_cat"] == cat]
        rv, cv = g["rho"].dropna().to_numpy(), g["captura10"].dropna().to_numpy()
        rb = np.array([RNG.choice(rv, len(rv)).mean() for _ in range(B)])
        cb = np.array([RNG.choice(cv, len(cv)).mean() for _ in range(B)])
        lines.append(
            f"| {cat} | {rv.mean():.3f} [{np.percentile(rb,2.5):.3f}, {np.percentile(rb,97.5):.3f}]"
            f" | {cv.mean():.3f} [{np.percentile(cb,2.5):.3f}, {np.percentile(cb,97.5):.3f}] | {len(rv)} |")

    # 4. sensibilidad geocodificación
    oracle = pd.read_parquet(ORACLE_FILE, columns=["ubigeo", "year", "real_coord_ratio"])
    oracle["ubigeo"] = oracle["ubigeo"].astype(str).str.zfill(6)
    ratio = (oracle[oracle["year"] == TEST_YEAR].groupby("ubigeo")["real_coord_ratio"].mean())
    med = ratio.median()
    hi_dists, lo_dists = set(ratio[ratio >= med].index), set(ratio[ratio < med].index)
    hi = st4[st4["ubigeo"].isin(hi_dists)]
    lo = st4[st4["ubigeo"].isin(lo_dists)]
    rh_, rhl, rhh = boot_macro(hi, "rho")
    rl_, rll, rlh = boot_macro(lo, "rho")
    lines += ["", "## 4. Sensibilidad al sesgo del oráculo (M4c)", "",
              f"Split por mediana de `real_coord_ratio` distrital ({med:.3f}).", "",
              "| mitad | macro ρ [CI 95%] | n distritos |", "|:--|:--|:--|",
              f"| ALTA geocodificación | {rh_:.3f} [{rhl:.3f}, {rhh:.3f}] | {hi.ubigeo.nunique()} |",
              f"| BAJA geocodificación | {rl_:.3f} [{rll:.3f}, {rlh:.3f}] | {lo.ubigeo.nunique()} |"]

    # 5. per capita
    st_pc = per_district_stats(preds["M4c_transporte"], per_capita=True)
    p, pl, ph = boot_macro(st_pc, "rho")
    lines += ["", "## 5. Variante per cápita (M4c)", "",
              f"ρ intra de pred/pop vs target/pop: **{p:.3f}** [{pl:.3f}, {ph:.3f}] "
              f"(conteos: {macro_from_stats(st4, 'rho'):.3f}). Si cae mucho, parte del ranking",
              "de conteos era volumen poblacional, no riesgo."]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGuardado → {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
