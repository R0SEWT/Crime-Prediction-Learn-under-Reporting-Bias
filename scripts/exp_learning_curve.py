#!/usr/bin/env python3
"""
Curva de aprendizaje local vs transferencia (slr-b4b, paper SIMBig).

¿Cuántos años de historia local necesita Arequipa para superar al modelo
transferido de Lima? Entrena el modelo LOCAL con ventanas crecientes
{2022, 2021-22, 2020-22, 2019-22} y compara cada punto contra el TRANSFER
fijo (Lima 2019-2022), ambos evaluados en Arequipa 2023 contra su oráculo,
con bootstrap pareado sobre distritos.

Requiere los artefactos de transfer_arequipa.py (oracle + features AQP).
Uso: python3 scripts/exp_learning_curve.py [--city arequipa]
Salida: analysis/etapa2_learning_curve.md
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
from eval_modality_ladder import CATS, load_panel, available  # noqa: E402
from transfer_arequipa import CITIES, TRANSFER_FEATS, TRAIN_YEARS, TEST_YEAR, MIN_CELLS  # noqa: E402

OUT_MD = ROOT / "analysis/etapa2_learning_curve.md"
B = 2000
RNG = np.random.default_rng(42)
LOGGER = logging.getLogger("exp_learning_curve")


def build_city_panel(city: str) -> pd.DataFrame:
    prefix = "aqp" if city == "arequipa" else city
    out_dir = ROOT / f"data/silver/transfer_{city}"
    oracle = pd.read_parquet(out_dir / f"{prefix}_oracle.parquet")
    feats = pd.read_parquet(out_dir / f"{prefix}_features.parquet")
    cells = feats["h3_index"].tolist()
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
    panel = base.merge(feats, on="h3_index", how="left", suffixes=("", "_mzn"))
    if "ubigeo_mzn" in panel.columns:
        panel["ubigeo"] = panel["ubigeo"].fillna(panel["ubigeo_mzn"].astype(str).str.zfill(6))
        panel = panel.drop(columns=["ubigeo_mzn"])
    panel = panel[panel["ubigeo"].notna() | (panel["population"].fillna(0) > 0)].copy()
    panel["ubigeo"] = panel["ubigeo"].fillna("XXXXXX")
    return panel


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    med = tr[feats].median(numeric_only=True)
    m = HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=15,
                                      min_samples_leaf=20, learning_rate=0.06,
                                      l2_regularization=0.05, random_state=42)
    m.fit(tr[feats].fillna(med).to_numpy(), np.log1p(tr["target"].to_numpy(float)))
    return np.expm1(np.maximum(m.predict(te[feats].fillna(med).to_numpy()), 0.0))


def intra_stats(te: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (cat, ub), g in te.groupby(["crime_cat", "ubigeo"]):
        if len(g) < MIN_CELLS or g["target"].nunique() < 2 or g[pred_col].nunique() < 2:
            continue
        rho, _ = spearmanr(g[pred_col], g["target"])
        rows.append({"crime_cat": cat, "ubigeo": ub, "rho": rho})
    return pd.DataFrame(rows)


def macro(st: pd.DataFrame) -> float:
    return float(st.groupby("crime_cat").rho.mean().mean()) if len(st) else float("nan")


def boot_delta(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, float]:
    m = a.merge(b, on=["crime_cat", "ubigeo"], suffixes=("_a", "_b"))
    by_cat = {c: g[["rho_a", "rho_b"]].dropna().to_numpy() for c, g in m.groupby("crime_cat")}
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=sorted(CITIES), default="arequipa")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    label = CITIES[args.city]["label"]

    city = build_city_panel(args.city)
    lima, _ = load_panel()
    feats = [f for f in TRANSFER_FEATS if f in lima.columns and f in city.columns]
    LOGGER.info("%s: %d features transferibles", label, len(feats))

    test = city[city["year"] == TEST_YEAR].copy()
    windows = {"1 año (2022)": [2022], "2 años (2021-22)": [2021, 2022],
               "3 años (2020-22)": [2020, 2021, 2022], "4 años (2019-22)": [2019, 2020, 2021, 2022]}

    # transfer fijo: Lima 2019-2022
    preds = {}
    for cat in CATS:
        tr = lima[(lima["crime_cat"] == cat) & lima["year"].isin(TRAIN_YEARS)]
        te = test[test["crime_cat"] == cat].copy()
        te["pred_transfer"] = fit_predict(tr, te, feats)
        for wlabel, yrs in windows.items():
            trl = city[(city["crime_cat"] == cat) & city["year"].isin(yrs)]
            te[f"pred_{wlabel}"] = fit_predict(trl, te, feats)
        preds[cat] = te
    allte = pd.concat(preds.values(), ignore_index=True)

    st_transfer = intra_stats(allte, "pred_transfer")
    lines = [
        f"# Curva de aprendizaje local vs transferencia — {label} (slr-b4b)",
        "",
        f"**Pregunta:** ¿cuántos años de historia local necesita {label} para superar al "
        f"modelo transferido de Lima (entrenado 2019-2022, sin datos locales)? Test {TEST_YEAR}, "
        f"Spearman intra-distrital macro, bootstrap pareado sobre distritos (B={B}).",
        "",
        f"Transfer fijo (Lima→{label}): **{macro(st_transfer):.3f}**",
        "",
        "| historia local | macro ρ | Δ (local − transfer) [CI 95%] | ¿CI excluye 0? |",
        "|:--|--:|:--|:--|",
    ]
    for wlabel in windows:
        st_l = intra_stats(allte, f"pred_{wlabel}")
        d, dl, dh = boot_delta(st_transfer, st_l)
        sig = "SÍ" if (dl > 0 or dh < 0) else "no"
        lines.append(f"| {wlabel} | {macro(st_l):.3f} | {d:+.4f} [{dl:+.4f}, {dh:+.4f}] | {sig} |")

    # detalle raras vs densas con la ventana completa
    st_full = intra_stats(allte, "pred_4 años (2019-22)")
    lines += ["", "## Por categoría (transfer vs local 4 años)", "",
              "| categoría | transfer | local 4 años |", "|:--|--:|--:|"]
    for cat in CATS:
        t = st_transfer[st_transfer.crime_cat == cat].rho.mean()
        l = st_full[st_full.crime_cat == cat].rho.mean()
        lines.append(f"| {cat} | {t:.3f} | {l:.3f} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGuardado → {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
