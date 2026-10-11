#!/usr/bin/env python3
"""
Eval HONESTO de las predicciones STGNN contra el oraculo geocodificado.

Compara la fusion cross-modal (--fusion attention) contra el baseline concat en
Spearman intra-distrital vs el oraculo observado geocodificado (NO contra el
target latente redistribuido, que da lecturas circulares; ver
`analysis/stgnn_metrics.md` y `ch5-resultados.typ` §Q1).

Evalua todas las variantes de prediccion presentes en disco (generadas por
train_stgnn.py con --target {latente,hybrid} x --fusion {concat,attention}):
  data/silver/predictions/stgnn_latente_h3[_hybrid][_attention].parquet
y el oraculo:
  data/silver/h3_features/h3_observed_geocoded.parquet

Uso:
    python3 scripts/eval_fusion_oracle.py
    python3 scripts/eval_fusion_oracle.py --year 2023
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ORACLE_FILE = ROOT / "data/silver/h3_features/h3_observed_geocoded.parquet"
PRED_DIR = ROOT / "data/silver/predictions"
VARIANTS = {
    "concat": "stgnn_latente_h3.parquet",
    "attention": "stgnn_latente_h3_attention.parquet",
    "hyb_concat": "stgnn_latente_h3_hybrid.parquet",
    "hyb_attn": "stgnn_latente_h3_hybrid_attention.parquet",
}
CATS = ["robo_hurto_callejero", "extorsion", "estafa", "violencia_familiar_sexual", "secuestro"]
MIN_CELLS = 5


def intra_spearman(pred_path: Path, oracle: pd.DataFrame) -> tuple[dict, float]:
    pred = pd.read_parquet(pred_path)
    pred = pred[pred["year"] == oracle.attrs["year"]].copy()
    pred["h3_index"] = pred["h3_index"].astype(str)
    per_cat = {}
    for cat in CATS:
        pcol = pred[["h3_index", f"pred_{cat}"]].rename(columns={f"pred_{cat}": "pred"})
        oc = oracle[oracle["crime_cat"] == cat][["h3_index", "ubigeo", "obs_geo_count"]]
        d = oc.merge(pcol, on="h3_index", how="left")
        d["pred"] = d["pred"].fillna(0.0)
        rhos = []
        for _, g in d.groupby("ubigeo"):
            if len(g) < MIN_CELLS or g["obs_geo_count"].nunique() < 2 or g["pred"].nunique() < 2:
                continue
            r, _ = spearmanr(g["pred"], g["obs_geo_count"])
            if not np.isnan(r):
                rhos.append(r)
        per_cat[cat] = (float(np.mean(rhos)) if rhos else float("nan"), len(rhos))
    macro = float(np.nanmean([v[0] for v in per_cat.values()]))
    return per_cat, macro


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()

    oracle = pd.read_parquet(ORACLE_FILE)
    oracle = oracle[oracle["year"] == args.year].copy()
    oracle["h3_index"] = oracle["h3_index"].astype(str)
    oracle["ubigeo"] = oracle["ubigeo"].astype(str).str.zfill(6)
    oracle.attrs["year"] = args.year

    res = {}
    for label, fname in VARIANTS.items():
        path = PRED_DIR / fname
        if not path.exists():
            print(f"(skip {label}: falta {fname})")
            continue
        res[label] = intra_spearman(path, oracle)
    if not res:
        print("No hay predicciones — corre train_stgnn.py primero")
        return 1

    labels = list(res)
    print(f"\n=== STGNN vs ORACULO geocodificado — Spearman intra-distrital ({args.year}) ===")
    print("cat".ljust(28) + "".join(lb.rjust(11) for lb in labels))
    for cat in CATS:
        vals = "".join(f"{res[lb][0][cat][0]:11.4f}" for lb in labels)
        print(cat.ljust(28) + vals)
    print("-" * (28 + 11 * len(labels)))
    print("MACRO".ljust(28) + "".join(f"{res[lb][1]:11.4f}" for lb in labels))
    print("\nRefs: prior_historical 0.394 · ladder tabular M4c 0.4636 · "
          "attention-circular (Lightning) 0.2079")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
