#!/usr/bin/env python3
"""
Degradación SELECTIVA de geocodificación: mecanismo de fallo + curva de evaluabilidad (slr-7lsf).

Reemplaza el binomial thinning UNIFORME de exp_geocoding_degradation.py (slr-dj7) por una
degradación selectiva anclada al mecanismo empírico de fallo observado en Lima. Ancla:
limitación 4 del SIMBig ('Binomial thinning is an optimistic bound — real geocoding loss
is selective').

FASE 1 — mecanismo de fallo (universo: denuncias limpias solo_denuncia=1, 5 categorías,
2019-2023, LIMA.parquet read-only). Dos mecanismos de pérdida REPORTADOS POR SEPARADO:
  (a) SIN COORDENADA  : estado_coord != 'CON COORDENADA'
  (b) artefacto COMISARÍA: CON COORDENADA pero en pile-up de coordenada exacta
      (> --max-per-coord denuncias en el mismo punto = geocodificada a la comisaría,
      no al hecho; mismo umbral que el anti-centroide del oráculo,
      scripts/build_observed_h3_points.py)
Éxito limpio = CON COORDENADA ∧ bbox Lima ∧ no pile-up (lo que entra al oráculo, módulo
el filtro a celdas del feature matrix, que aquí se omite por costo y se declara).
Se caracteriza cada fallo por categoría, año, canal (comisaria_registro VIRTUAL), distrito
(correlación con pobreza INEI 2018) y tipo de vía, con IC de Wilson y regresión logística
sin regularizar (SE de Wald vía información de Fisher; sin cluster por distrito — se declara).

FASE 2 — curva de evaluabilidad selectiva. Igual protocolo que exp_geocoding_degradation.py
(re-entrenar M4c sobre targets degradados, medible = rho intra vs oráculo degradado,
real = vs oráculo completo, persistencia degradada), pero el thinning es por celda:
p_celda = min(1, s · q_dc), con q_dc = propensión empírica de éxito limpio del
distrito × categoría de la celda (con shrinkage hacia la tasa de la categoría), y s
resuelto por bisección para que la retención esperada global (ponderada por conteos)
alcance cada nivel objetivo. Supuesto declarado: la pérdida MARGINAL sigue el gradiente
transversal de fallo observado (los distritos/categorías que hoy geocodifican peor
pierden primero). Se compara contra thinning uniforme en los MISMOS niveles y seeds.

Uso:
    python3 scripts/eval_selective_degradation.py             # fase 1 + fase 2
    python3 scripts/eval_selective_degradation.py --phase 1   # solo mecanismo
    python3 scripts/eval_selective_degradation.py --seeds 3   # grid rápido

Salida: analysis/selective_geocoding_degradation.md
        data/silver/predictions/selective_geocoding_degradation.csv (per-seed, gitignored)

Entrada: el volcado crudo de denuncias de Lima (export MININTER/SIDPOL, no redistribuible)
se lee de la variable de entorno INFELIX_LIMA_PARQUET o, si no está definida, de
data/raw/external/LIMA.parquet (relativo a la raíz del repo). `--source` lo sobrescribe.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_modality_ladder import CATS, FEATURE_SETS, available, load_panel  # noqa: E402

DEFAULT_SOURCE = Path(os.environ.get("INFELIX_LIMA_PARQUET",
                                    str(ROOT / "data/raw/external/LIMA.parquet")))
POBREZA_FILE = ROOT / "data/raw/inei/pobreza_distrital_2018.csv"
OUT_MD = ROOT / "analysis/selective_geocoding_degradation.md"
OUT_CSV = ROOT / "data/silver/predictions/selective_geocoding_degradation.csv"

BASE_RATE = 0.834          # convención del experimento original (slr-dj7); medido aquí: ver md
RATES = [0.834, 0.80, 0.60, 0.50, 0.40, 0.249, 0.10]   # 0.249 = Trujillo exacto; 0.50 = fila de Tabla 4 SIMBig
N_SEEDS = 10
TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023
MIN_CELLS = 5
SHRINK_M = 50              # pseudo-conteos del shrinkage de q_dc hacia la tasa de categoría
YEAR_MIN, YEAR_MAX = 2019, 2023
LAT_MIN, LAT_MAX = -12.45, -11.55   # bbox Lima Metropolitana (idéntico al oráculo)
LNG_MIN, LNG_MAX = -77.30, -76.60

LOGGER = logging.getLogger("eval_selective_degradation")

SOURCE_COLS = [
    "solo_denuncia", "estado_coord", "id_dist_hecho", "distrito_hecho",
    "lat_hecho", "long_hecho", "año_hecho", "tipo_hecho", "subtipo_hecho",
    "modalidad_hecho", "tipo_via_hecho", "comisaria_registro",
]
NAMED_VIAS = ["Avenida", "Jiron", "Calle", "Pasaje", "Urbanización", "AA.HH",
              "Panamericana", "Parque", "Malecón", "Carretera", "Centro Poblado",
              "Upis", "Comunidad", "Anexo", "Caserio"]


# ────────────────────────── FASE 1: mecanismo de fallo ──────────────────────────

def map_category(tipo: pd.Series, subtipo: pd.Series, modalidad: pd.Series) -> pd.Series:
    """Crosswalk idéntico a build_observed_h3_points.map_category."""
    t = tipo.fillna("").str.upper()
    s = subtipo.fillna("").str.upper()
    m = modalidad.fillna("").str.upper()
    cat = pd.Series(np.nan, index=tipo.index, dtype=object)
    patrimonio = t == "PATRIMONIO (DELITO)"
    cat[patrimonio & s.isin(["ROBO", "HURTO"])] = "robo_hurto_callejero"
    cat[patrimonio & (s == "ESTAFA Y OTRAS DEFRAUDACIONES")] = "estafa"
    cat[patrimonio & (s == "EXTORSION")] = "extorsion"
    ley_mujer = t == "LEY DE VIOLENCIA CONTRA LA MUJER Y GRUPOS VULNERABLES"
    viol_sexual = (t == "LIBERTAD (DELITO)") & (s == "VIOLACION DE LA LIBERTAD SEXUAL")
    cat[ley_mujer | viol_sexual] = "violencia_familiar_sexual"
    cat[m.str.contains("SECUESTRO", na=False)] = "secuestro"
    return cat


def load_universe(source: Path, max_per_coord: int) -> tuple[pd.DataFrame, pd.Series]:
    """Universo de denuncias limpias (5 cats, 2019-2023) con flags de fallo + nombres de distrito."""
    df = pd.read_parquet(source, columns=SOURCE_COLS)
    df = df[df["solo_denuncia"] == 1].copy()
    df["year"] = pd.to_numeric(df["año_hecho"], errors="coerce")
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df["year"] = df["year"].astype(int)
    df["crime_cat"] = map_category(df["tipo_hecho"], df["subtipo_hecho"], df["modalidad_hecho"])
    df = df[df["crime_cat"].notna()].copy()
    df["ubigeo"] = df["id_dist_hecho"].astype(str).str.zfill(6)

    df["con_coord"] = df["estado_coord"].astype(str).str.upper() == "CON COORDENADA"
    df["virtual"] = df["comisaria_registro"].astype(str).str.upper().str.contains("VIRTUAL")
    df["named_via"] = df["tipo_via_hecho"].isin(NAMED_VIAS)

    # artefacto comisaría: pile-up en coordenada EXACTA dentro del bbox
    lat = pd.to_numeric(df["lat_hecho"], errors="coerce")
    lng = pd.to_numeric(df["long_hecho"], errors="coerce")
    inbox = df["con_coord"] & lat.between(LAT_MIN, LAT_MAX) & lng.between(LNG_MIN, LNG_MAX)
    geo = pd.DataFrame({"lat": lat[inbox], "lng": lng[inbox]})
    pile = geo.groupby(["lat", "lng"]).size()
    bad = set(pile[pile > max_per_coord].index)
    df["centroid"] = False
    df.loc[geo.index, "centroid"] = [k in bad for k in zip(geo["lat"], geo["lng"])]
    df["inbox"] = inbox
    df["clean"] = df["inbox"] & ~df["centroid"]
    LOGGER.info("universo: %d denuncias | con_coord=%.1f%% | pile-ups=%d coords, %d denuncias | clean=%.1f%%",
                len(df), 100 * df["con_coord"].mean(), len(bad),
                int(df["centroid"].sum()), 100 * df["clean"].mean())
    names = df.groupby("ubigeo")["distrito_hecho"].first()
    return df[["year", "crime_cat", "ubigeo", "con_coord", "virtual", "named_via",
               "centroid", "inbox", "clean"]], names


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def rate_table(df: pd.DataFrame, by: str, outcome: str, labels=None) -> list[str]:
    """Tabla markdown: tasa del outcome por nivel de `by`, con IC 95% Wilson."""
    g = df.groupby(by)[outcome].agg(["sum", "size"])
    if labels is not None:
        g = g.reindex(labels)
    lines = [f"| {by} | n | tasa {outcome} | IC 95% |", "|:--|--:|--:|:--|"]
    for idx, r in g.iterrows():
        lo, hi = wilson_ci(r["sum"], r["size"])
        lines.append(f"| {idx} | {int(r['size']):,} | {r['sum'] / r['size']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    return lines


def fit_logit(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    """Logística sin regularizar + SE Wald (Fisher). Filas markdown con OR e IC 95%."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=3000)  # sin regularizar
    lr.fit(X, y)
    beta = np.r_[lr.intercept_, lr.coef_.ravel()]
    Xd = np.c_[np.ones(len(X), dtype=np.float64), X.astype(np.float64)]
    p = 1.0 / (1.0 + np.exp(-(Xd @ beta)))
    w = p * (1 - p)
    cov = np.linalg.inv(Xd.T @ (Xd * w[:, None]))
    se = np.sqrt(np.diag(cov))
    lines = ["| covariable | OR | IC 95% |", "|:--|--:|:--|"]
    for nm, b, s in zip(["(intercepto)"] + names, beta, se):
        lines.append(f"| {nm} | {np.exp(b):.3f} | [{np.exp(b - 1.96 * s):.3f}, {np.exp(b + 1.96 * s):.3f}] |")
    return lines


def build_design(d: pd.DataFrame, pov_z: pd.Series, extra: list[str]) -> tuple[np.ndarray, list[str]]:
    """Matriz de diseño: dummies cat (ref robo) + año (ref 2019) + extras + pobreza_z."""
    parts, names = [], []
    for cat in [c for c in CATS if c != "robo_hurto_callejero"]:
        parts.append((d["crime_cat"] == cat).to_numpy(float)); names.append(f"cat={cat}")
    for yr in range(YEAR_MIN + 1, YEAR_MAX + 1):
        parts.append((d["year"] == yr).to_numpy(float)); names.append(f"año={yr}")
    for col in extra:
        parts.append(d[col].to_numpy(float)); names.append(col)
    parts.append(d["ubigeo"].map(pov_z).fillna(0.0).to_numpy(float)); names.append("pobreza_z")
    return np.column_stack(parts), names


def phase1(df: pd.DataFrame, matrix_ubigeos: set[str],
           dist_names: pd.Series) -> tuple[list[str], pd.DataFrame, dict]:
    """Caracteriza los dos mecanismos de fallo.

    Devuelve (líneas md, tabla distrital, hechos portantes para el registro canónico)."""
    pov = pd.read_csv(POBREZA_FILE, dtype={"ubigeo": str})
    pov_map = pov.set_index("ubigeo")["poverty_rate"]
    pov_z = (pov_map - pov_map.mean()) / pov_map.std()

    df = df.copy()
    df["fail_nocoord"] = ~df["con_coord"]

    lines: list[str] = []
    lines += ["## Fase 1 — mecanismo empírico de fallo", "",
              f"Universo: {len(df):,} denuncias limpias (solo_denuncia=1, 5 categorías, "
              f"{YEAR_MIN}-{YEAR_MAX}). Dos mecanismos de pérdida, reportados por separado:",
              "",
              f"- **(a) SIN COORDENADA**: {df['fail_nocoord'].mean():.1%} del universo.",
              f"- **(b) artefacto comisaría** (pile-up en coordenada exacta, umbral >{ARGS.max_per_coord}): "
              f"{df.loc[df['con_coord'], 'centroid'].mean():.1%} de las CON COORDENADA "
              f"({int(df['centroid'].sum()):,} denuncias en {df[df['centroid']].groupby(['ubigeo']).ngroups} distritos).",
              f"- **Éxito limpio** (lo que entra al oráculo, módulo filtro a celdas del matrix): "
              f"{df['clean'].mean():.1%}.", ""]

    lines += ["### (a) SIN COORDENADA — tasas por covariable", ""]
    for by, labels in [("crime_cat", CATS), ("year", list(range(YEAR_MIN, YEAR_MAX + 1))),
                       ("virtual", [False, True]), ("named_via", [False, True])]:
        lines += rate_table(df, by, "fail_nocoord", labels) + [""]

    lines += ["### (b) artefacto comisaría — tasas por covariable (entre las CON COORDENADA en bbox)", ""]
    geo = df[df["inbox"] | df["centroid"]]
    for by, labels in [("crime_cat", CATS), ("year", list(range(YEAR_MIN, YEAR_MAX + 1))),
                       ("virtual", [False, True])]:
        lines += rate_table(geo, by, "centroid", labels) + [""]

    # gradiente distrital y pobreza (solo distritos del feature matrix, evita el
    # artefacto de bbox: p.ej. Pucusana cae fuera del bbox y su clean rate sería 0)
    dm = df[df["ubigeo"].isin(matrix_ubigeos)]
    dist = dm.groupby("ubigeo").agg(
        n=("clean", "size"), nocoord=("fail_nocoord", "mean"), clean=("clean", "mean"))
    geo_m = geo[geo["ubigeo"].isin(matrix_ubigeos)]
    dist = dist.join(geo_m.groupby("ubigeo")["centroid"].mean().rename("centroid"))
    dist = dist.join(pov_map.rename("pobreza")).reset_index()
    big = dist[dist["n"] >= 500].dropna(subset=["pobreza"])
    r_no = spearmanr(big["nocoord"], big["pobreza"])
    r_ce = spearmanr(big["centroid"], big["pobreza"])
    r_cl = spearmanr(big["clean"], big["pobreza"])
    lines += ["### Gradiente distrital (distritos del feature matrix, n≥500) vs pobreza INEI 2018", "",
              f"- Spearman tasa SIN COORDENADA vs pobreza: **ρ={r_no.statistic:.2f}** (p={r_no.pvalue:.1e}, n={len(big)})",
              f"- Spearman tasa artefacto comisaría vs pobreza: **ρ={r_ce.statistic:.2f}** (p={r_ce.pvalue:.1e})",
              f"- Spearman tasa de éxito limpio vs pobreza: **ρ={r_cl.statistic:.2f}** (p={r_cl.pvalue:.1e})", "",
              "| | éxito limpio | sin coord | artefacto | pobreza |", "|:--|--:|--:|--:|--:|"]
    ext = pd.concat([big.nsmallest(5, "clean").assign(_g="peores 5"),
                     big.nlargest(5, "clean").assign(_g="mejores 5")])
    for _, r in ext.iterrows():
        nm = str(dist_names.get(r["ubigeo"], "")).title()
        lines.append(f"| {nm} {r['ubigeo']} ({r['_g']}) | {r['clean']:.3f} | {r['nocoord']:.3f} "
                     f"| {r['centroid']:.3f} | {r['pobreza']:.1f}% |")
    lines.append("")

    # regresiones logísticas
    lines += ["### Regresión logística — P(SIN COORDENADA)",
              "(ref: robo_hurto_callejero, año 2019; SE Wald sin cluster por distrito — "
              "con n=1.1M los IC son optimistas, se leen como descriptivos)", ""]
    X, names = build_design(df, pov_z, ["virtual", "named_via"])
    lines += fit_logit(X, df["fail_nocoord"].to_numpy(), names) + [""]

    lines += ["### Regresión logística — P(artefacto comisaría | CON COORDENADA en bbox)", ""]
    Xg, names_g = build_design(geo, pov_z, ["virtual"])
    lines += fit_logit(Xg, geo["centroid"].to_numpy(), names_g) + [""]

    facts = {
        "n_universe": float(len(df)),
        "rho_nocoord_pov": float(r_no.statistic),
        "rho_centroid_pov": float(r_ce.statistic),
        "rho_clean_pov": float(r_cl.statistic),
        "n_districts_gradient": float(len(big)),
    }
    for cat, v in df.groupby("crime_cat")["fail_nocoord"].mean().items():
        facts[f"nocoord_pct.{cat}"] = 100.0 * float(v)
    return lines, dist, facts


def propensities(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """q_dc = P(éxito limpio | distrito × categoría) con shrinkage hacia la categoría."""
    cat_rate = df.groupby("crime_cat")["clean"].mean()
    g = df.groupby(["ubigeo", "crime_cat"])["clean"].agg(["sum", "size"])
    prior = g.index.get_level_values("crime_cat").map(cat_rate).to_numpy()
    q = (g["sum"].to_numpy() + SHRINK_M * prior) / (g["size"].to_numpy() + SHRINK_M)
    return pd.Series(q, index=g.index, name="q_dc"), cat_rate


# ────────────────────────── FASE 2: curva selectiva vs uniforme ──────────────────────────

def intra_macro(df: pd.DataFrame, pred_col: str, target_col: str) -> tuple[float, float]:
    """(macro-rho intra-distrital, nº medio de distritos evaluables por categoría).
    Idéntico a exp_geocoding_degradation.intra_macro + cobertura."""
    per_cat, cover = [], []
    for _, dcat in df.groupby("crime_cat"):
        rhos = []
        for _, g in dcat.groupby("ubigeo"):
            if len(g) < MIN_CELLS or g[target_col].nunique() < 2 or g[pred_col].nunique() < 2:
                continue
            r, _ = spearmanr(g[pred_col], g[target_col])
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            per_cat.append(np.mean(rhos))
            cover.append(len(rhos))
    return (float(np.mean(per_cat)) if per_cat else float("nan"),
            float(np.mean(cover)) if cover else 0.0)


def solve_scale(q: np.ndarray, w: np.ndarray, target_ret: float) -> float:
    """Bisección: s tal que Σ w·min(1, s·q) / Σ w = target_ret."""
    def f(s: float) -> float:
        return float(np.sum(w * np.minimum(1.0, s * q)) / np.sum(w)) - target_ret
    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        hi *= 2
        if hi > 1e6:
            break
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def phase2(q_dc: pd.Series, cat_rate: pd.Series, n_seeds: int) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor

    panel, _ = load_panel()
    feats = available(FEATURE_SETS["M4c_transporte"], panel)
    panel = panel[["h3_index", "ubigeo", "year", "crime_cat", "target", "persist"] + feats].copy()
    LOGGER.info("panel: %d filas, %d features M4c", len(panel), len(feats))

    # propensión por fila (celda): q del distrito×categoría; fallback = tasa de categoría
    key = pd.MultiIndex.from_frame(panel[["ubigeo", "crime_cat"]])
    q_row = pd.Series(q_dc.reindex(key).to_numpy(), index=panel.index)
    q_row = q_row.fillna(panel["crime_cat"].map(cat_rate))
    q_arr = q_row.to_numpy(float)
    w_arr = panel["target"].to_numpy(float)
    LOGGER.info("q_dc por fila: min=%.3f p50=%.3f max=%.3f (fallback en %.1f%% de filas)",
                q_arr.min(), np.median(q_arr), q_arr.max(),
                100 * q_dc.reindex(key).isna().mean())

    rows = []
    for rate in RATES:
        ret = min(1.0, rate / BASE_RATE)
        p_by_mech = {"uniforme": np.full(len(panel), ret)}
        if ret < 1.0:
            s = solve_scale(q_arr, w_arr, ret)
            p_sel = np.minimum(1.0, s * q_arr)
            p_by_mech["selectivo"] = p_sel
            LOGGER.info("rate=%.3f ret=%.3f | selectivo: s=%.3f p∈[%.3f, %.3f]",
                        rate, ret, s, p_sel.min(), p_sel.max())
        else:
            p_by_mech["selectivo"] = np.ones(len(panel))

        n_eff = 1 if ret >= 1.0 else n_seeds
        for mech, p_vec in p_by_mech.items():
            for seed in range(n_eff):
                rng = np.random.default_rng(int(rate * 1000) * 100 + seed * 7 + (0 if mech == "uniforme" else 1))
                d = panel.copy()
                if ret < 1.0:
                    d["target_thin"] = rng.binomial(d["target"].to_numpy(int), p_vec).astype(float)
                    d["persist_thin"] = rng.binomial(d["persist"].to_numpy(int), p_vec).astype(float)
                else:
                    d["target_thin"], d["persist_thin"] = d["target"], d["persist"]
                achieved = d["target_thin"].sum() / max(d["target"].sum(), 1e-9)

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

                medible, cov_med = intra_macro(te_all, "pred_feat", "target_thin")
                real, _ = intra_macro(te_all, "pred_feat", "target")
                persist_med, _ = intra_macro(te_all, "persist_thin", "target_thin")
                rows.append({"mechanism": mech, "rate": rate, "seed": seed,
                             "achieved_ret": achieved, "medible": medible, "real": real,
                             "persist": persist_med, "cover_dist": cov_med,
                             "feat_gana": medible > persist_med})
                if seed == 0:
                    LOGGER.info("%s rate=%.3f seed=0: medible=%.3f real=%.3f persist=%.3f cover=%.0f",
                                mech, rate, medible, real, persist_med, cov_med)

    return pd.DataFrame(rows)


def phase2_lines(res: pd.DataFrame, n_seeds: int) -> list[str]:
    """Sección markdown de la fase 2 a partir del per-seed (fresco o desde OUT_CSV)."""
    agg = (res.groupby(["mechanism", "rate"])
           .agg(medible=("medible", "mean"), medible_sd=("medible", "std"),
                real=("real", "mean"), real_sd=("real", "std"),
                persist=("persist", "mean"), cover=("cover_dist", "mean"),
                feat_gana=("feat_gana", "mean"))
           .reset_index().sort_values(["rate", "mechanism"], ascending=[False, False]))

    lines = ["## Fase 2 — curva de evaluabilidad: selectivo vs uniforme", "",
             f"Mismo protocolo que slr-dj7 (M4c re-entrenado sobre targets degradados, "
             f"{n_seeds} seeds por nivel, base {BASE_RATE:.1%}); 'selectivo' = retención por "
             f"celda ∝ propensión de éxito limpio de su distrito×categoría (shrinkage "
             f"m={SHRINK_M}), escalada por nivel; 'cobertura' = nº medio de distritos "
             f"evaluables por categoría en el rho medible.", "",
             "| tasa geocod. | mecanismo | ρ medible (±sd) | ρ real (±sd) | ρ persistencia "
             "| cobertura (distritos) | % seeds feat>persist |",
             "|--:|:--|:--|:--|--:|--:|--:|"]
    for _, r in agg.iterrows():
        sd_m = 0.0 if np.isnan(r["medible_sd"]) else r["medible_sd"]
        sd_r = 0.0 if np.isnan(r["real_sd"]) else r["real_sd"]
        lines.append(f"| {r['rate']:.1%} | {r['mechanism']} | {r['medible']:.3f} (±{sd_m:.3f}) "
                     f"| {r['real']:.3f} (±{sd_r:.3f}) | {r['persist']:.3f} "
                     f"| {r['cover']:.1f} | {r['feat_gana']:.0%} |")
    lines.append("")

    # tabla de brechas selectivo − uniforme
    piv = agg.pivot(index="rate", columns="mechanism", values=["medible", "real", "persist"])
    lines += ["### Brecha selectivo − uniforme (Δρ; negativo = el thinning uniforme era optimista)", "",
              "| tasa geocod. | Δ medible | Δ real | Δ persistencia |", "|--:|--:|--:|--:|"]
    for rate in sorted(RATES, reverse=True):
        dm = piv.loc[rate, ("medible", "selectivo")] - piv.loc[rate, ("medible", "uniforme")]
        dr = piv.loc[rate, ("real", "selectivo")] - piv.loc[rate, ("real", "uniforme")]
        dp = piv.loc[rate, ("persist", "selectivo")] - piv.loc[rate, ("persist", "uniforme")]
        lines.append(f"| {rate:.1%} | {dm:+.3f} | {dr:+.3f} | {dp:+.3f} |")
    lines.append("")
    return lines


# ────────────────────────── main ──────────────────────────

ARGS: argparse.Namespace


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--max-per-coord", type=int, default=30)
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--log-level", default="INFO")
    ARGS = ap.parse_args()
    logging.basicConfig(level=getattr(logging, ARGS.log_level), format="%(levelname)s: %(message)s")

    if not ARGS.source.exists():
        LOGGER.error("Fuente no encontrada: %s", ARGS.source)
        return 1

    df, dist_names = load_universe(ARGS.source, ARGS.max_per_coord)
    matrix_ubigeos = set(
        pd.read_parquet(ROOT / "data/silver/h3_feature_matrix.parquet", columns=["ubigeo"])
        ["ubigeo"].astype(str).str.zfill(6).unique())

    md = [
        "# Degradación selectiva de geocodificación: mecanismo + curva (slr-7lsf)",
        "",
        "**Ancla:** limitación 4 del SIMBig — el binomial thinning uniforme (slr-dj7, "
        "`analysis/etapa2_geocoding_degradation.md`) es una cota optimista; aquí se "
        "caracteriza el mecanismo REAL de fallo de geocodificación en Lima y se re-corre "
        "la curva de evaluabilidad con pérdida selectiva empírica. "
        "Script: `scripts/eval_selective_degradation.py` (seed fija, regenerable).",
        "",
        f"Nota de base: la convención del experimento original usa {BASE_RATE:.1%}; en este "
        f"universo la tasa CON COORDENADA medida es {df['con_coord'].mean():.1%} y el éxito "
        f"limpio (lo que sobrevive al anti-centroide) {df['clean'].mean():.1%}.",
        "",
    ]

    lines1, _dist, facts = phase1(df, matrix_ubigeos, dist_names)
    md += lines1

    res = None
    if ARGS.phase in ("2", "all"):
        q_dc, cat_rate = propensities(df)
        res = phase2(q_dc, cat_rate, ARGS.seeds)
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT_CSV, index=False)
        LOGGER.info("per-seed → %s", OUT_CSV)
    elif OUT_CSV.exists():
        res = pd.read_csv(OUT_CSV)
        LOGGER.info("fase 2 reutilizada desde %s", OUT_CSV)
    if res is not None:
        n_seeds = int(res["seed"].max()) + 1
        md += phase2_lines(res, n_seeds)

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nGuardado → {OUT_MD}")
    emit_canon(facts, res)
    return 0


# ─── registro canónico (cifras portantes del camera-ready SIMBig §5.4 / Tabla 4) ──
CANON_LEVELS = {0.834: "83", 0.50: "50", 0.249: "25", 0.10: "10"}


def emit_canon(facts: dict, res: pd.DataFrame | None) -> None:
    """Emite al registro canónico. Las policies (precisión/variante) son hand-written
    en analysis/canonical_numbers.json; el script no se autoproclama canónico."""
    import canon

    script = str(Path(__file__).resolve())
    inputs = [str(ARGS.source), str(POBREZA_FILE),
              str(ROOT / "data/silver/h3_feature_matrix.parquet")]
    variant = "lima_2019_2023"
    for k in ("rho_nocoord_pov", "rho_centroid_pov", "rho_clean_pov"):
        canon.emit(f"geo_failure.{k}", facts[k], variant=variant,
                   unit="Spearman ρ entre tasa distrital y pobreza INEI 2018 (distritos n≥500)",
                   estimator="spearmanr sobre tasas distritales del universo limpio 2019-2023",
                   inputs=inputs, script=script)
    for k, v in facts.items():
        if k.startswith("nocoord_pct."):
            canon.emit(f"geo_failure.{k}", v, variant=variant,
                       unit="% de denuncias sin coordenada capturada",
                       estimator="media por categoría, universo limpio 2019-2023",
                       inputs=inputs, script=script)
    if res is None:
        return
    variant = "m4c_hgb_10seeds"
    agg = res.groupby(["mechanism", "rate"]).agg(
        medible=("medible", "mean"), medible_sd=("medible", "std"),
        real=("real", "mean"), real_sd=("real", "std"),
        persist=("persist", "mean"), persist_sd=("persist", "std"),
        feat_wins=("feat_gana", "mean"))
    mech_en = {"uniforme": "uniform", "selectivo": "selective"}
    for (mech, rate), row in agg.iterrows():
        if rate not in CANON_LEVELS:
            continue
        lvl = CANON_LEVELS[rate]
        for metric in ("medible", "real", "persist"):
            canon.emit(f"geo_degradation.{mech_en[mech]}_{metric}_{lvl}", row[metric],
                       variant=variant, unit="macro ρ intra-distrital (media sobre seeds)",
                       estimator=f"M4c HGB reentrenado sobre target degradado, {mech}, nivel {rate}",
                       inputs=inputs, script=script)
            sd = row[f"{metric}_sd"]
            canon.emit(f"geo_degradation.{mech_en[mech]}_{metric}_sd_{lvl}",
                       0.0 if pd.isna(sd) else sd,
                       variant=variant, unit="sd entre seeds del macro ρ",
                       estimator=f"sd muestral sobre seeds, {mech}, nivel {rate}",
                       inputs=inputs, script=script)
    # brecha selectivo − uniforme y % de seeds con features > persistencia
    for rate, lvl in CANON_LEVELS.items():
        if rate >= BASE_RATE:
            continue
        u, s_ = agg.loc[("uniforme", rate)], agg.loc[("selectivo", rate)]
        for metric in ("medible", "real", "persist"):
            canon.emit(f"geo_degradation.delta_sel_uni_{metric}_{lvl}", s_[metric] - u[metric],
                       variant=variant, unit="Δ macro ρ (selectivo − uniforme)",
                       estimator=f"diferencia de medias sobre seeds, nivel {rate}",
                       inputs=inputs, script=script)
    canon.emit("geo_degradation.feat_wins_pct_min", 100.0 * float(agg["feat_wins"].min()),
               variant=variant, unit="% de seeds con ρ(features) > ρ(persistencia), mínimo sobre niveles y mecanismos",
               estimator="min sobre celdas (mecanismo, nivel) de la fracción de seeds feat_gana",
               inputs=inputs, script=script)


if __name__ == "__main__":
    sys.exit(main())
