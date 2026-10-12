#!/usr/bin/env python3
"""
Métricas de evaluación Etapa 2 (F0b / slr-7xn). Módulo importable, sin estado.

Usado por eval_model.py (F1), la ladder de modalidades (F2), CV (F3) y observado-vs-latente
(F4). Mide qué tan bien una predicción a H3 recupera la concentración REAL del crimen
(oráculo geocodificado de F0a), con énfasis en:

  - RANKING:        Spearman ρ, Kendall τ
  - HOTSPOT/POLICY: Recall@k% (masa de crimen real capturada en el top-k% predicho),
                    NDCG@k
  - CONCENTRACIÓN:  PAI (Predictive Accuracy Index), PEI (Predictive Efficiency Index)
  - MAGNITUD:       MAE normalizado, sMAPE, log-RMSE
  - ROBUSTEZ:       bootstrap CI
  - MACRO-average:  promedio sobre categorías/distritos (no micro) → no esconde categorías raras

CLAVE: `eval_intradistrict` mide el ranking DENTRO de cada distrito y macro-promedia, aislando
la señal intra-distrital del efecto distrital. Esto es lo que decide si el modelo aporta
resolución fina sobre un prior distrital (la pregunta Q2).

No usar MAE bruto como métrica principal: las escalas difieren mucho entre categorías
(estafa/secuestro) → se reporta normalizado y se macro-promedia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-9


# ─── Ranking ──────────────────────────────────────────────────────────────────

def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) < EPS or np.std(y_pred) < EPS:
        return np.nan
    return float(stats.spearmanr(y_true, y_pred).statistic)


def kendall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) < EPS or np.std(y_pred) < EPS:
        return np.nan
    return float(stats.kendalltau(y_true, y_pred).statistic)


# ─── Hotspot / policy ─────────────────────────────────────────────────────────

def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10) -> float:
    """Fracción de la masa de crimen REAL capturada en el top-(frac) de celdas por predicción.

    Métrica policial: si patrullo el frac% de celdas de mayor riesgo predicho, ¿qué % del
    crimen real cubro? Equivale al numerador del PAI.
    """
    n = len(y_true)
    if n == 0 or y_true.sum() <= 0:
        return np.nan
    k = max(1, int(round(frac * n)))
    top = np.argsort(y_pred)[::-1][:k]
    return float(y_true[top].sum() / y_true.sum())


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int | None = None) -> float:
    """NDCG con relevancia = conteo real. k=None → todas las celdas."""
    n = len(y_true)
    if n == 0 or y_true.sum() <= 0:
        return np.nan
    k = n if k is None else min(k, n)

    def dcg(order: np.ndarray) -> float:
        rel = y_true[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
        return float((rel * discounts).sum())

    dcg_pred = dcg(np.argsort(y_pred)[::-1])
    dcg_ideal = dcg(np.argsort(y_true)[::-1])
    return dcg_pred / dcg_ideal if dcg_ideal > EPS else np.nan


def pai(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10) -> float:
    """Predictive Accuracy Index = recall@frac / frac. PAI>1 → concentración útil."""
    r = recall_at_k(y_true, y_pred, frac)
    return r / frac if r == r else np.nan  # r==r descarta NaN


def pei(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10) -> float:
    """Predictive Efficiency Index = PAI(pred) / PAI(oracle perfecto). Normaliza a [0,1]."""
    pai_pred = pai(y_true, y_pred, frac)
    pai_best = pai(y_true, y_true, frac)
    if pai_pred != pai_pred or pai_best != pai_best or pai_best < EPS:
        return np.nan
    return float(pai_pred / pai_best)


# ─── Magnitud ─────────────────────────────────────────────────────────────────

def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE / media(y_true). Comparable entre categorías de distinta escala."""
    denom = np.mean(y_true)
    if abs(denom) < EPS:
        return np.nan
    return float(np.mean(np.abs(y_pred - y_true)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > EPS
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]))


def log_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.log1p(np.clip(y_true, 0, None))
    b = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ─── Robustez ─────────────────────────────────────────────────────────────────

def bootstrap_ci(metric_fn, y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """(point, lo, hi) por bootstrap de filas. Para CI por distrito/categoría."""
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_pred)
    n = len(y_true)
    if n < 5:
        return point, np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(y_true[idx], y_pred[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ─── Intra-distrital (la prueba clave de resolución fina) ─────────────────────

def eval_intradistrict(df: pd.DataFrame, pred_col: str, oracle_col: str,
                       district_col: str = "ubigeo",
                       min_cells: int = 5,
                       metrics: tuple[str, ...] = ("spearman", "recall10", "ndcg", "pai")) -> dict:
    """Macro-promedio sobre distritos de métricas calculadas DENTRO de cada distrito.

    Aísla la señal intra-distrital: ordena celdas dentro de cada distrito por pred vs oracle.
    Un prior poblacional pierde aquí (no reordena dentro del distrito); un buen modelo gana.

    Devuelve dict con cada métrica → (macro_mean, n_districts) y la lista cruda por distrito.
    """
    fns = {
        "spearman": spearman,
        "kendall": kendall,
        "recall5": lambda t, p: recall_at_k(t, p, 0.05),
        "recall10": lambda t, p: recall_at_k(t, p, 0.10),
        "ndcg": lambda t, p: ndcg_at_k(t, p, None),
        "pai": lambda t, p: pai(t, p, 0.10),
        "pei": lambda t, p: pei(t, p, 0.10),
    }
    per_district: dict[str, dict] = {}
    for dist, g in df.groupby(district_col):
        if len(g) < min_cells:
            continue
        t = g[oracle_col].to_numpy(dtype=float)
        p = g[pred_col].to_numpy(dtype=float)
        if t.sum() <= 0:
            continue
        per_district[dist] = {m: fns[m](t, p) for m in metrics}

    out: dict = {"per_district": per_district, "n_districts": len(per_district)}
    for m in metrics:
        vals = [d[m] for d in per_district.values() if d[m] == d[m]]
        out[m] = float(np.mean(vals)) if vals else np.nan
    return out


def evaluate(df: pd.DataFrame, pred_col: str, oracle_col: str,
             district_col: str = "ubigeo", boot: bool = True) -> dict:
    """Suite completa (pooled/micro) + intra-distrital (macro). Para una (categoría, año, modelo)."""
    t = df[oracle_col].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    res = {
        "n_cells": len(df),
        "spearman_global": spearman(t, p),
        "kendall_global": kendall(t, p),
        "recall5": recall_at_k(t, p, 0.05),
        "recall10": recall_at_k(t, p, 0.10),
        "ndcg": ndcg_at_k(t, p, None),
        "pai10": pai(t, p, 0.10),
        "pei10": pei(t, p, 0.10),
        "nmae": normalized_mae(t, p),
        "smape": smape(t, p),
        "log_rmse": log_rmse(t, p),
    }
    if boot:
        _, lo, hi = bootstrap_ci(spearman, t, p)
        res["spearman_ci_low"], res["spearman_ci_high"] = lo, hi
    intra = eval_intradistrict(df, pred_col, oracle_col, district_col)
    res["intra_spearman"] = intra["spearman"]
    res["intra_recall10"] = intra["recall10"]
    res["intra_ndcg"] = intra["ndcg"]
    res["intra_pai"] = intra["pai"]
    res["intra_n_districts"] = intra["n_districts"]
    return res


def macro_average(per_category: dict[str, dict], keys: tuple[str, ...]) -> dict:
    """Macro-promedio sobre categorías (no micro) → categorías raras pesan igual."""
    out = {}
    for k in keys:
        vals = [r[k] for r in per_category.values() if k in r and r[k] == r[k]]
        out[f"macro_{k}"] = float(np.mean(vals)) if vals else np.nan
        out[f"min_{k}"] = float(np.min(vals)) if vals else np.nan  # peor categoría (maximin)
    return out


def _selftest() -> None:
    """Verificación rápida con señal sintética."""
    rng = np.random.default_rng(0)
    n = 500
    true = rng.poisson(3, n).astype(float)
    good = true + rng.normal(0, 0.5, n)      # predicción correlacionada
    bad = rng.normal(0, 1, n)                 # ruido
    assert spearman(true, good) > 0.8, spearman(true, good)
    assert abs(spearman(true, bad)) < 0.2
    assert recall_at_k(true, true, 0.10) >= recall_at_k(true, good, 0.10) >= recall_at_k(true, bad, 0.10)
    assert 0 <= pei(true, good, 0.10) <= 1.0001
    assert pai(true, true, 0.10) >= pai(true, good, 0.10)
    df = pd.DataFrame({
        "ubigeo": np.repeat(["A", "B", "C"], n // 3 + 1)[:n],
        "oracle": true, "pred_good": good, "pred_bad": bad,
    })
    rg = eval_intradistrict(df, "pred_good", "oracle")
    rb = eval_intradistrict(df, "pred_bad", "oracle")
    assert rg["spearman"] > rb["spearman"]
    print("etapa2_metrics selftest OK:")
    print(f"  spearman good={spearman(true, good):.3f} bad={spearman(true, bad):.3f}")
    print(f"  recall@10 perfect={recall_at_k(true, true, .1):.3f} good={recall_at_k(true, good, .1):.3f} bad={recall_at_k(true, bad, .1):.3f}")
    print(f"  PAI good={pai(true, good, .1):.2f}  PEI good={pei(true, good, .1):.3f}")
    print(f"  intra-district spearman good={rg['spearman']:.3f} bad={rb['spearman']:.3f}")


if __name__ == "__main__":
    _selftest()
