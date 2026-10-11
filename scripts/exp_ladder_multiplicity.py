#!/usr/bin/env python3
"""
¿Sobrevive el escalón de manzana censal a un control de error familiar? (slr-p1u.18, A4)

El titular del paper SIMBig es que el ladder tabular tiene EXACTAMENTE UN escalón
real: INEI manzana sobre demografía distrital (+0.075). Ese veredicto sale de mirar
si el CI pareado al 95% de cada escalón excluye 0 — una decisión por escalón, sin
control de multiplicidad. Un referee puede tumbar el titular con una frase: "tres
comparaciones al 5% cada una no son una conclusión al 5%".

Este script responde formalmente, sin re-entrenar: el bootstrap del protocolo opera
sobre ρ por (categoría, distrito) YA calculado, así que lo caro se hace una vez.

Lo que cambia respecto de `eval_metric_robustness.py`:

1. **Bootstrap COMPARTIDO.** Allá cada comparación llama a `boot_paired_delta` por
   separado, con su propio sorteo. Acá se sortean los distritos UNA vez por réplica y
   se evalúan TODOS los escalones sobre ese mismo remuestreo, de modo que las réplicas
   conservan la correlación entre estadísticos — que es justo lo que un control de
   error familiar necesita para no ser absurdamente conservador.

2. **Westfall-Young step-down max-t.** El ajuste por remuestreo estándar cuando los
   contrastes están correlacionados (acá lo están, y mucho: comparten modelo base,
   distritos y oráculo). Bonferroni/Holm se reportan al lado como referencia
   conservadora, no como el resultado principal.

3. **Banda simultánea max-t.** Un `c` tal que P(max_j |t*_j| ≤ c) = 0.95, y los CIs
   d_j ± c·se_j son válidos SIMULTÁNEAMENTE sobre toda la familia. Es lo que la Tabla 2
   del paper debería poder mostrar si quiere hablar de "exactamente un escalón".

4. **Tolerancia de familia.** El m más grande para el que el escalón de manzana sigue
   excluyendo 0 bajo Bonferroni. Responde con un número la objeción de que la familia
   "de verdad" es más grande (celdas del factorial, ciudades, niveles de thinning).

FAMILIA PRIMARIA = los tres contrastes que la Tabla 2 del manuscrito somete a prueba:
  H1  M1b_manzana − M1_demografia      (fila "+ INEI census blocks")
  H2  M2_osm      − M1b_manzana        (fila "+ OSM urban form")
  H3  M4c_transporte − M1b_manzana     (fila "+ 8 further sources", delta CONJUNTO)
Se reporta además una familia ampliada que agrega M4c − M2, para mostrar que el
veredicto no depende de cómo se dibuje la familia.

Uso: python3 scripts/exp_ladder_multiplicity.py [--B 20000]
Salida: analysis/etapa2_ladder_multiplicity.md + entradas canónicas ladder_mult.*
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import canon  # noqa: E402
from eval_metric_robustness import (  # noqa: E402
    STEPS,
    TEST_YEAR,
    TRAIN_YEARS,
    fit_predict,
    per_district_stats,
)
from eval_modality_ladder import FEATURE_SETS, available, load_panel  # noqa: E402

OUT_MD = ROOT / "analysis/etapa2_ladder_multiplicity.md"
ALPHA = 0.05
SEED = 20260731

# (etiqueta, escalón base, escalón con la fuente agregada)
FAMILY_PRIMARY = [
    ("H1 manzana censal", "M1_demografia", "M1b_manzana"),
    ("H2 forma urbana OSM", "M1b_manzana", "M2_osm"),
    ("H3 ocho fuentes (conjunto)", "M1b_manzana", "M4c_transporte"),
]
FAMILY_EXTRA = [("H4 transporte sobre OSM", "M2_osm", "M4c_transporte")]

LOGGER = logging.getLogger("exp_ladder_multiplicity")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ─── bootstrap compartido ─────────────────────────────────────────────────────
def aligned_rho_matrix(stats: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Por categoría, matriz (n_distritos × n_escalones) de ρ intra alineada.

    Solo entran los distritos evaluables en TODOS los escalones: un remuestreo
    compartido exige que la unidad sorteada exista en cada brazo. Se reporta cuántos
    se pierden — si fueran muchos, el pareo dejaría de ser inocuo.
    """
    merged = None
    for step in STEPS:
        s = stats[step][["crime_cat", "ubigeo", "rho"]].rename(columns={"rho": step})
        merged = s if merged is None else merged.merge(s, on=["crime_cat", "ubigeo"], how="inner")
    merged = merged.dropna(subset=STEPS)
    return {cat: g[STEPS].to_numpy(float) for cat, g in merged.groupby("crime_cat")}, merged


def shared_bootstrap(mats: dict[str, np.ndarray], B: int, seed: int) -> np.ndarray:
    """Draws (B × n_escalones) del macro-ρ, sorteando distritos UNA vez por réplica.

    El macro es media entre categorías de la media entre distritos, igual que el
    protocolo del ladder. La clave es que el mismo `idx` alimenta todos los escalones.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((B, len(STEPS)))
    for cat, R in mats.items():
        n = R.shape[0]
        idx = rng.integers(0, n, size=(B, n))
        acc += R[idx].mean(axis=1)          # (B, n_escalones)
    return acc / len(mats)


def macro_point(mats: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([R.mean(axis=0) for R in mats.values()], axis=0)


# ─── multiplicidad ────────────────────────────────────────────────────────────
def westfall_young(t_obs: np.ndarray, t_null: np.ndarray) -> np.ndarray:
    """p ajustados step-down max-t (Westfall-Young), respetando la correlación.

    `t_null` es (B × k) de estadísticos CENTRADOS bajo el nulo. Se ordena por |t_obs|
    descendente; para cada hipótesis el nulo es el máx sobre las AÚN no rechazadas, y
    al final se impone monotonía — sin ella el procedimiento no controla FWER.
    """
    k = len(t_obs)
    order = np.argsort(-np.abs(t_obs))
    adj = np.empty(k)
    for pos, j in enumerate(order):
        remaining = order[pos:]
        maxnull = np.abs(t_null[:, remaining]).max(axis=1)
        adj[j] = (1 + np.sum(maxnull >= abs(t_obs[j]))) / (len(t_null) + 1)
    running = 0.0
    for j in order:                      # monotonía en el orden de step-down
        running = max(running, adj[j])
        adj[j] = running
    return adj


def holm(p: np.ndarray) -> np.ndarray:
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k)
    running = 0.0
    for pos, j in enumerate(order):
        running = max(running, min(1.0, (k - pos) * p[j]))
        adj[j] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=20000, help="réplicas bootstrap (default 20000)")
    args = ap.parse_args()
    B = args.B

    panel, _ = load_panel()
    stats = {}
    for step in STEPS:
        feats = available(FEATURE_SETS[step], panel)
        LOGGER.info("%s: %d features", step, len(feats))
        stats[step] = per_district_stats(fit_predict(panel, feats))

    mats, merged = aligned_rho_matrix(stats)
    n_pairs = len(merged)
    n_raw = min(len(stats[s]) for s in STEPS)
    LOGGER.info("pares (cat, distrito) alineados: %d (mínimo por escalón suelto: %d)", n_pairs, n_raw)

    point = macro_point(mats)                       # (n_escalones,)
    draws = shared_bootstrap(mats, B, SEED)         # (B, n_escalones)
    step_ix = {s: i for i, s in enumerate(STEPS)}

    fam = FAMILY_PRIMARY + FAMILY_EXTRA
    d_obs, d_draws = [], []
    for _, base, adv in fam:
        i, j = step_ix[base], step_ix[adv]
        d_obs.append(point[j] - point[i])
        d_draws.append(draws[:, j] - draws[:, i])
    d_obs = np.array(d_obs)
    d_draws = np.stack(d_draws, axis=1)             # (B, k)

    se = d_draws.std(axis=0, ddof=1)
    t_obs = d_obs / se
    t_null = (d_draws - d_obs) / se                 # centrado bajo el nulo

    k_prim = len(FAMILY_PRIMARY)
    # p marginal (sin ajuste) por remuestreo, con el piso 1/(B+1) explícito
    p_marg = np.array([(1 + np.sum(np.abs(t_null[:, j]) >= abs(t_obs[j]))) / (B + 1)
                       for j in range(len(fam))])
    p_wy_prim = westfall_young(t_obs[:k_prim], t_null[:, :k_prim])
    p_wy_full = westfall_young(t_obs, t_null)
    p_holm_prim = holm(p_marg[:k_prim])

    # banda simultánea max-t sobre la familia primaria
    maxnull_prim = np.abs(t_null[:, :k_prim]).max(axis=1)
    c_sim = float(np.percentile(maxnull_prim, 100 * (1 - ALPHA)))
    lo_sim, hi_sim = d_obs - c_sim * se, d_obs + c_sim * se

    # CI percentil marginal 95% (reproduce la tabla existente: control de sanidad)
    lo_marg = np.percentile(d_draws, 100 * ALPHA / 2, axis=0)
    hi_marg = np.percentile(d_draws, 100 * (1 - ALPHA / 2), axis=0)

    # tolerancia de familia para H1 bajo Bonferroni
    p1 = p_marg[0]
    at_floor = np.sum(np.abs(t_null[:, 0]) >= abs(t_obs[0])) == 0
    m_tol = int(np.floor(ALPHA / p1))

    lines = [
        "# Multiplicidad del ladder: ¿sobrevive el escalón de manzana? (slr-p1u.18, A4)",
        "",
        f"**Protocolo:** train {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}, test {TEST_YEAR}, oráculo "
        f"geocodificado, HGB con params del ladder. Bootstrap **compartido** sobre distritos "
        f"dentro de categoría, B={B:,}, seed {SEED}. Los cuatro escalones se evalúan sobre el "
        f"MISMO remuestreo en cada réplica, así que las réplicas conservan la correlación entre "
        f"contrastes. Pares (categoría, distrito) alineados en los cuatro escalones: "
        f"**{n_pairs}**.",
        "",
        "## 1. Contrastes, marginal vs simultáneo",
        "",
        "| # | contraste | Δρ | CI 95% marginal | CI 95% **simultáneo** (max-t) | p Westfall-Young |",
        "|:--|:--|--:|:--|:--|--:|",
    ]
    floor_p = 1 / (B + 1)

    def fmt_p(p: float) -> str:
        """Nunca imprimir 0.0000: un p en el piso del bootstrap es una COTA, no un cero."""
        return f"< {floor_p:.1e}" if p <= floor_p else f"{p:.4f}"

    for j, (lab, _, _) in enumerate(fam):
        fam_tag = "" if j < k_prim else " *(fuera de la familia primaria)*"
        pwy = p_wy_prim[j] if j < k_prim else p_wy_full[j]
        sim = (f"[{lo_sim[j]:+.4f}, {hi_sim[j]:+.4f}]" if j < k_prim else "—")
        lines.append(
            f"| {j+1} | {lab}{fam_tag} | {d_obs[j]:+.4f} | "
            f"[{lo_marg[j]:+.4f}, {hi_marg[j]:+.4f}] | {sim} | {fmt_p(pwy)} |")

    lines += [
        "",
        f"Valor crítico simultáneo max-t al 95%: **c = {c_sim:.3f}** "
        f"(un CI marginal usaría ≈1.96; el precio de la simultaneidad sobre {k_prim} "
        f"contrastes correlacionados es ese {c_sim/1.96:.2f}×).",
        "",
        "## 2. Ajustes por multiplicidad (familia primaria de 3)",
        "",
        "| contraste | p marginal | p Holm-Bonferroni | p Westfall-Young | ¿sobrevive α=0.05? |",
        "|:--|--:|--:|--:|:--|",
    ]
    for j, (lab, _, _) in enumerate(FAMILY_PRIMARY):
        surv = "**SÍ**" if p_wy_prim[j] < ALPHA else "no"
        lines.append(f"| {lab} | {fmt_p(p_marg[j])} | {fmt_p(p_holm_prim[j])} | "
                     f"{fmt_p(p_wy_prim[j])} | {surv} |")

    floor_note = (
        f"En las {B:,} réplicas NINGUNA alcanzó el |t| observado del escalón de manzana, "
        f"así que su p marginal está en el piso de resolución del bootstrap, "
        f"p ≤ {p1:.2e} = 1/(B+1); todo lo que sigue usa ese piso y por lo tanto "
        f"**subestima** la tolerancia real."
        if at_floor else
        f"El p marginal del escalón de manzana es {p1:.2e}, por encima del piso de "
        f"resolución 1/(B+1) = {1/(B+1):.2e}."
    )
    lines += [
        "",
        "## 3. Tolerancia de familia del escalón de manzana",
        "",
        floor_note,
        "",
        f"Bajo Bonferroni el escalón sigue excluyendo 0 para cualquier familia de hasta "
        f"**m ≈ {m_tol:,} contrastes simultáneos**. La familia primaria tiene 3; incluso "
        f"la lectura más hostil —contar cada fuente del panel, cada celda del factorial, "
        f"cada ciudad y cada nivel de thinning como un contraste más— no se acerca a ese "
        f"orden de magnitud.",
        "",
        "## 4. Lectura",
        "",
    ]
    h1_ok = p_wy_prim[0] < ALPHA
    others_ok = all(p_wy_prim[j] >= ALPHA for j in range(1, k_prim))
    if h1_ok and others_ok:
        lines += [
            "El veredicto del ladder **no era un artefacto de multiplicidad**. Con control de "
            "error familiar por remuestreo, el escalón de manzana censal sigue siendo "
            "significativo y los otros dos siguen sin serlo: la conclusión 'exactamente un "
            "escalón real' se mantiene *después* de pagar el precio de las tres pruebas, no "
            "solo antes.",
            "",
            "Vale registrar la asimetría lógica: el control de multiplicidad **fortalece** el "
            "lado positivo del claim (el escalón sobrevive) pero **no** convierte los dos "
            "no-significativos en evidencia de ausencia. Que OSM y las ocho fuentes restantes "
            "no crucen el umbral —ni marginal ni simultáneo— sigue siendo compatible con "
            "efectos pequeños que este diseño no puede resolver; el CI simultáneo, más ancho "
            "que el marginal, lo dice explícitamente.",
        ]
    else:
        lines += [
            "**El veredicto cambia bajo control de multiplicidad.** Ver la tabla 2: el claim "
            "del manuscrito necesita reescribirse, no solo matizarse.",
        ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    inputs = [str(ROOT / "data/silver/h3_feature_matrix.parquet")]
    canon.emit("ladder_mult.manzana_delta", float(d_obs[0]), variant="shared_boot",
               unit="Δ macro ρ intra-distrital, manzana censal sobre demografía distrital",
               estimator=f"bootstrap compartido sobre distritos, B={B}, seed {SEED}",
               inputs=inputs, script=__file__)
    canon.emit("ladder_mult.manzana_p_wy", float(p_wy_prim[0]), variant="shared_boot",
               unit="p ajustado Westfall-Young step-down max-t, familia primaria de 3",
               estimator=f"max-t sobre réplicas centradas, B={B}, seed {SEED}",
               inputs=inputs, script=__file__)
    # Los dos nulos también son portantes: sostienen "el resto sigue sin ser significativo
    # DESPUÉS de la corrección", que es media conclusión del titular.
    for j, slug in ((1, "osm"), (2, "ocho_fuentes")):
        canon.emit(f"ladder_mult.{slug}_p_wy", float(p_wy_prim[j]), variant="shared_boot",
                   unit=f"p ajustado Westfall-Young del contraste {FAMILY_PRIMARY[j][0]}",
                   estimator=f"max-t step-down, familia primaria de 3, B={B}, seed {SEED}",
                   inputs=inputs, script=__file__)
    canon.emit("ladder_mult.simultaneous_crit", c_sim, variant="shared_boot",
               unit="valor crítico max-t para CIs simultáneos al 95%, familia de 3",
               estimator=f"percentil 95 de max_j |t*_j|, B={B}, seed {SEED}",
               inputs=inputs, script=__file__)
    canon.emit("ladder_mult.manzana_sim_lo", float(lo_sim[0]), variant="shared_boot",
               unit="límite inferior del CI SIMULTÁNEO 95% del escalón de manzana",
               estimator=f"d ± c·se con c max-t, B={B}, seed {SEED}",
               inputs=inputs, script=__file__)
    canon.emit("ladder_mult.manzana_sim_hi", float(hi_sim[0]), variant="shared_boot",
               unit="límite superior del CI SIMULTÁNEO 95% del escalón de manzana",
               estimator=f"d ± c·se con c max-t, B={B}, seed {SEED}",
               inputs=inputs, script=__file__)
    canon.emit("ladder_mult.family_tolerance", float(m_tol), variant="shared_boot",
               unit="familia máxima de contrastes que el escalón tolera bajo Bonferroni",
               estimator=f"floor(0.05 / p_marginal); p en el piso 1/(B+1) si at_floor, B={B}",
               inputs=inputs, script=__file__)

    print(f"report -> {OUT_MD}")
    for j, (lab, _, _) in enumerate(FAMILY_PRIMARY):
        print(f"  {lab}: Δ={d_obs[j]:+.4f}  p_WY={p_wy_prim[j]:.4f}  "
              f"sim=[{lo_sim[j]:+.4f}, {hi_sim[j]:+.4f}]")
    print(f"  c_max-t={c_sim:.3f}   tolerancia de familia m≈{m_tol:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
