#!/usr/bin/env python3
"""¿El p=0.025 de heterogeneidad distrital sobrevive calibración y design-effects? (C2, slr-x50g)

MOTIVO. Tras el bug-fix de pooling (7e0db4c), el pseudo-LRT contra el límite homogéneo da
p=0.025 (χ²(1)) — heterogeneidad distrital de r̂ débil pero detectable. Antes de decidir si
el paper la upgradea de "solo orden" a "hallazgo débil" (decisión Rody+Christian), dos
chequeos que el χ² asintótico no cubre:

1. **Calibración por bootstrap paramétrico.** Bajo H0 (r común, φ→∞) se simulan paneles
   y*_d ~ Binomial(n_d, μ̂0) y se recomputa el LRT perfilado → p calibrado por simulación,
   sin apelar al χ²(1) (que además ignora el borde del espacio de parámetros).
2. **Sensibilidad al design-effect.** Los n_eff=(Σw)²/Σw² son tamaños efectivos de diseño;
   si el design effect real fuera peor (n_eff sobreestimado), el LRT observado se encoge.
   Se re-escala (y_eff, n_eff)→κ·(y_eff, n_eff) para κ∈{0.5,0.75,1.25,1.5} (preserva r̂_d)
   y se recomputa el LRT+p. La pregunta operativa: ¿el rechazo sobrevive κ=0.5 (design
   effects 2× peores de lo estimado)?

Lectura para la decisión: si p_boot>0.05 o el rechazo muere con κ<1, la opción conservadora
(solo-orden, texto actual de §5.3) se decide sola; si sobrevive, el upgrade es defendible.
Informe de síntesis: analysis/phi_heterogeneity_decision.typ (vivo desde canon).

Uso:  python3 scripts/phi_lrt_bootstrap.py [--sims 400] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402
from build_reporting_rate import (  # noqa: E402
    ENAPRES_RAW, betabinom_nll, eff, lima_districts_canonical, load_cells,
)

CAT = "robo_hurto_callejero"
YEARS = list(range(2019, 2025))


def load_pooled():
    """Pooling canónico (espejo de build_reporting_rate; mismo del recheck corregido)."""
    cells = load_cells(YEARS)
    lima = [u for u in lima_districts_canonical() if u.startswith("1501")]
    y, n = [], []
    for u in lima:
        agg = dict(sw=0.0, sw2=0.0, swd=0.0)
        for t in YEARS:
            d = cells.get((t, CAT, u))
            if d:
                for k in agg:
                    agg[k] += d[k]
        _, n_eff, y_eff = eff(agg)
        if n_eff > 0:
            y.append(y_eff)
            n.append(n_eff)
    return np.array(y), np.array(n)


def profile_lrt(y, n) -> tuple[float, float]:
    """LRT del φ̂ interior (perfil) contra el límite homogéneo φ→∞. Devuelve (stat, φ̂)."""
    def nll_at(phi):
        r = minimize(lambda m: betabinom_nll([m[0], np.log(phi)], y, n),
                     [max(y.sum() / n.sum(), 1e-4)],
                     method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8))
        return float(r.fun)
    opt = minimize_scalar(lambda lp: nll_at(float(np.exp(lp))),
                          bounds=(np.log(5), np.log(5_000)), method="bounded",
                          options=dict(xatol=1e-3))
    nll_hat = float(opt.fun)
    return 2 * (nll_at(1e6) - nll_hat), float(np.exp(opt.x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    y, n = load_pooled()
    lrt_obs, phi_hat = profile_lrt(y, n)
    p_chi2 = chi2.sf(max(lrt_obs, 0.0), df=1)
    print(f"observado: LRT={lrt_obs:.3f}  φ̂={phi_hat:.0f}  p_chi2(1)={p_chi2:.4f}  "
          f"(mezcla de borde ≈ {p_chi2 / 2:.4f})")

    # 1) bootstrap paramétrico bajo H0 (r común; Binomial con n redondeado)
    mu0 = float(y.sum() / n.sum())
    n_int = np.maximum(np.round(n).astype(int), 1)
    stats = []
    for s in range(args.sims):
        y_s = rng.binomial(n_int, mu0).astype(float)
        stat_s, _ = profile_lrt(y_s, n_int.astype(float))
        stats.append(max(stat_s, 0.0))
        if (s + 1) % 100 == 0:
            print(f"  ... {s + 1}/{args.sims} sims")
    stats = np.array(stats)
    p_boot = float((np.sum(stats >= lrt_obs) + 1) / (args.sims + 1))
    print(f"\n1) BOOTSTRAP H0 ({args.sims} sims, seed={args.seed}): "
          f"p_boot={p_boot:.4f}   (LRT* q95={np.percentile(stats, 95):.2f}, "
          f"masa en 0: {np.mean(stats < 1e-6):.0%})")

    # 2) sensibilidad al design-effect
    print("\n2) SENSIBILIDAD κ (escala de n_eff; κ<1 = design effects peores):")
    kappa_p = {}
    for k in (0.5, 0.75, 1.25, 1.5):
        stat_k, _ = profile_lrt(k * y, k * n)
        pk = chi2.sf(max(stat_k, 0.0), df=1)
        kappa_p[k] = pk
        flag = "  ← rechazo muere" if pk >= 0.05 else ""
        print(f"   κ={k:>4}  LRT={stat_k:6.3f}  p={pk:.4f}{flag}")

    inputs = sorted(ENAPRES_RAW.glob("*.zip"))
    canon.emit(f"phi_lrt_p.{CAT}", p_boot, variant="boot",
               unit="p calibrado por bootstrap paramétrico bajo H0 (r común)",
               estimator=f"{args.sims} sims Binomial, seed={args.seed}, LRT perfilado",
               inputs=inputs, script=__file__)
    for k in (0.5, 0.75, 1.25, 1.5):
        canon.emit(f"phi_lrt_p.{CAT}", float(kappa_p[k]), variant=f"kappa{int(k * 100):03d}",
                   unit=f"p χ²(1) del LRT con n_eff re-escalado ×{k} (sensibilidad design-effect)",
                   estimator="LRT perfilado sobre (κ·y_eff, κ·n_eff)",
                   inputs=inputs, script=__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
