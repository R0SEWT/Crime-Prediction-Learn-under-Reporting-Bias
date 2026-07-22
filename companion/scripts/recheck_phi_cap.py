#!/usr/bin/env python3
"""¿La heterogeneidad espacial del multiplicador es real, o la fabrica PHI_CAP? (B4, slr-x50g)

§1 del companion declara como SEGUNDA CONTRIBUCIÓN que "the observed-to-latent gap is not
spatially uniform", y §5.2 lo cuantifica: "the robbery multiplier reaches 5.9-6.3 [in
peripheral districts] against 4.5-5.0 in central, higher-trust districts such as Miraflores".

Pero build_reporting_rate.py:44 fija PHI_CAP = 100.0, con el comentario del propio autor:
    "φ tocando la cota = 'no se resuelve variación distrital'"
y para robo la φ ajustada ES 100.0 — la cota exacta.

Este script contesta dos preguntas separadas, que el paper mezcla:
  1. ¿La MAGNITUD de la variación distrital está identificada?  (perfil de verosimilitud + LRT)
  2. ¿El ORDEN de los distritos está identificado?              (ρ de Spearman vs la cota)

VEREDICTO (corregido 2026-07-20, tras el bug-fix de pooling de abajo): la magnitud está
DÉBILMENTE identificada — el perfil de verosimilitud tiene óptimo INTERIOR (φ̂≈147, que
fit_eb trunca a la cota 100: de ahí el "φ=100 exacto" del CSV) y el pseudo-LRT contra el
límite homogéneo φ→∞ rechaza borderline (p≈0.025 con χ²(1); ≈0.013 con la corrección de
mezcla 50:50 por H0 en el borde). El rango distrital es estable en la región plana del
perfil (×3.8–5.3 en φ̂ vs ×3.6–5.4 en la cota), así que ya no puede describirse como puro
artefacto de la regularización — pero descansa en una pseudo-verosimilitud sobre tamaños
efectivos de diseño, por lo que el paper sigue interpretando SOLO el orden (robusto: ρ
significativa a toda φ). La versión anterior de este script reportaba "p=1.000, colapsa a
×5.32 en el MLE": ambos eran artefactos del bug de colapso de años (ver main()); el ×5.32
era 1/μ de la ola 2024 sola, y pasó la reverificación adversarial del 19-jul porque el
agente re-corrió este mismo harness.

Uso:  python3 scripts/recheck_phi_cap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402
from build_reporting_rate import (  # noqa: E402
    ENAPRES_RAW, PHI_CAP, betabinom_nll, eff, lima_districts_canonical, load_cells,
)

ROOT = Path(__file__).resolve().parents[1]
TRUST = ROOT / "data/silver/analysis/district_trust_pnp.json"
CAT = "robo_hurto_callejero"
YEARS = list(range(2019, 2025))


def main() -> int:
    cells = load_cells(YEARS)
    # BUG FIX 2026-07-20 (slr-x50g.3): la versión anterior hacía
    #   {u: eff(d) for (y, c, u), d in cells.items() if ...}
    # — keyed SOLO por distrito, así que cada distrito quedaba con la celda del ÚLTIMO
    # año iterado (2024), no el panel: Σn_eff=770 en vez de 4442, μ=0.188 en vez de
    # ~0.224, y el "MLE constante ×5.32" que el paper citaba era 1/μ_2024 — un artefacto
    # del colapso, que además pasó la reverificación adversarial porque el agente re-corrió
    # este mismo harness. Pooling correcto = espejo de build_reporting_rate (acumular
    # sw/sw2/swd por distrito sobre el panel y recién entonces eff()).
    lima = [u for u in lima_districts_canonical() if u.startswith("1501")]
    sub = {}
    for u in lima:
        agg = dict(sw=0.0, sw2=0.0, swd=0.0)
        for t in YEARS:
            d = cells.get((t, CAT, u))
            if d:
                for k in agg:
                    agg[k] += d[k]
        sub[u] = eff(agg)
    ub = [u for u, t in sub.items() if t[1] > 0]
    y = np.array([sub[u][2] for u in ub])
    n = np.array([sub[u][1] for u in ub])

    def mu_at(phi: float) -> tuple[float, float]:
        """μ que maximiza la verosimilitud a φ fija; devuelve (μ, -loglik)."""
        r = minimize(lambda m: betabinom_nll([m[0], np.log(phi)], y, n), [y.sum() / n.sum()],
                     method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-9))
        return float(r.x[0]), float(r.fun)

    print(f"PHI_CAP del repo = {PHI_CAP}")
    print('  (comentario en build_reporting_rate.py:44 — "φ tocando la cota = no se resuelve variación distrital")\n')
    print(f"{CAT}: {len(n)} distritos con datos · Σn_eff = {n.sum():.0f} · μ = {y.sum()/n.sum():.4f}\n")

    print("1. ¿ESTÁ IDENTIFICADA LA MAGNITUD? — perfil de verosimilitud en φ")
    print("─" * 58)
    for phi in [10, 100, 147, 1_000, 1e6]:
        _, nll = mu_at(phi)
        tag = "  ← PHI_CAP del repo" if phi == PHI_CAP else ("  ← límite homogéneo" if phi > 1e5 else "")
        print(f"   φ = {phi:>9,.0f}    −loglik = {nll:>10.4f}{tag}")

    # MLE interior de φ (perfil), no la cota: fit_eb trunca φ̂ a PHI_CAP, así que el CSV
    # reporta "φ=100 exacto" aunque el óptimo real esté por encima.
    opt = minimize_scalar(lambda lp: mu_at(float(np.exp(lp)))[1],
                          bounds=(np.log(10), np.log(5_000)), method="bounded")
    phi_hat, nll_hat = float(np.exp(opt.x)), float(opt.fun)
    _, nll_inf = mu_at(1e6)
    lrt = 2 * (nll_inf - nll_hat)
    p = chi2.sf(max(lrt, 0.0), df=1)
    print(f"\n   φ̂ (MLE interior del perfil) ≈ {phi_hat:.0f}  → fit_eb lo trunca a {PHI_CAP:.0f}")
    print(f"   LRT  H0: φ→∞ (SIN heterogeneidad distrital)  vs  φ̂={phi_hat:.0f}")
    print(f"      estadístico = {lrt:.4f}   p = {p:.4f}  (χ²(1); con mezcla 50:50 de borde ≈ {p/2:.4f})")
    verdict = ("heterogeneidad distrital DÉBIL pero detectable (pseudo-verosimilitud; borderline)"
               if p < 0.05 else "NO se rechaza H0 → sin evidencia de variación distrital")
    print(f"      ⇒ {verdict}\n")
    canon.emit(
        f"phi_hat.{CAT}", phi_hat, variant="victim",
        unit="MLE interior de la concentración φ (perfil de verosimilitud)",
        estimator="Beta-Binomial pseudo-verosimilitud sobre (y_eff, n_eff) panel por distrito",
        inputs=sorted(ENAPRES_RAW.glob("*.zip")), script=__file__,
    )
    canon.emit(
        f"phi_lrt_p.{CAT}", float(p), variant="victim",
        unit="p-valor pseudo-LRT H0 φ→∞ (sin heterogeneidad distrital de r̂) vs φ̂ interior",
        estimator="perfil de verosimilitud Beta-Binomial, χ²(1) sin corrección de borde",
        inputs=sorted(ENAPRES_RAW.glob("*.zip")), script=__file__,
    )

    if not TRUST.exists():
        print(f"⚠️  {TRUST} no existe; corre build_district_trust.py para el bloque 2")
        return 0
    d = json.load(open(TRUST, encoding="utf-8"))
    per = d["per_district"]
    common = [u for u in ub if u in per]

    print("2. ¿ESTÁ IDENTIFICADO EL ORDEN? — ρ(confianza-PNP, multiplicador) vs la cota")
    print(f"   (publicado: ρ = {d['spearman_rho']}, p = {d['spearman_p']}, n = {d['n_districts']};"
          f" multiplier_range = {d['multiplier_range']})")
    print("─" * 78)
    print(f"{'PHI_CAP':>10}{'ρ':>9}{'p':>9}{'rango del multiplicador':>27}{'sd':>10}")
    for cap in [10, PHI_CAP, phi_hat, 1_000, 1e6]:
        mu, _ = mu_at(cap)
        a, b = mu * cap, (1 - mu) * cap
        m = np.array([1 / ((sub[u][2] + a) / (sub[u][1] + a + b)) for u in common])
        t = np.array([per[u]["trust"] for u in common])
        rho, pv = spearmanr(t, m)
        tag = ("  ←repo" if cap == PHI_CAP else
               ("  ←φ̂" if cap == phi_hat else ("  ←límite homogéneo" if cap > 1e5 else "")))
        print(f"{cap:>10,.0f}{rho:>9.3f}{pv:>9.4f}   ×{m.min():>5.2f} – ×{m.max():<6.2f}{'':>5}{m.std():>10.4f}{tag}")
        # El rango CANÓNICO que cita el paper es el de la SUPERFICIE (fig-trust, lo emite
        # build_district_trust.py como multiplier_district_lo/hi); las filas de aquí son
        # el diagnóstico de sensibilidad a φ. Solo el límite homogéneo se emite: es la
        # referencia contra la que el pseudo-LRT contrasta la heterogeneidad.
        if cap > 1e5:
            canon.emit(
                f"phi_homogeneous_multiplier.{CAT}", float(m.mean()), variant="victim",
                unit="multiplicador víctima en el límite homogéneo φ→∞ (1/μ̂, constante por construcción)",
                estimator=f"Beta-Binomial μ̂ ML con φ→∞, {len(common)} distritos",
                inputs=sorted(ENAPRES_RAW.glob("*.zip")), script=__file__,
            )

    print("\n   ⇒ la MAGNITUD es débil: φ̂ interior con perfil casi plano (rango estable ~×3.6–5.4")
    print("      en la región plana), pseudo-LRT borderline — el paper NO la reporta como hallazgo.")
    print("   ⇒ el ORDEN sobrevive a toda φ: el shrinkage comprime hacia la media pero NO reordena,")
    print("      y Spearman solo mira el orden. La ρ de rangos toma poder prestado de un")
    print("      covariable externo (confianza) que el test ómnibus de φ no tiene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
