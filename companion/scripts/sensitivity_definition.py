#!/usr/bin/env python3
"""Sensibilidad de la DEFINICIÓN de robo: consumado-only vs +intentos vs 2024-only (§6.1).

MOTIVO (slr-x50g, empuje C1). La reverificación del 19-jul encontró un punto de
credibilidad que el paper no usaba: excluir los intentos del crosswalk de robo fue una
decisión **conservadora** — incluirlos baja r̂ y SUBE el multiplicador (~×6.6-7), así que
los autores tomaron el multiplicador menor pudiendo tomar el mayor. Eso preempta la
objeción "results-motivated". Pero citarlo exige computarlo con la construcción canónica
y emitirlo, no copiar el trío suelto del reporte de verificación (que usaba otra
construcción y por eso no cuadra dígito a dígito con el canon).

CONSTRUCCIÓN (idéntica al pipeline canónico, build_reporting_rate Peldaño 2):
pooling sw/sw2/swd por distrito de Lima sobre los años del alcance → eff() → fit_eb
(Beta-Binomial ML, φ truncado en PHI_CAP) → μ; multiplicador = Σ_d Y_d/r̂_d / Σ_d Y_d
con r̂_d = posterior EB por distrito e Y_d = denuncias robo pooled 2019–2024 de la
superficie (Y fijo entre definiciones: aísla el efecto de la definición).

Tres definiciones:
  consumado_panel  ["6A"] / [9]       2019–2024  ← la elegida (GATE: debe reproducir
                                                   r_pooled=0.224 y multiplicador 4.627)
  attempt_panel    ["6A","6B"] / [9,10] 2019–2024  (P601_6B existe en las 5 olas clásicas)
  consumado_2024   ["6A"] / [9]       solo 2024   (la base del ×5.5 de la versión previa)

Uso:  python3 scripts/sensitivity_definition.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402
import build_reporting_rate as brr  # noqa: E402

ROOT = canon.ROOT
LATENT = ROOT / "data/datasets/silver/latent_surface/latente_distrito_categoria_anio.csv"
CAT = "robo_hurto_callejero"
DEFS = {
    "consumado_panel": dict(clasico=["6A"], y2024=[9], years=list(range(2019, 2025))),
    "attempt_panel": dict(clasico=["6A", "6B"], y2024=[9, 10], years=list(range(2019, 2025))),
    "consumado_2024": dict(clasico=["6A"], y2024=[9], years=[2024]),
}
# gates de reproducción contra el canon (consumado_panel ES el pipeline canónico)
GATE_R, GATE_M, TOL = 0.224, 4.627, 0.02


def observed_pooled() -> dict:
    y = defaultdict(float)
    with open(LATENT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["categoria"] == CAT:
                y[r["ubigeo"]] += float(r["observado"])
    return dict(y)


def run_definition(clasico, y2024, years) -> dict:
    """Corre el pooling canónico con los códigos dados. Restaura CAT_CODES al salir."""
    saved = dict(brr.CAT_CODES[CAT])
    brr.CAT_CODES[CAT] = {"clasico": clasico, "2024": y2024, "panel": True}
    try:
        cells = brr.load_cells(years)
    finally:
        brr.CAT_CODES[CAT] = saved
    lima = [u for u in brr.lima_districts_canonical() if u.startswith("1501")]
    per_dist = {}
    for u in lima:
        agg = dict(sw=0.0, sw2=0.0, swd=0.0)
        for t in years:
            d = cells.get((t, CAT, u))
            if d:
                for k in agg:
                    agg[k] += d[k]
        p, n_eff, y_eff = brr.eff(agg)
        per_dist[u] = dict(p=p, n_eff=n_eff, y_eff=y_eff)
    fit_cells = [c for c in per_dist.values() if c["n_eff"] > 0]
    mu, phi = brr.fit_eb(fit_cells)
    total_neff = sum(c["n_eff"] for c in fit_cells)

    Y = observed_pooled()
    a0, b0 = mu * phi, (1 - mu) * phi
    num = den = 0.0
    for u, c in per_dist.items():
        if u not in Y:
            continue
        a = a0 + c["y_eff"]
        b = b0 + max(c["n_eff"] - c["y_eff"], 0)
        r_hat = a / (a + b)
        num += Y[u] / r_hat
        den += Y[u]
    return dict(mu=mu, phi=phi, total_neff=total_neff, multiplier=num / den)


def main() -> int:
    results = {}
    for name, spec in DEFS.items():
        r = run_definition(spec["clasico"], spec["y2024"], spec["years"])
        results[name] = r
        print(f"{name:<16} μ={r['mu']:.4f}  φ={r['phi']:.0f}  Σn_eff={r['total_neff']:.0f}"
              f"  multiplicador=×{r['multiplier']:.2f}")

    # GATE 1: la definición elegida reproduce el canon al dígito
    base = results["consumado_panel"]
    if abs(base["mu"] - GATE_R) > TOL or abs(base["multiplier"] - GATE_M) > TOL * 2:
        raise SystemExit(f"consumado_panel NO reproduce el canon "
                         f"(μ={base['mu']:.4f} vs {GATE_R}, M={base['multiplier']:.3f} vs {GATE_M})")
    # GATE 2: los códigos de intento efectivamente ampliaron el universo de víctimas
    if results["attempt_panel"]["total_neff"] <= base["total_neff"] * 1.05:
        raise SystemExit("attempt_panel no amplió n_eff — los códigos de intento no matchearon")
    print("\ngates OK: consumado_panel reproduce el canon; attempt amplía el universo.")

    inputs = sorted(brr.ENAPRES_RAW.glob("*.zip")) + [LATENT]
    for name, r in results.items():
        canon.emit(f"r_definition.{CAT}", r["mu"], variant=name,
                   unit="r̂ pooled (EB μ víctima) bajo la definición dada de robo",
                   estimator="pooling canónico Peldaño 2, PHI_CAP vigente",
                   inputs=inputs, script=__file__)
        canon.emit(f"multiplier_definition.{CAT}", r["multiplier"], variant=name,
                   unit="multiplicador Σλ*/Σy con Y fijo (denuncias robo pooled 19-24)",
                   estimator="EB por distrito bajo la definición dada; Y invariante",
                   inputs=inputs, script=__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
