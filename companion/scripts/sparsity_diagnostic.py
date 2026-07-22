#!/usr/bin/env python3
"""Diagnóstico de sparsity del panel ENAPRES-Lima que decide Estimador A vs B (§4.3).

MOTIVO (slr-x50g.6). §4.3 del companion citaba "median cell size 7 (range 0–312), 1.9% of
cells ≥30" atribuido al nivel de 21 subtipos, y "~14" para el armonizado — números legacy
que NO reproducen bajo ninguna construcción del pipeline actual (armonizado por-año da
mediana ~2.9 / 6.4%≥30; armonizado pooled da ~7 / ~21%≥30). La CONCLUSIÓN operativa es
robusta a la construcción (todo queda lejísimos del umbral del 70% de la regla A/B), pero
las cifras citadas deben salir de un emisor, no de memoria.

Construcción canónica: celdas distrito×categoría del panel POOLED 2019–2024 (la unidad
sobre la que el EB efectivamente ajusta, misma agregación sw/sw2 que build_reporting_rate),
para las CINCO categorías del companion, Lima (prefijo 1501). n_eff = (Σw)²/Σw².

Uso:  python3 scripts/sparsity_diagnostic.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402
from build_reporting_rate import ENAPRES_RAW, eff, load_cells  # noqa: E402

YEARS = list(range(2019, 2025))
CATS = ["robo_hurto_callejero", "extorsion", "secuestro",
        "violencia_familiar_sexual", "estafa"]
RULE_PCT = 70.0    # regla pre-especificada: ≥70% de celdas con n_eff≥30 → Estimador A
RULE_N = 30


def main() -> int:
    cells = load_cells(YEARS)
    pooled = defaultdict(lambda: dict(sw=0.0, sw2=0.0, swd=0.0))
    for (y, c, u), d in cells.items():
        if c in CATS and isinstance(u, str) and u.startswith("1501"):
            for k in ("sw", "sw2", "swd"):
                pooled[(c, u)][k] += d[k]
    ne = np.array([eff(d)[1] for d in pooled.values()])
    med, hi = float(np.median(ne)), float(ne.max())
    pct30 = float(100 * np.mean(ne >= RULE_N))

    print(f"celdas distrito×categoría (pooled {YEARS[0]}–{YEARS[-1]}, {len(CATS)} cats, Lima): {len(ne)}")
    print(f"  mediana n_eff = {med:.1f}   máx = {hi:.0f}   %celdas con n_eff≥{RULE_N} = {pct30:.1f}%")
    verdict = "A (directo)" if pct30 >= RULE_PCT else "B (shrinkage jerárquico)"
    print(f"  regla ({RULE_PCT:.0f}% ≥ {RULE_N}) → Estimador {verdict}")

    inputs = sorted(ENAPRES_RAW.glob("*.zip"))
    canon.emit("sparsity_median_neff.lima", med, variant="pooled_dc",
               unit="mediana de n_eff por celda distrito×categoría, panel pooled",
               estimator="(Σw)²/Σw² por celda, 5 categorías companion",
               inputs=inputs, script=__file__)
    canon.emit("sparsity_pct30.lima", pct30, variant="pooled_dc",
               unit="% de celdas distrito×categoría con n_eff≥30",
               estimator="misma construcción; umbral de la regla A/B",
               inputs=inputs, script=__file__)
    canon.emit("sparsity_neff_max.lima", hi, variant="pooled_dc",
               unit="n_eff máximo por celda (cota superior del rango)",
               estimator="misma construcción",
               inputs=inputs, script=__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
