#!/usr/bin/env python3
"""Robustez no-pandémica de las tasas de reporte r̂ (§3.1 del companion).

MOTIVO (reverificación v2, hallazgo #1). El párrafo COVID de §3.1 citaba shifts de r̂
(+2.3pp extorsión, +1.6pp VF) sin código versionado, y remataba con claims composicionales
del encuadre pre-reescritura ("the latent surface still corrects this distortion") que
contradicen el resultado negativo de §5.4 y usan la construcción de olas desiguales
repudiada (clase B1). Este script computa la ÚNICA parte defendible del check —cuánto se
mueve el r̂ EB pooled por categoría al excluir los años pandémicos (2020, 2021)— con la
construcción canónica, y lo emite.

Alcance por categoría: el pooling usa los años con datos de Lima disponibles dentro del
scope; robo tiene las 6 olas (panel completo), las demás categorías carecen de geografía
de ocurrencia en la ola 2024, así que su no-pandémico efectivo es {2019, 2022, 2023}.
El paper lo declara en vez de decir "4 of 6 waves" para todas.

Uso:  python3 scripts/eval_covid_robustness_rates.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402
import build_reporting_rate as brr  # noqa: E402

CATS = ["robo_hurto_callejero", "extorsion", "secuestro",
        "violencia_familiar_sexual", "estafa"]
FULL = list(range(2019, 2025))
NONPAND = [2019, 2022, 2023, 2024]   # excluye 2020-2021; 2024 solo aporta a robo


def mu_for(cells, cat: str, years: list[int]) -> tuple[float, float]:
    lima = [u for u in brr.lima_districts_canonical() if u.startswith("1501")]
    per = []
    n_tot = 0.0
    for u in lima:
        agg = dict(sw=0.0, sw2=0.0, swd=0.0)
        for t in years:
            d = cells.get((t, cat, u))
            if d:
                for k in agg:
                    agg[k] += d[k]
        p, n_eff, y_eff = brr.eff(agg)
        if n_eff > 0:
            per.append(dict(n_eff=n_eff, y_eff=y_eff))
            n_tot += n_eff
    mu, _ = brr.fit_eb(per)
    return mu, n_tot


def main() -> int:
    cells = brr.load_cells(FULL)
    inputs = sorted(brr.ENAPRES_RAW.glob("*.zip"))
    print(f"{'categoría':<28}{'μ full':>8}{'μ no-pand':>10}{'Δ (pp)':>8}{'n_eff np':>10}")
    for cat in CATS:
        mu_f, _ = mu_for(cells, cat, FULL)
        mu_n, n_n = mu_for(cells, cat, NONPAND)
        delta_pp = 100 * (mu_n - mu_f)
        print(f"{cat:<28}{mu_f:8.4f}{mu_n:10.4f}{delta_pp:+8.1f}{n_n:10.0f}")
        canon.emit(
            f"r_nonpandemic_delta.{cat}", delta_pp, variant="victim",
            unit="Δ en puntos porcentuales del r̂ EB pooled al excluir 2020-2021",
            estimator="EB μ Lima pooled, scope no-pandémico {2019,2022,2023,2024}",
            inputs=inputs, script=__file__,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
