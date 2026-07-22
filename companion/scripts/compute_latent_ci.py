#!/usr/bin/env python3
"""B10: computa DE VERDAD la propagación de incertidumbre que §4.5 del companion describe.

MOTIVO (slr-x50g.2). El paper describía un delta method (§4.5, con fórmula explícita) y el
abstract prometía "approximate credible intervals propagated from the reporting-rate
posterior" — pero NADA de eso se computaba en el repo (bloqueante B10 de la revisión
pre-DOI). Decisión editorial 2026-07-20 (Rody): computarlo, no retirar el claim.

QUÉ COMPUTA, por categoría y a nivel víctima (más la variante incidente para robo):

1. **Per-celda (delta method, la fórmula de §4.5).** Reconstruye el posterior Beta(a,b) de
   cada celda distrito×categoría desde `r_distrital_eb.csv` (a+b = φ+n_eff, a = r̂·(a+b) —
   validado contra los ic_low/ic_high del propio CSV) y propaga
   Var(λ̂*) ≈ y²/r̂⁴·Var(r̂) al conteo latente distrital. Artefacto por distrito.
2. **Multiplicador pooled (Monte Carlo).** El multiplicador Σλ*/Σy es un cociente de sumas
   sobre posteriors distritales; su CI se obtiene muestreando r̂_d ~ Beta(a_d, b_d)
   (independientes, μ y φ fijos) y tomando percentiles 2.5/97.5 de M = Σ_d Y_d/r̂_d / Σ_d Y_d.
   Semilla fija → reproducible. Los extremos se emiten al registro canónico.

VERIFICACIÓN INTERNA (aborta si falla):
- Los CIs Beta reconstruidos reproducen los ic_low/ic_high del CSV (tolerancia de redondeo).
- El punto M con r̂_d del CSV reproduce el multiplicador canónico (×4.627 víctima robo).

QUÉ CONDICIONA los intervalos (declarado en §4.5, no esconderlo): son condicionales al
modelo EB — (μ, φ) del prior se tratan como fijos (subestima incertidumbre, caveat clásico
de EB), φ está truncado en PHI_CAP=100 (heterogeneidad débilmente identificada, §5.3), la
λ imputada no aporta término Poisson propio, y los n_eff son tamaños efectivos de diseño.
No cubren la composición de olas ni el universe-mismatch: de eso se ocupan §4.2 y §6.

Uso:  python3 scripts/compute_latent_ci.py [--samples 20000] [--seed 0]
Salida: data/silver/analysis/latent_ci.json + emits multiplier_ci_lo/hi al canon.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402

ROOT = canon.ROOT
EB = {
    "victim": ROOT / "data/datasets/silver/reporting_rate/r_distrital_eb.csv",
    "incident": ROOT / "data/datasets/silver/reporting_rate/r_distrital_eb_incident.csv",
}
LATENT = {
    "victim": ROOT / "data/datasets/silver/latent_surface/latente_distrito_categoria_anio.csv",
    "incident": ROOT / "data/datasets/silver/latent_surface/latente_distrito_categoria_anio_incident.csv",
}
DEST = ROOT / "data/silver/analysis/latent_ci.json"
# Víctima = estimando canónico (todas las categorías del companion); incidente solo robo,
# porque es la única categoría cuyo ×15.8 el paper cita (secuestro incidente es r̂=0, patología).
CATS_BY_UNIT = {
    "victim": ["robo_hurto_callejero", "extorsion", "secuestro",
               "violencia_familiar_sexual", "estafa"],
    "incident": ["robo_hurto_callejero"],
}
HEADLINE = "robo_hurto_callejero"


def load_eb(path: Path) -> dict:
    """(categoria, ubigeo) → dict con posterior Beta reconstruido y validado."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r_hat, n_eff, phi = float(r["r_hat"]), float(r["n_eff"]), float(r["phi_cat"])
            if r_hat <= 0:
                continue  # celdas sin tasa utilizable (p.ej. secuestro incidente)
            ab = phi + n_eff
            a, b = r_hat * ab, (1 - r_hat) * ab
            out[(r["categoria"], r["ubigeo"])] = dict(
                a=a, b=b, r_hat=r_hat,
                ic_low=float(r["ic_low"]), ic_high=float(r["ic_high"]),
            )
    return out


def validate_reconstruction(cells: dict, tol: float = 0.006) -> float:
    """Los (a,b) reconstruidos deben reproducir los CIs Beta que el CSV ya trae."""
    errs = []
    for c in cells.values():
        lo, hi = beta_dist.ppf([0.025, 0.975], c["a"], c["b"])
        errs.append(max(abs(lo - c["ic_low"]), abs(hi - c["ic_high"])))
    worst = float(max(errs))
    if worst > tol:
        raise SystemExit(f"reconstrucción Beta NO reproduce los CIs del CSV (peor error {worst:.4f} > {tol})")
    return worst


def load_observed(path: Path) -> dict:
    """(categoria, ubigeo) → Σ_t observado (y pooled)."""
    y = defaultdict(float)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y[(r["categoria"], r["ubigeo"])] += float(r["observado"])
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200_000)  # 20k dejaba el extremo inferior del CI incidente en filo de redondeo (15.35↔15.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    result = {
        "_generated_by": "scripts/compute_latent_ci.py",
        "_method": {
            "per_cell": "delta: Var(λ*)=y²/r̂⁴·Var(r̂), Var del posterior Beta(a,b) reconstruido",
            "multiplier": f"MC {args.samples} muestras r̂_d~Beta(a_d,b_d), percentiles 2.5/97.5 de Σy_d/r̂_d/Σy_d",
            "seed": args.seed,
            "conditioning": "μ,φ fijos (EB), φ truncado en cap=100, sin término Poisson de λ, n_eff de diseño",
        },
        "units": {},
    }

    for unit, cats in CATS_BY_UNIT.items():
        cells = load_eb(EB[unit])
        worst = validate_reconstruction(cells)
        y = load_observed(LATENT[unit])
        print(f"— unidad {unit}: reconstrucción Beta OK (peor desvío de CI {worst:.4f})")
        result["units"][unit] = {}
        for cat in cats:
            ds = sorted(u for (c, u) in cells if c == cat and y.get((cat, u), 0.0) > 0)
            if not ds:
                print(f"  {cat:<26} sin celdas utilizables — omitida")
                continue
            Y = np.array([y[(cat, u)] for u in ds])
            A = np.array([cells[(cat, u)]["a"] for u in ds])
            B = np.array([cells[(cat, u)]["b"] for u in ds])
            R = np.array([cells[(cat, u)]["r_hat"] for u in ds])

            # punto: debe reproducir el multiplicador de la superficie
            m_point = float((Y / R).sum() / Y.sum())

            # MC del multiplicador pooled
            draws = rng.beta(A[None, :].repeat(args.samples, 0), B[None, :].repeat(args.samples, 0))
            m_mc = (Y[None, :] / draws).sum(axis=1) / Y.sum()
            lo, hi = np.percentile(m_mc, [2.5, 97.5])

            # per-celda (delta): CI del conteo latente distrital
            var_r = A * B / ((A + B) ** 2 * (A + B + 1))
            lam = Y / R
            sd_lam = np.sqrt(Y**2 / R**4 * var_r)
            per_d = {u: dict(lambda_star=round(float(l), 1),
                             sd_delta=round(float(s), 1),
                             r_hat=round(float(r), 4))
                     for u, l, s, r in zip(ds, lam, sd_lam, R)}

            # Near-floor: con r̂ diminuto la inversión 1/r̂ es tan convexa que el MC queda
            # entero por encima del punto plug-in (Jensen). No es bug: cuantifica que el
            # multiplicador de esas categorías no es establemente invertible — otra cara
            # de por qué la superficie validada se restringe a robo.
            unstable = bool(lo > m_point)
            result["units"][unit][cat] = dict(
                n_districts=len(ds), multiplier_point=round(m_point, 4),
                multiplier_ci95=[round(float(lo), 4), round(float(hi), 4)],
                unstable_inversion=unstable,
                per_district=per_d,
            )
            flag = "  [INESTABLE: punto fuera del CI, near-floor]" if unstable else ""
            print(f"  {cat:<26} M={m_point:6.3f}  CI95=[{lo:.3f}, {hi:.3f}]  ({len(ds)} distritos){flag}")

            if cat == HEADLINE:
                # verificación dura contra el canónico antes de emitir
                fam = "multiplier.robo_hurto_callejero"
                reg = canon.load()["entries"].get(f"{fam}.{unit}")
                if reg and abs(m_point - reg["value"]) > 0.02:
                    raise SystemExit(
                        f"M punto {unit} = {m_point:.4f} NO reproduce el canónico "
                        f"{reg['value']:.4f} — revisar antes de emitir CIs")
                for suf, val in [("lo", lo), ("hi", hi)]:
                    canon.emit(
                        f"multiplier_ci_{suf}.{HEADLINE}", float(val), variant=unit,
                        unit=f"extremo IC95 MC del multiplicador pooled ({unit}-level)",
                        estimator=f"MC {args.samples} sobre posteriors Beta EB, seed={args.seed}",
                        inputs=[EB[unit], LATENT[unit]], script=__file__,
                    )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n→ {DEST.relative_to(ROOT)}")
    print(f"→ {canon.REGISTRY.relative_to(ROOT)} (multiplier_ci_lo/hi.{HEADLINE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
