#!/usr/bin/env python3
"""
Idea 6 (slr-9wd9): cotas de identificación parcial para la tasa de denuncia
r = P(R=1|W=1) bajo misreporting de ENCUESTA, à la Fé 2024 (JQC) /
Kreider & Pepper 2007 — con el fraude (estafa) como caso ancla.

Ancla: limitación 3 del companion — el fraude se excluye del ranking por
escasez; Fé recomienda reportar regiones de identificación en vez de puntos
para categorías raras/estigmatizadas.

MARCO (notación de Fé 2024, mapeada a ENAPRES CAP_600):
  W = víctima real; X = declara victimización al encuestador (P601/P615);
  R = la PNP se enteró (P606/P619=1 & P607/P620=1); Z = declaración veraz
  (Z=1 sii X=W). El companion asume Z≡1 (X=W): r̂ = P(R=1,X=1)/P(X=1).
  Bajo el supuesto P(Z=1) ≥ ν (proporción mínima de respuestas veraces) y
  NO-OVER-REPORTING (nadie inventa victimización que además dice haber
  denunciado: P(R=1,X=1,Z=0)=0 → δ=0, ec. 9 de Fé), las cotas computables:

      LB(ν) = P(R=1,X=1) / (P(X=1) + (1−ν))
      UB(ν) = (P(R=1,X=1) + (1−ν)) / (P(X=1) + (1−ν))     [γ peor-caso = 1−ν]

  En ν=1 ambas colapsan al punto r̂. El ancho explota como (1−ν)/P(X=1):
  ESO es la formalización de por qué las categorías raras no soportan puntos —
  el mismo (1−ν) que apenas mueve al robo (P(X=1) grande) vuelve
  no-informativa a la estafa.

Probabilidades a nivel PERSONA (any-subtipo por categoría, ponderado FACTOR),
pooled 2019-2024, scopes nacional y Lima Metro (residencia 1501*/0701*).
Se reporta además ν* = mayor ν en la grilla con UB/LB ≥ 2 ("umbral de colapso
identificatorio") y la comparación con el ancho del IC muestral Wilson (la
incertidumbre de MUESTREO vs la de IDENTIFICACIÓN son de naturaleza distinta;
Imbens-Manski para combinarlas queda como extensión declarada).

Uso:  python3 scripts/eval_fraud_bounds.py
Salida: data/silver/analysis/fraud_bounds.json (regenerable, gitignored)
Reporte: analysis/fraud_bounds.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_reporting_rate import (  # noqa: E402
    CAT_CODES, ENAPRES_RAW, SCHEMAS, detect_schema, open_cap600, wilson_ci,
)

OUT_JSON = ROOT / "data/silver/analysis/fraud_bounds.json"
YEARS = range(2019, 2025)
NU_GRID = [1.0, 0.995, 0.99, 0.975, 0.95, 0.90]


def accumulate() -> dict:
    """Sumas ponderadas persona-nivel por categoría y scope."""
    # el denominador P(X=1) de cada categoría solo cuenta respondentes de los
    # años cuyo esquema define el subtipo (robo_hurto_callejero: solo 2024)
    acc = {scope: {cat: dict(w_x=0.0, w_rx=0.0, n_x=0, sw2_x=0.0, w_base=0.0)
                   for cat in CAT_CODES} | {"_all": dict(w=0.0, n=0)}
           for scope in ("nacional", "lima")}
    for year in YEARS:
        zpath = ENAPRES_RAW / f"{year}.zip"
        if not zpath.exists():
            continue
        factory, cols, _ = open_cap600(zpath)
        if not cols:
            continue
        idx = {c.upper(): i for i, c in enumerate(cols)}
        sch = detect_schema(cols)
        S = SCHEMAS[sch]
        i_w = idx.get("FACTOR")
        i_dd, i_pp = idx.get("CCDD"), idx.get("CCPP")

        def col(tpl, x):
            return idx.get(tpl.format(x=x).upper())

        for row in factory():
            if i_w is None or len(row) <= i_w:
                continue
            try:
                w = float(row[i_w].replace(",", "."))
            except (ValueError, AttributeError):
                continue
            if w <= 0:
                continue
            dd = (row[i_dd] or "").strip().zfill(2) if i_dd is not None else ""
            pp = (row[i_pp] or "").strip().zfill(2) if i_pp is not None else ""
            scopes = ["nacional"] + (["lima"] if (dd == "15" and pp == "01")
                                     or (dd == "07" and pp == "01") else [])
            for sc in scopes:
                acc[sc]["_all"]["w"] += w
                acc[sc]["_all"]["n"] += 1
            for cat, info in CAT_CODES.items():
                if not info[sch]:
                    continue  # el esquema de este año no define la categoría
                for sc in scopes:
                    acc[sc][cat]["w_base"] += w
                vict = pnp = False
                for x in info[sch]:
                    iv = col(S["vict"], x)
                    if iv is None or row[iv].strip() != "1":
                        continue
                    vict = True
                    jden, jdon = col(S["den"], x), col(S["donde"], x)
                    if (jden is not None and row[jden].strip() == "1"
                            and jdon is not None and row[jdon].strip() == "1"):
                        pnp = True
                if not vict:
                    continue
                for sc in scopes:
                    a = acc[sc][cat]
                    a["w_x"] += w
                    a["sw2_x"] += w * w
                    a["n_x"] += 1
                    if pnp:
                        a["w_rx"] += w
    return acc


def bounds_for(px: float, prx: float, nu: float) -> tuple[float, float]:
    lb = prx / (px + (1 - nu))
    ub = min(1.0, (prx + (1 - nu)) / (px + (1 - nu)))
    return lb, ub


def bounds_monotone(px: float, prx: float, nu: float) -> tuple[float, float]:
    """Variante con supuesto adicional de MONOTONICIDAD del reporte oculto:
    P(R=1, X=0) ≤ r̂·(1−ν) — las víctimas que se ocultan del encuestador
    denuncian a la PNP como máximo a la tasa de las que declaran. δ=0 (no
    over-reporting). Se plugea γ = r̂(1−ν) en la UB de la ec. 5 de Fé/K-P
    (válido porque la UB es creciente en γ cuando px−(1−ν) > 2·prx; si no,
    se degrada al peor caso)."""
    r_point = prx / px
    gamma = r_point * (1 - nu)
    lb = prx / (px + (1 - nu))
    den = px + 2 * gamma - (1 - nu)
    if den <= 0 or (px - (1 - nu)) <= 2 * prx:
        return bounds_for(px, prx, nu)
    ub = min(1.0, (prx + gamma) / den)
    return lb, max(ub, lb)


def main() -> int:
    acc = accumulate()
    results: dict = {"nu_grid": NU_GRID}
    for scope, cats in acc.items():
        out_scope: dict = {"n_respondents": cats["_all"]["n"]}
        for cat in CAT_CODES:
            a = cats[cat]
            if a["w_x"] <= 0 or a["w_base"] <= 0:
                continue
            px = a["w_x"] / a["w_base"]        # P(X=1) sobre años con el subtipo
            prx = a["w_rx"] / a["w_base"]      # P(R=1, X=1)
            r_point = prx / px
            n_eff = a["w_x"] ** 2 / a["sw2_x"] if a["sw2_x"] > 0 else 0
            lo_s, hi_s = wilson_ci(r_point, n_eff)
            entry = {
                "prevalence_pct": round(100 * px, 3),
                "r_hat_point": round(r_point, 4),
                "wilson_ci95_sampling": [round(lo_s, 4), round(hi_s, 4)],
                "n_victims": a["n_x"],
                "bounds": {},
            }
            nu_star = None
            entry["bounds_monotone"] = {}
            for nu in NU_GRID:
                lb, ub = bounds_for(px, prx, nu)
                entry["bounds"][str(nu)] = [round(lb, 4), round(ub, 4)]
                lbm, ubm = bounds_monotone(px, prx, nu)
                entry["bounds_monotone"][str(nu)] = [round(lbm, 4), round(ubm, 4)]
                if nu < 1 and lb > 0 and ub / lb >= 2 and nu_star is None:
                    nu_star = nu
            entry["nu_collapse_ratio2"] = nu_star  # mayor ν de la grilla con UB/LB≥2
            out_scope[cat] = entry
        results[scope] = out_scope

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Cifras portantes de §6.3 del companion → registro canónico (slr-x50g.6/v2).
    # La corrida pre-fix del crosswalk dejó un artefacto stale (prevalencia 8.9% con el
    # código 8 = intento de robo de bicicleta) que el paper hand-copió; con emits + anclas
    # ese drift queda bloqueado por check_canon.
    import canon
    inputs = sorted(ENAPRES_RAW.glob("*.zip"))
    for cat in ("robo_hurto_callejero", "estafa"):
        e = results.get("lima", {}).get(cat)
        if not e:
            continue
        kp = e["bounds"]["0.99"]
        canon.emit(f"kp_bound_lo.{cat}", float(kp[0]), variant="lima_nu99",
                   unit="cota inferior región identificación parcial de r̂ (ν=0.99, Lima pooled)",
                   estimator="Kreider-Pepper: declaraciones veraces ≥ν, sin sobre-reporte",
                   inputs=inputs, script=__file__)
        canon.emit(f"kp_bound_hi.{cat}", float(kp[1]), variant="lima_nu99",
                   unit="cota superior región identificación parcial de r̂ (ν=0.99, Lima pooled)",
                   estimator="Kreider-Pepper: declaraciones veraces ≥ν, sin sobre-reporte",
                   inputs=inputs, script=__file__)
        canon.emit(f"kp_prevalence.{cat}", float(e["prevalence_pct"]), variant="lima_pooled",
                   unit="% prevalencia declarada P(X=1), Lima pooled (denominador de la región KP)",
                   estimator="FACTOR-weighted, años con el subtipo disponible",
                   inputs=inputs, script=__file__)
        canon.emit(f"kp_point.{cat}", float(e["r_hat_point"]), variant="lima_pooled",
                   unit="r̂ directo pooled P(R|X) del universo KP (≠ EB μ de la superficie)",
                   estimator="Σw·RX/Σw·X, Lima pooled",
                   inputs=inputs, script=__file__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
