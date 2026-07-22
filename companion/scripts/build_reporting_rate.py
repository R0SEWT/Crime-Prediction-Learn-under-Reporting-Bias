#!/usr/bin/env python3
"""Tasa de reporte r(distrito, categoría, año) — estimador escalonado A→B (bead slr-86y).

r = P(la víctima denunció ante la PNP | victimizada en d,c,t). De-sesga las denuncias:
`latente ≈ observado / r` (§4.0). Arranque alineado al estado del arte (§11):

  Peldaño 1 (A, directo — Wheeler & Piquero 2025): r̂ a nivel categoría×año (nacional y
            Lima por OCURRENCIA), ponderado por FACTOR (diseño), con IC por n efectivo.
  Peldaño 2 (B, SAE — Whitworth 2021 / Stoner 2019): abrir a distrito×categoría×año con
            empirical-Bayes Beta-Binomial; shrinkage de celdas ralas hacia la media
            categoría×año (la dispersión lo exige: solo ~17% de celdas Lima con n≥30).

Pesos FACTOR vía **tamaño muestral efectivo** n_eff=(Σw)²/Σw² (design effect, V5).
Rejilla §4.0: granularidad {incidente|víctima} × equivalencia {solo-PNP|cualquier}.
El mapa distrital (Peldaño 2) puede usar la binomial a nivel VÍCTIMA (estructura
Bernoulli) o exposición a HECHOS; el modo incidente produce un artefacto separado.
Esquemas ENAPRES clásico (2019-2023) y rediseño 2024 mapeados por variable (ver SCHEMAS).

Uso:  python3 scripts/build_reporting_rate.py [--years 2019-2024] [--unit victim|incident]
Requiere: numpy, scipy. Reusa scripts/harmonize_taxonomy.py (open_cap600, CROSSWALK).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonize_taxonomy import CROSSWALK, ENAPRES_RAW, open_cap600  # noqa: E402
import canon  # noqa: E402  (registro de números canónicos, hardening)

ROOT = Path(__file__).resolve().parent.parent
SILVER = ROOT / "data" / "datasets" / "silver" / "reporting_rate"

# Cota superior débilmente-informativa sobre la concentración φ. Evita la falsa precisión:
# sin ella, bajo dispersión extrema el MLE manda φ→∞ (prior domina) y los IC por distrito
# colapsan a bandas irrealmente angostas. φ≤100 ⇒ shrinkage fuerte pero IC honestos
# (sd mínima ≈ √(μ(1-μ)/101)). φ tocando la cota = "no se resuelve variación distrital".
PHI_CAP = 100.0

# Mapa de variables por esquema (verificado en diccionarios CAP600 2022 y 2024).
SCHEMAS = {
    "clasico": dict(  # 2019-2023
        vict="P601_{x}", times="P602_{x}", den_any="P602E_{x}",
        pnp_cnt="P602F_{x}_1_ENT", den="P606_{x}", donde="P607_{x}",
        occ=("P602B_{x}", "P602C_{x}", "P602D_{x}_COD"),
    ),
    "2024": dict(
        vict="P615_{x}", times="P616_{x}", den_any="P617_{x}",
        pnp_cnt="P618_{x}_1_ENT", den="P619_{x}", donde="P620_{x}",
        occ=("P623_{x}_COD", "P624_{x}_COD", "P625_{x}_COD"),
    ),
}
# códigos por categoría (del crosswalk canónico)
CAT_CODES = {c[0]: {"clasico": c[2], "2024": c[3], "panel": c[4]} for c in CROSSWALK}


def _int(s):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None


def detect_schema(cols) -> str:
    return "2024" if any(c.upper().startswith("P615_") for c in cols) else "clasico"


def load_cells(years):
    """Acumula por (anio, categoria, ubigeo_ocurrencia) las sumas ponderadas necesarias.

    Cada (persona × subtipo victimizado) es un registro. d_pnp = denunció ante PNP
    (víctima-nivel: den==1 & donde==1). Devuelve dict cell→stats y otro a nivel
    categoría×año (sin geo) para el directo nacional.
    """
    cells = defaultdict(lambda: dict(
        sw=0.0, sw2=0.0, swd=0.0,  # víctima-nivel (Bernoulli)
        w_times=0.0, w_times2=0.0, w_pnp=0.0, w_anyc=0.0, n=0,  # incidente-nivel
    ))
    for year in years:
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
        ncol = len(cols)

        def col(tpl, x):
            return idx.get(tpl.format(x=x).upper())

        for row in factory():
            if len(row) < ncol:
                continue
            w = None
            if i_w is not None:
                try:
                    w = float(row[i_w].replace(",", "."))
                except (ValueError, AttributeError):
                    w = None
            if not w or w <= 0:
                continue
            for cat, info in CAT_CODES.items():
                for x in info[sch]:
                    iv = col(S["vict"], x)
                    if iv is None or row[iv].strip() != "1":
                        continue  # no víctima de este subtipo
                    # víctima-nivel
                    jden, jdon = col(S["den"], x), col(S["donde"], x)
                    den = jden is not None and row[jden].strip() == "1"
                    pnp = den and jdon is not None and row[jdon].strip() == "1"
                    # ocurrencia
                    od, op, odi = (col(t, x) for t in S["occ"])
                    dep = _int(row[od]) if od is not None else None
                    prov = _int(row[op]) if op is not None else None
                    dist = _int(row[odi]) if odi is not None else None
                    if None in (dep, prov, dist):
                        ubi = None
                    else:
                        ubi = f"{dep:02d}{prov:02d}{dist:02d}"
                    # incidente-nivel
                    jt, jp, ja = col(S["times"], x), col(S["pnp_cnt"], x), col(S["den_any"], x)
                    times = _int(row[jt]) if jt is not None else None
                    times = times if (times and 0 < times <= 50) else 1
                    pnp_cnt = _int(row[jp]) if jp is not None else (1 if pnp else 0)
                    pnp_cnt = min(pnp_cnt or 0, times)
                    anyc = _int(row[ja]) if ja is not None else None
                    anyc = min(anyc, times) if (anyc is not None) else (1 if den else 0)

                    for key in ((year, cat, ubi), (year, cat, "_NAC")):
                        if key[2] is None:
                            continue
                        d = cells[key]
                        d["sw"] += w
                        d["sw2"] += w * w
                        d["swd"] += w * pnp
                        incident_weight = w * times
                        d["w_times"] += incident_weight
                        d["w_times2"] += incident_weight * incident_weight
                        d["w_pnp"] += w * pnp_cnt
                        d["w_anyc"] += w * anyc
                        d["n"] += 1
    return cells


def wilson_ci(p, n_eff, z=1.96):
    if n_eff <= 0:
        return (0.0, 1.0)
    d = 1 + z * z / n_eff
    c = p + z * z / (2 * n_eff)
    h = z * np.sqrt(p * (1 - p) / n_eff + z * z / (4 * n_eff * n_eff))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def betabinom_nll(params, y, n):
    """-loglik Beta-Binomial reparam (mu, phi) sobre celdas (y_eff, n_eff)."""
    mu, log_phi = params
    if not (1e-4 < mu < 1 - 1e-4):
        return 1e12
    phi = np.exp(log_phi)
    a, b = mu * phi, (1 - mu) * phi
    ll = (betaln(a + y, b + n - y) - betaln(a, b)).sum()
    return -ll


def fit_eb(cells_cy):
    """Ajusta (mu, phi) Beta-Binomial por categoría×año (o pooled) y devuelve posterior."""
    y = np.array([c["y_eff"] for c in cells_cy])
    n = np.array([c["n_eff"] for c in cells_cy])
    keep = n > 0
    y, n = y[keep], n[keep]
    if len(n) < 3 or y.sum() == 0:
        mu = (y.sum() / n.sum()) if n.sum() > 0 else 0.0
        return mu, 1.0  # sin info para shrinkage → prior débil
    p0 = [min(max(y.sum() / n.sum(), 1e-3), 1 - 1e-3), np.log(5.0)]
    res = minimize(betabinom_nll, p0, args=(y, n), method="Nelder-Mead",
                   options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
    mu, phi = res.x[0], float(np.exp(res.x[1]))
    return mu, min(phi, PHI_CAP)


def lima_districts_canonical():
    """43 distritos de la prov. Lima (1501xx) desde el silver INEI; [] si no existe."""
    p = ROOT / "data" / "datasets" / "silver" / "inei_poblacion" / "poblacion_distrital_2018_2026.csv"
    if not p.exists():
        return []
    s = set()
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ubigeo"].startswith("1501"):
                s.add(r["ubigeo"])
    return sorted(s)


def eff(d):
    sw, sw2 = d["sw"], d["sw2"]
    n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
    p = (d["swd"] / sw) if sw > 0 else 0.0
    return p, n_eff, p * n_eff


def eff_incident(d):
    """Razón PNP/hechos y tamaño efectivo con exposición ponderada por hechos.

    Cada registro ENAPRES aporta ``times`` hechos con peso FACTOR. El pseudo-tamaño
    efectivo conserva el ajuste de diseño usado por el EB de víctimas, pero sobre la
    exposición a hechos. Es el denominador dimensionalmente compatible con un conteo
    policial de denuncias de hechos.
    """
    exposure, exposure2 = d["w_times"], d["w_times2"]
    n_eff = (exposure * exposure / exposure2) if exposure2 > 0 else 0.0
    p = (d["w_pnp"] / exposure) if exposure > 0 else 0.0
    return p, n_eff, p * n_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019-2024")
    ap.add_argument(
        "--unit", choices=("victim", "incident"), default="victim",
        help="unidad del EB distrital; victim preserva el artefacto canónico actual",
    )
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))
    cells = load_cells(years)
    SILVER.mkdir(parents=True, exist_ok=True)

    # ---------- Peldaño 1: directo categoría×año (nacional y Lima por ocurrencia) ----------
    print("=== PELDAÑO 1 — tasa directa r̂ (víctima-nivel, ponderada FACTOR) ===")
    print(f"{'categoria':<26}{'año':>5}{'ámbito':>8}{'r_pnp':>8}{'IC95':>16}{'r_inc':>8}{'n':>7}")
    direct_rows = []
    for cat in CAT_CODES:
        for t in years:
            for amb, ubis in (("NAC", ["_NAC"]),
                              ("LIMA", [u for (yy, cc, u) in cells
                                       if yy == t and cc == cat and isinstance(u, str)
                                       and u.startswith("1501")])):
                agg = dict(
                    sw=0.0, sw2=0.0, swd=0.0, w_times=0.0, w_times2=0.0,
                    w_pnp=0.0, w_anyc=0.0, n=0,
                )
                for u in ubis:
                    d = cells.get((t, cat, u))
                    if not d:
                        continue
                    for k in agg:
                        agg[k] += d[k]
                if agg["n"] == 0:
                    continue
                p, n_eff, y_eff = eff(agg)
                lo_ci, hi_ci = wilson_ci(p, n_eff)
                r_inc = (agg["w_pnp"] / agg["w_times"]) if agg["w_times"] > 0 else float("nan")
                direct_rows.append((cat, t, amb, round(p, 4), round(lo_ci, 4),
                                    round(hi_ci, 4), round(r_inc, 4), round(n_eff, 1), agg["n"]))
                if amb == "LIMA":
                    print(f"{cat:<26}{t:>5}{amb:>8}{p:>8.3f}  [{lo_ci:.2f},{hi_ci:.2f}]"
                          f"{r_inc:>8.3f}{agg['n']:>7}")
    with open(SILVER / "r_directo_categoria_anio.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "anio", "ambito", "r_pnp_victima", "ic_low", "ic_high",
                    "r_pnp_incidente", "n_eff", "n_victimas"])
        w.writerows(direct_rows)

    # ---------- diagnóstico de dispersión distrital (motiva B) ----------
    lima_cells = [(k, cells[k]) for k in cells if isinstance(k[2], str) and k[2].startswith("1501")]
    n_effs = np.array([eff(d)[1] for _, d in lima_cells])
    print(f"\nDispersión Lima (distrito×categoría×año): celdas={len(n_effs)}  "
          f"%n_eff≥10={100*np.mean(n_effs >= 10):.1f}  %≥30={100*np.mean(n_effs >= 30):.1f}"
          f"  → justifica Peldaño 2 (shrinkage)")

    # ---------- Peldaño 2: Beta-Binomial EB por categoría (pooled años para panel) ----------
    unit_eff = eff if args.unit == "victim" else eff_incident
    print("\n=== PELDAÑO 2 — Beta-Binomial EB: "
          f"r̂(distrito,categoría) a nivel de {'víctima' if args.unit == 'victim' else 'hecho'} ===")
    LIMA_DISTRICTS = lima_districts_canonical() or sorted(
        {k[2] for k in cells if isinstance(k[2], str) and k[2].startswith("1501")})
    eb_rows = []
    for cat, info in CAT_CODES.items():
        # panel: pooled 2019-2024; 2024-only para robo_callejero (sin micro previo)
        scope_years = years if info["panel"] else [2024]
        per_dist = {}
        for u in LIMA_DISTRICTS:
            agg = dict(sw=0.0, sw2=0.0, swd=0.0, w_times=0.0, w_times2=0.0, w_pnp=0.0)
            for t in scope_years:
                d = cells.get((t, cat, u))
                if d:
                    for k in agg:
                        agg[k] += d[k]
            p, n_eff, y_eff = unit_eff(agg)
            per_dist[u] = dict(p=p, n_eff=n_eff, y_eff=y_eff)
        fit_cells = [per_dist[u] for u in LIMA_DISTRICTS if per_dist[u]["n_eff"] > 0]
        if not fit_cells:
            continue
        mu, phi = fit_eb(fit_cells)
        if mu <= 0:
            # La variable de conteo PNP no está definida/utilizable para este universo
            # (actualmente secuestro). No inventar una tasa positiva vía el prior.
            for u in LIMA_DISTRICTS:
                c = per_dist[u]
                eb_rows.append((u, cat, "panel" if info["panel"] else "2024",
                                0.0, 0.0, 0.0, round(c["n_eff"], 1), 0.0, round(phi, 1)))
            print(f"  {cat:<26} μ=0.000 φ={phi:6.1f}  sin hechos PNP observables; excluible")
            continue
        a0, b0 = mu * phi, (1 - mu) * phi
        for u in LIMA_DISTRICTS:
            c = per_dist[u]
            a = a0 + c["y_eff"]
            b = b0 + max(c["n_eff"] - c["y_eff"], 0)
            r_hat = a / (a + b)
            lo_ci, hi_ci = beta_dist.ppf([0.025, 0.975], a, b)
            eb_rows.append((u, cat, "panel" if info["panel"] else "2024",
                            round(r_hat, 4), round(float(lo_ci), 4), round(float(hi_ci), 4),
                            round(c["n_eff"], 1), round(mu, 4), round(phi, 1)))
        print(f"  {cat:<26} μ={mu:.3f} φ={phi:6.1f}  scope={'panel' if info['panel'] else '2024'}"
              f"  distritos={len(per_dist)}")
        # Emite la tasa de reporte pooled al registro canónico (hardening anti-stale).
        canon.emit(
            f"r_pooled.{cat}", mu, variant=args.unit,
            unit="P(víctima denunció ante PNP), pooled",
            estimator=f"Beta-Binomial EB μ (pooled {'panel' if info['panel'] else '2024'}), {args.unit}-level",
            inputs=sorted(ENAPRES_RAW.glob("*.zip")), script=__file__,
        )
    eb_dest = SILVER / (
        "r_distrital_eb.csv" if args.unit == "victim" else "r_distrital_eb_incident.csv"
    )
    with open(eb_dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ubigeo", "categoria", "scope", "r_hat", "ic_low", "ic_high",
                    "n_eff", "mu_cat", "phi_cat"])
        w.writerows(eb_rows)

    print(f"\n→ {SILVER}/r_directo_categoria_anio.csv  ({len(direct_rows)} filas)")
    print(f"→ {eb_dest}  ({len(eb_rows)} filas, {len(LIMA_DISTRICTS)} distritos Lima)")
    print(f"OK: r {args.unit}-nivel listo (Peldaño 1 directo + Peldaño 2 shrinkage).")


if __name__ == "__main__":
    main()
