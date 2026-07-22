#!/usr/bin/env python3
"""Confianza distrital en la PNP desde ENAPRES CAP_600 vs multiplicador de cifra negra.

Resuelve la contradicción §3.3-vs-§5.3 del companion (bead slr-srwe): §5.3 reportaba
un rho=-0.41 confianza-PNP × multiplicador que NINGÚN script computaba y que no podía
existir con el Observatorio (solo departamental). Pero ENAPRES CAP_600 SÍ tiene el
ítem individual `P616A_1` ("¿QUÉ TANTA CONFIANZA LE INSPIRA: La Policía Nacional?",
1=Nada 2=Poca 3=Suficiente 4=Bastante, 5=No sabe), agregable a distrito de RESIDENCIA
(CCDD+CCPP+CCDI) ponderado por FACTOR. Solo presente 2022-2023 con ese código.

Score de confianza distrital (default): % ponderado FACTOR con confianza >= Suficiente
  trust(d) = Σ FACTOR·1[P616A_1∈{3,4}] / Σ FACTOR·1[P616A_1∈{1,2,3,4}]
Se cruza (Spearman) con el multiplicador distrital real de robo callejero:
  mult(d) = Σ_t latente(d,robo,t) / Σ_t observado(d,robo,t)   (pooled 2019-2024)
desde data/silver/latent_surface/latente_distrito_categoria_anio.csv (factor_subrep).

Uso:  python3 scripts/build_district_trust.py
Requiere: numpy, scipy. Reporta el rho REAL (con su signo y p) — no el -0.41 inventado.
Salida: data/silver/analysis/district_trust_pnp.json (regenerable, gitignored).
"""
import csv
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
ENAPRES_RAW = ROOT / "data" / "datasets" / "enapres" / "raw"
LATENT_DIR = ROOT / "data" / "datasets" / "silver" / "latent_surface"
OUT_DIR = ROOT / "data" / "silver" / "analysis"

TRUST_YEARS = (2022, 2023)          # P616A_1 solo existe con ese código en 2022-2023
ROBBERY_CAT = "robo_hurto_callejero"
LIMA_PROV = "1501"                   # Lima Metropolitana (provincia), 43 distritos


def _open_cap600(year: int):
    """Devuelve (columnas, iterador de filas dict) del CAP_600 del año."""
    zp = ENAPRES_RAW / f"{year}.zip"
    z = zipfile.ZipFile(zp)
    inner = [n for n in z.namelist()
             if "CAP_600" in n.upper() and n.lower().endswith(".csv")]
    if not inner:
        raise FileNotFoundError(f"CAP_600 csv no hallado en {zp}")
    f = z.open(inner[0])
    tw = io.TextIOWrapper(f, encoding="latin-1")
    head = tw.readline().strip()
    sep = ";" if head.count(";") > head.count(",") else ","
    cols = head.split(sep)
    return cols, (dict(zip(cols, line.rstrip("\n").split(sep))) for line in tw)


def _factor(v: str) -> float:
    """FACTOR viene con decimal-coma español ('32,858...')."""
    try:
        return float(v.replace(",", "."))
    except (ValueError, AttributeError):
        return 0.0


def district_trust() -> dict:
    """trust(ubigeo) = share FACTOR-pond. con confianza >= Suficiente (P616A_1∈{3,4})."""
    num = {}   # Σ FACTOR·1[conf>=suficiente]
    den = {}   # Σ FACTOR·1[respondió 1..4]
    n_raw = {}
    for year in TRUST_YEARS:
        cols, rows = _open_cap600(year)
        need = {"CCDD", "CCPP", "CCDI", "FACTOR", "P616A_1"}
        missing = need - set(cols)
        if missing:
            raise KeyError(f"{year}: faltan columnas {missing}")
        for r in rows:
            if r.get("CCDD") != "15" or r.get("CCPP") != "01":
                continue  # solo Lima provincia (1501xx)
            ubigeo = f'{r["CCDD"]}{r["CCPP"]}{r["CCDI"]}'
            val = (r.get("P616A_1") or "").strip()
            if val not in {"1", "2", "3", "4"}:
                continue  # excluye 5=No sabe y vacíos
            w = _factor(r.get("FACTOR"))
            den[ubigeo] = den.get(ubigeo, 0.0) + w
            n_raw[ubigeo] = n_raw.get(ubigeo, 0) + 1
            if val in {"3", "4"}:
                num[ubigeo] = num.get(ubigeo, 0.0) + w
    return {u: {"trust": num.get(u, 0.0) / den[u], "n": n_raw[u]}
            for u in den if den[u] > 0}


def district_robbery_multiplier(latent: Path) -> dict:
    """mult(ubigeo) = Σ_t latente / Σ_t observado para robo callejero (pooled)."""
    lat, obs = {}, {}
    for r in csv.DictReader(open(latent, encoding="utf-8")):
        if r["categoria"] != ROBBERY_CAT:
            continue
        u = r["ubigeo"]
        if not u.startswith(LIMA_PROV):
            continue
        lat[u] = lat.get(u, 0.0) + float(r["latente"])
        obs[u] = obs.get(u, 0.0) + float(r["observado"])
    return {u: lat[u] / obs[u] for u in obs if obs[u] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=("victim", "incident"), default="victim")
    args = ap.parse_args()
    suffix = "" if args.unit == "victim" else "_incident"
    latent = LATENT_DIR / f"latente_distrito_categoria_anio{suffix}.csv"
    out = OUT_DIR / f"district_trust_pnp{suffix}.json"
    trust = district_trust()
    mult = district_robbery_multiplier(latent)
    common = sorted(set(trust) & set(mult))
    t = np.array([trust[u]["trust"] for u in common])
    m = np.array([mult[u] for u in common])
    rho, p = spearmanr(t, m)

    result = {
        "n_districts": len(common),
        "trust_var": "P616A_1", "trust_years": list(TRUST_YEARS),
        "trust_score": "share FACTOR-weighted P616A_1 in {Suficiente,Bastante}",
        "multiplier": f"pooled Σlatente/Σobservado, {ROBBERY_CAT}",
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 5),
        "trust_range": [round(float(t.min()), 4), round(float(t.max()), 4)],
        "multiplier_range": [round(float(m.min()), 2), round(float(m.max()), 2)],
        "per_district": {u: {"trust": round(trust[u]["trust"], 4),
                             "n_resp": trust[u]["n"],
                             "robbery_multiplier": round(mult[u], 2)}
                         for u in common},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)

    # Cifras portantes de §5.3/fig-trust del companion → registro canónico (solo la
    # variante víctima, que es la del paper). El rango distrital del multiplicador se
    # cita como descriptivo-de-la-superficie (su magnitud es sensible al shrinkage,
    # ver recheck_phi_cap.py); la ρ con confianza es el claim ordinal.
    if args.unit == "victim":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import canon
        inputs = sorted(ENAPRES_RAW.glob("*.zip")) + [latent]
        canon.emit(
            f"trust_spearman_rho.{ROBBERY_CAT}", float(rho), variant="victim",
            unit="Spearman ρ(confianza-PNP distrital, multiplicador robo de la superficie)",
            estimator=f"P616A_1 pooled 2022–2023 vs Σlat/Σobs pooled, {len(common)} distritos",
            inputs=inputs, script=__file__,
        )
        for fam, val in [("multiplier_district_lo", m.min()), ("multiplier_district_hi", m.max())]:
            canon.emit(
                f"{fam}.{ROBBERY_CAT}", float(val), variant="victim",
                unit="extremo del rango distrital del multiplicador víctima en la superficie",
                estimator=f"Σlatente/Σobservado por distrito, pooled 2019–2024, {len(common)} distritos",
                inputs=inputs, script=__file__,
            )

    print(f"Distritos con ambos datos: {len(common)} / 43")
    print(f"Confianza-PNP (P616A_1, 2022-2023, %≥Suficiente): "
          f"rango [{result['trust_range'][0]:.3f}, {result['trust_range'][1]:.3f}]")
    print(f"Multiplicador robo (pooled Σlat/Σobs): "
          f"rango [{result['multiplier_range'][0]:.2f}, {result['multiplier_range'][1]:.2f}]")
    print(f"\n>>> Spearman(confianza, multiplicador) = {rho:.3f}  (p = {p:.4f}, n = {len(common)})")
    print(f"    Escrito: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
