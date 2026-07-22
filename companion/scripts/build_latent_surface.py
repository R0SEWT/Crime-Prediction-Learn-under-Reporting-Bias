#!/usr/bin/env python3
"""Corrección observado → latente (bead slr-0rn).

`latente(d,c,t) = denuncias_observadas(d,c,t) / r̂(d,c)` — de-sesga las denuncias
MININTER con la tasa de reporte de slr-86y, propagando la incertidumbre de `r`.
Salida = **contrato de interfaz Etapa1→Etapa2** (§1): distrito×categoría×año →
(riesgo_latente, incertidumbre, r̂, factor de sub-representación).

Categorías de-sesgables = intersección {numerador MININTER P_MODALIDADES} ∩
{denominador con r en ENAPRES}: estafa, extorsion, secuestro,
violencia_familiar_sexual, robo_hurto_callejero (Robo+Hurto). "Otros" MININTER no
tiene r → fuera. El modo ``incident`` puede excluir categorías cuyo r̂ por hecho es
nulo/no definido, en vez de emitir latentes infinitos.

Uso:  python3 scripts/build_latent_surface.py [--unit victim|incident]
Requiere: stdlib. Lee silver de slr-86y (reporting_rate) + INEI + MININTER CSV.
"""
import csv
import argparse
import math
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # registro de números canónicos (hardening)

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "datasets"
SILVER = DS / "silver" / "latent_surface"
MININTER = DS / "DATASET_Denuncias_Policiales_Ene 2018 a Mayo 2026.csv"
R_EB_DIR = DS / "silver" / "reporting_rate"
POB = DS / "silver" / "inei_poblacion" / "poblacion_distrital_2018_2026.csv"
YEARS = range(2018, 2025)
R_UNSTABLE = 0.05  # r̂ por debajo de esto → punto inestable (cifra negra extrema)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.strip()


# MININTER P_MODALIDADES (normalizado) → categoría de-sesgo
MIN2CAT = {
    "estafa": "estafa",
    "extorsion": "extorsion",
    "secuestro": "secuestro",
    "violencia contra la mujer e integrantes": "violencia_familiar_sexual",
    "robo": "robo_hurto_callejero",
    "hurto": "robo_hurto_callejero",
}


def load_observed():
    obs = defaultdict(int)  # (ubigeo, cat, year) -> denuncias
    with open(MININTER, encoding="utf-8") as f:  # el CSV público es UTF-8 (¡no latin-1!)
        for r in csv.DictReader(f):
            u = r["UBIGEO_HECHO"].strip().zfill(6)
            if not u.startswith("1501"):
                continue
            y = int(r["ANIO"])
            if y not in YEARS:
                continue
            cat = MIN2CAT.get(norm(r["P_MODALIDADES"]))
            if cat:
                obs[(u, cat, y)] += int(r["cantidad"])
    return obs


def load_r(path: Path):
    r = {}  # (ubigeo, cat) -> (r_hat, ic_low, ic_high, scope)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r[(row["ubigeo"], row["categoria"])] = (
                float(row["r_hat"]), float(row["ic_low"]), float(row["ic_high"]), row["scope"])
    return r


def load_pop():
    p = {}
    with open(POB, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p[(row["ubigeo"], int(row["anio"]))] = int(row["poblacion"])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unit", choices=("victim", "incident"), default="victim",
        help="victim conserva la superficie canónica; incident usa r̂ por hechos",
    )
    a = ap.parse_args()
    rate_file = R_EB_DIR / (
        "r_distrital_eb.csv" if a.unit == "victim" else "r_distrital_eb_incident.csv"
    )
    if not rate_file.exists():
        raise SystemExit(
            f"falta {rate_file}; genera primero el EB con "
            f"python3 scripts/build_reporting_rate.py --unit {a.unit}"
        )

    obs, rr, pop = load_observed(), load_r(rate_file), load_pop()
    SILVER.mkdir(parents=True, exist_ok=True)
    rows = []
    excluded = defaultdict(int)
    for (u, cat, y), o in sorted(obs.items()):
        key = (u, cat)
        if key not in rr:
            excluded[(cat, "sin r̂ distrital")] += 1
            continue
        r_hat, r_lo, r_hi, scope = rr[key]
        if not math.isfinite(r_hat) or r_hat <= 0:
            excluded[(cat, "r̂ nulo o no finito")] += 1
            continue
        lat = o / r_hat
        if not math.isfinite(lat):
            excluded[(cat, "latente no finito")] += 1
            continue
        # r alto → latente bajo; un límite nulo no define un IC superior finito.
        lat_lo = o / r_hi if math.isfinite(r_hi) and r_hi > 0 else None
        lat_hi = o / r_lo if math.isfinite(r_lo) and r_lo > 0 else None
        popn = pop.get((u, y))
        rate = (lat / popn * 1e5) if popn else float("nan")
        rows.append(dict(
            ubigeo=u, categoria=cat, anio=y, observado=o,
            r_hat=round(r_hat, 4), r_ic_low=round(r_lo, 4), r_ic_high=round(r_hi, 4),
            r_scope=scope, latente=round(lat, 1),
            latente_ic_low=round(lat_lo, 1) if lat_lo is not None else "",
            latente_ic_high=round(lat_hi, 1) if lat_hi is not None else "",
            factor_subrep=round(1 / r_hat, 1),
            latente_rate_100k=round(rate, 1) if popn else "",
            inestable=int(r_hat < R_UNSTABLE),
        ))
    if not rows:
        raise SystemExit("no se produjo ninguna fila latente finita")
    cols = list(rows[0].keys())
    dest = SILVER / (
        "latente_distrito_categoria_anio.csv"
        if a.unit == "victim"
        else "latente_distrito_categoria_anio_incident.csv"
    )
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---------- resumen Lima: observado vs latente por categoría (pooled años) ----------
    print("=== Lima: observado vs latente por categoría (suma 2018-2024) ===")
    print(f"{'categoria':<26}{'observado':>11}{'latente':>13}{'factor':>8}{'flag':>10}")
    by_cat = defaultdict(lambda: [0, 0.0])
    for r in rows:
        by_cat[r["categoria"]][0] += r["observado"]
        by_cat[r["categoria"]][1] += r["latente"]
    for cat, (o, lt) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        flag = "INESTABLE" if (o and lt / o > 1 / R_UNSTABLE) else ""
        print(f"{cat:<26}{o:>11,}{lt:>13,.0f}{lt/max(o,1):>8.1f}{flag:>10}")
        # Emite el multiplicador pooled al registro canónico (hardening anti-stale).
        canon.emit(
            f"multiplier.{cat}", lt / max(o, 1), variant=a.unit,
            unit="latente/observado (adimensional)",
            estimator=f"pooled superficie Σλ*/Σy {YEARS.start}-{YEARS.stop - 1}, r̂ EB {a.unit}-level",
            inputs=[rate_file, MININTER], script=__file__,
        )

    # ---------- desplazamiento de ranking (total de-sesgable por distrito, 2024) ----------
    obs_d, lat_d = defaultdict(int), defaultdict(float)
    for r in rows:
        if r["anio"] != 2024 or r["inestable"]:  # excluir estafa (inestable) del total
            continue
        obs_d[r["ubigeo"]] += r["observado"]
        lat_d[r["ubigeo"]] += r["latente"]
    rank_o = {u: i for i, u in enumerate(sorted(obs_d, key=obs_d.get, reverse=True), 1)}
    rank_l = {u: i for i, u in enumerate(sorted(lat_d, key=lat_d.get, reverse=True), 1)}
    shifts = sorted(((u, rank_o[u], rank_l[u], rank_o[u] - rank_l[u]) for u in obs_d),
                    key=lambda t: -abs(t[3]))
    print("\n=== Desplazamiento de ranking distrital 2024 (de-sesgable, sin estafa) ===")
    print("  (Δ>0 = el distrito SUBE al de-sesgar → más sub-representado de lo que parecía)")
    for u, ro, rl, d in shifts[:6]:
        print(f"  {u}: observado #{ro:>2} → latente #{rl:>2}  (Δ{d:+d})")

    print(f"\n→ {dest}  ({len(rows)} filas, {len({r['ubigeo'] for r in rows})} distritos)")
    if excluded:
        print("Categorías/fila excluidas (nunca se emite latente infinito):")
        for (cat, why), n in sorted(excluded.items()):
            print(f"  {cat}: {n} filas — {why}")
    print(f"Contrato Etapa1→Etapa2 {a.unit}-nivel listo. "
          "estafa queda marcada inestable cuando r̂<0.05.")


if __name__ == "__main__":
    main()
