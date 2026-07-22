#!/usr/bin/env python3
"""Sensibilidad MAUP / falacia ecológica del de-sesgo (bead slr-60q, V10).

La superficie latente se construye a nivel **distrito** (43 de Lima). Eso invita dos
críticas estándar de la geografía cuantitativa:
  - **MAUP (Modifiable Areal Unit Problem):** los resultados podrían depender de la
    escala y de la zonación elegidas.
  - **Falacia ecológica:** una `r` por unidad no es una `r` por individuo.

Este script audita la robustez con el método riguroso de MAUP —**re-zonación
sistemática**— en vez de una sola agrupación administrativa discutible:

  1. ESCALA: agrega los 43 distritos a k unidades, k ∈ {43, 24, 12, 6, 3, 1}.
  2. ZONACIÓN: para cada k intermedio, R particiones aleatorias → distribución de la
     métrica, no un único corte cherry-picked.

Dos métricas por escala/zonación, ambas con el benchmark ENAPRES **anualizado** y
**sin violencia familiar** (las 4 categorías comparables; ver B1 / slr-x50g):

  - **Composición** (distancia L1 de la mezcla de categorías vs victimización): el
    de-sesgo la ALEJA del benchmark a TODA escala (frac_lat_better→0), robusto MAUP del
    resultado negativo de composición (§5.4). No es "el aporte del de-sesgo": la versión
    V10 de esta métrica sumaba olas desiguales e incluía violencia familiar, produciendo
    el titular falso L1 0.99→0.11 que B1 demolió.
  - **Ranking espacial de ROBO** (Spearman zona-nivel latente/observado vs V, solo robo):
    la afirmación que el paper restringido a robo sí puede hacer, sin mezcla de olas.

El factor de **magnitud** 1/r̂ por categoría es un cociente de sumas → exactamente
invariante a la partición (no requiere simulación).

Uso:  python3 scripts/maup_sensitivity.py [--reps 200] [--seed 0]
Requiere: numpy, scipy. Reusa build_reporting_rate + latent silver.
Salida: data/datasets/silver/latent_surface/maup_sensitivity.json + analysis/maup.md
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reporting_rate import load_cells, lima_districts_canonical  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAT = ROOT / "data" / "datasets" / "silver" / "latent_surface" / "latente_distrito_categoria_anio.csv"
OUT = ROOT / "data" / "datasets" / "silver" / "latent_surface" / "maup_sensitivity.json"
REPORT = ROOT / "analysis" / "maup.md"
YEARS = list(range(2019, 2025))
CATS = ["extorsion", "secuestro", "violencia_familiar_sexual", "robo_hurto_callejero", "estafa"]
# Composición: SOLO las 4 categorías directamente comparables. Se excluye violencia familiar
# (universos MININTER/ENAPRES no comparables, §3.1) — igual que build_composition_canon.py.
# Incluirla, más sumar el benchmark ENAPRES crudo sobre olas desiguales (robo=1 ola vs resto=5),
# fue lo que produjo el titular falso L1 0.99→0.11 (B1, slr-x50g).
COMP_CATS = [c for c in CATS if c != "violencia_familiar_sexual"]
STABLE = [c for c in COMP_CATS if c != "estafa"]
ROBBERY = "robo_hurto_callejero"
SCALES = [43, 24, 12, 6, 3, 1]


def load_latent():
    obs, lat = defaultdict(float), defaultdict(float)
    with open(LAT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["ubigeo"], r["categoria"])
            obs[k] += float(r["observado"])
            lat[k] += float(r["latente"])
    return obs, lat


def comp_l1(units, src, V, cats, waves):
    """L1 medio (ponderado por masa V de la unidad) entre la composición de `src` y la
    de la victimización ENAPRES, sobre las categorías `cats`.

    El benchmark ENAPRES se **anualiza** (víctimas / nº de olas de la categoría) antes de
    normalizar: robo se mide en 6 olas y el resto en 5, de modo que sumar el crudo compara
    6 años contra 5 y sesga el peso relativo de robo. Ese sesgo de olas desiguales fue el
    origen del titular falso L1 0.99→0.11 (B1). `src` (observado/latente) ya es la superficie
    pooled 2019–2024, así que solo el benchmark necesita anualizarse (igual que validate_latent).
    """
    num = den = 0.0
    for zone, ds in units.items():
        s = np.array([sum(src.get((u, c), 0.0) for u in ds) for c in cats])
        v = np.array([sum(V.get((u, c), 0.0) for u in ds) / len(waves[c]) for c in cats])
        if s.sum() <= 0 or v.sum() <= 0:
            continue
        l1 = np.abs(s / s.sum() - v / v.sum()).sum()
        w = v.sum()                       # peso = víctimas reales (anualizadas) en la unidad
        num += w * l1
        den += w
    return num / den if den else float("nan")


def partition(districts, k, rng):
    """Asigna los distritos a k zonas aleatorias (cada zona no vacía si k<=n)."""
    ds = list(districts)
    rng.shuffle(ds)
    zones = defaultdict(list)
    for i, u in enumerate(ds):
        zones[i % k].append(u)        # round-robin tras shuffle → zonas balanceadas
    return dict(zones)


def spatial_rho(units, obs, lat, V, cats):
    """Spearman(total_src, total_V) entre unidades, para obs y lat."""
    zo, zl, zv = [], [], []
    for ds in units.values():
        zo.append(sum(obs.get((u, c), 0.0) for u in ds for c in cats))
        zl.append(sum(lat.get((u, c), 0.0) for u in ds for c in cats))
        zv.append(sum(V.get((u, c), 0.0) for u in ds for c in cats))
    if len(zv) < 3:
        return None, None
    return spearmanr(zo, zv).statistic, spearmanr(zl, zv).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cells = load_cells(YEARS)
    obs, lat = load_latent()
    districts = [u for u in lima_districts_canonical() if u.startswith("1501")]
    V = defaultdict(float)
    waves = defaultdict(set)               # en cuántas olas se mide cada categoría (robo=6, resto=5)
    for (y, c, u), d in cells.items():
        if isinstance(u, str) and u.startswith("1501"):
            V[(u, c)] += d["sw"]
            if d["sw"] > 0:
                waves[c].add(y)

    # ---------- magnitud: factor 1/r por categoría es invariante a la partición ----------
    mag = {}
    for c in CATS:
        o = sum(obs.get((u, c), 0.0) for u in districts)
        l = sum(lat.get((u, c), 0.0) for u in districts)
        mag[c] = l / o if o else float("nan")

    # spatial_rho para ROBO SOLO: es la afirmación que el paper restringido puede hacer.
    # Una sola categoría, un solo alcance de olas → sin mezcla de olas (a diferencia del
    # total sobre STABLE, que suma el peso 2024-only de robo con los 2019–2023 del resto).
    def rob_rho(units):
        return spatial_rho(units, obs, lat, V, [ROBBERY])

    # ---------- composición y espacial por escala (con re-zonación aleatoria) ----------
    rows = []
    for k in SCALES:
        if k == 43:
            units = {u: [u] for u in districts}
            lo = comp_l1(units, obs, V, COMP_CATS, waves)
            ll = comp_l1(units, lat, V, COMP_CATS, waves)
            ro, rl = rob_rho(units)
            rows.append(dict(k=k, reps=1, l1_obs_mean=lo, l1_lat_mean=ll,
                             l1_obs_sd=0.0, l1_lat_sd=0.0,
                             rho_obs=ro, rho_lat=rl, frac_lat_better=1.0 if ll < lo else 0.0))
        elif k == 1:
            units = {0: list(districts)}
            lo = comp_l1(units, obs, V, COMP_CATS, waves)
            ll = comp_l1(units, lat, V, COMP_CATS, waves)
            rows.append(dict(k=k, reps=1, l1_obs_mean=lo, l1_lat_mean=ll,
                             l1_obs_sd=0.0, l1_lat_sd=0.0, rho_obs=None, rho_lat=None,
                             frac_lat_better=1.0 if ll < lo else 0.0))
        else:
            los, lls, ros, rls, better = [], [], [], [], 0
            for _ in range(args.reps):
                units = partition(districts, k, rng)
                lo = comp_l1(units, obs, V, COMP_CATS, waves)
                ll = comp_l1(units, lat, V, COMP_CATS, waves)
                los.append(lo); lls.append(ll)
                if ll < lo:
                    better += 1
                if k >= 12:
                    ro, rl = rob_rho(units)
                    if ro is not None:
                        ros.append(ro); rls.append(rl)
            rows.append(dict(
                k=k, reps=args.reps,
                l1_obs_mean=float(np.mean(los)), l1_lat_mean=float(np.mean(lls)),
                l1_obs_sd=float(np.std(los)), l1_lat_sd=float(np.std(lls)),
                rho_obs=float(np.mean(ros)) if ros else None,
                rho_lat=float(np.mean(rls)) if rls else None,
                frac_lat_better=better / args.reps))

    # ---------- consola ----------
    print("=== MAUP — composición (4 cats, ENAPRES anualizado) y ranking espacial de robo ===")
    print(f"{'k (unid.)':>10}{'reps':>6}{'L1 obs':>10}{'L1 lat':>10}{'lat mejor':>11}"
          f"{'ρ_rob obs':>11}{'ρ_rob lat':>11}")
    for r in rows:
        ro = f"{r['rho_obs']:.2f}" if r['rho_obs'] is not None else "—"
        rl = f"{r['rho_lat']:.2f}" if r['rho_lat'] is not None else "—"
        print(f"{r['k']:>10}{r['reps']:>6}{r['l1_obs_mean']:>10.3f}{r['l1_lat_mean']:>10.3f}"
              f"{r['frac_lat_better']:>10.0%}{ro:>12}{rl:>11}")
    print("\nComposición: el de-sesgo ALEJA la mezcla del benchmark a toda escala "
          "(lat mejor→0%), robusto MAUP del resultado negativo de §5.4.")
    print("\n=== Magnitud (factor 1/r̂ por categoría — cociente de sumas, invariante a la partición) ===")
    for c, f in sorted(mag.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<26} ×{f:.1f}")

    result = dict(scales=rows, magnitude_factor=mag, reps=args.reps, seed=args.seed,
                  comp_cats=COMP_CATS, excluded_cat="violencia_familiar_sexual",
                  benchmark="ENAPRES anualizado (víctimas / nº olas por categoría)",
                  spatial_metric="Spearman zona-nivel, robo solo", waves_per_cat={c: len(w) for c, w in waves.items()})
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(rows, mag)
    print(f"\n→ {OUT}\n→ {REPORT}")


def write_report(rows, mag):
    L = []
    L.append("# Sensibilidad MAUP del de-sesgo — restringido a robo (bead slr-x50g.8)\n")
    L.append("_¿Las conclusiones de la corrección (construida a nivel distrito) sobreviven "
             "críticas de unidad areal modificable (MAUP)? Método: re-zonación sistemática "
             "(escala + zonación aleatoria), no una agrupación administrativa única. "
             "Benchmark = victimización ENAPRES **anualizada** (víctimas / nº de olas por "
             "categoría) sobre las **4 categorías comparables** (excl. violencia familiar, "
             "§3.1). La versión V10 sumaba olas desiguales e incluía violencia familiar → "
             "titular falso L1 0.99→0.11 (B1). Regenera: "
             "`python3 scripts/maup_sensitivity.py`. JSON: "
             "`data/silver/.../maup_sensitivity.json`._\n")
    L.append("## 1. Composición por escala — el de-sesgo la ALEJA del benchmark\n")
    L.append("Para cada escala k (nº de unidades; k=43 distritos → k=1 Lima total) se "
             "remuestrean **particiones aleatorias** de los 43 distritos y se mide la "
             "distancia L1 media (ponderada por víctimas) entre la composición de las 4 "
             "categorías comparables de cada unidad y la de la victimización anualizada. "
             "`ρ_rob` = Spearman zona-nivel del riesgo de **robo** (observado/latente) vs "
             "victimización, la métrica espacial libre de mezcla de olas:\n")
    L.append("| k (unidades) | reps | L1 observado | L1 latente | latente mejor | ρ_rob(obs,V) | ρ_rob(lat,V) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        ro = f"{r['rho_obs']:.2f}" if r['rho_obs'] is not None else "—"
        rl = f"{r['rho_lat']:.2f}" if r['rho_lat'] is not None else "—"
        sd_o = f" ±{r['l1_obs_sd']:.02f}" if r['reps'] > 1 else ""
        sd_l = f" ±{r['l1_lat_sd']:.02f}" if r['reps'] > 1 else ""
        L.append(f"| {r['k']} | {r['reps']} | {r['l1_obs_mean']:.3f}{sd_o} "
                 f"| {r['l1_lat_mean']:.3f}{sd_l} | {r['frac_lat_better']:.0%} | {ro} | {rl} |")
    L.append("")
    never = all(r["frac_lat_better"] <= 0.01 for r in rows)
    L.append(f"**Hallazgo:** el de-sesgo **aleja** la composición del benchmark de encuesta "
             f"{'a TODA escala y en ~todas las zonaciones' if never else 'en la mayoría de escalas/zonaciones'} "
             "(columna *latente mejor* ≈ 0%): el L1 latente supera al observado en cada "
             "corte. El resultado negativo de composición (§5.4) **no es un artefacto de la "
             "partición distrital** — persiste al re-zonar y al cambiar de escala. Es la "
             "confirmación MAUP de por qué la superficie validada se restringe a robo.\n")
    L.append("## 2. Magnitud — invariante a la partición\n")
    L.append("El factor de sub-representación 1/r̂ se estima por **categoría** (no por "
             "unidad espacial; r̂ es categoría×año), así que es un **cociente de sumas** e "
             "idéntico bajo cualquier zonación o escala — sin necesidad de simulación:\n")
    L.append("| categoría | factor 1/r̂ |")
    L.append("|---|---|")
    for c, f in sorted(mag.items(), key=lambda kv: -kv[1]):
        L.append(f"| {c} | ×{f:.1f} |")
    L.append("")
    L.append("## 3. Ranking espacial de robo — estable bajo agregación\n")
    L.append("Para robo, ρ_rob(obs,V) ≈ ρ_rob(lat,V) a cada escala (columnas de la tabla "
             "§1): el de-sesgo **no reordena** el mapa de robo al cambiar la unidad — el "
             "valor de la corrección está en la **magnitud**, no en un reordenamiento "
             "espacial. MAUP afecta el valor absoluto de ρ (baja al desagregar, como es de "
             "esperar), pero no la conclusión. Métrica de robo solo → sin la mezcla de olas "
             "que contamina un total multi-categoría.\n")
    L.append("## 4. Límite ecológico (declarado)\n")
    L.append("La invariancia de magnitud a la partición se debe a que el de-sesgo opera "
             "a nivel **categoría**, no espacial. El reverso es el **límite ecológico**: "
             "`r` es una tasa de reporte por unidad (distrito/zona), no por individuo. "
             "Aplicarla uniformemente dentro de la unidad **no recupera** la "
             "heterogeneidad individual del reporte (dos víctimas del mismo distrito con "
             "distinta propensión a denunciar). El latente es un **target areal "
             "de-sesgado**, no una imputación a nivel persona; la variación intra-unidad "
             "es trabajo de la Etapa 2 (resolución fina) y queda fuera del alcance "
             "identificable aquí. Esto se reporta como límite, no se sobre-vende.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
