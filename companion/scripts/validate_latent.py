#!/usr/bin/env python3
"""Validación de la superficie latente (bead slr-bm3; corregido en slr-x50g).

Pregunta (§5): ¿el riesgo LATENTE (denuncias/r) representa el crimen real mejor que el
OBSERVADO (denuncias)? Gold standard = victimización ENAPRES por distrito de OCURRENCIA
(ponderada FACTOR).

⚠️ HISTORIAL — DOS BUGS CORREGIDOS EL 2026-07-14 (slr-x50g). Léelos antes de tocar nada:

1. **Se correlacionaban CONTEOS y se los llamaba tasas.** Este script hacía
   `spearmanr(observado_conteo, victimas_conteo)` entre 43 distritos, mientras el paper
   (§5.5) afirmaba comparar "denuncias per 1,000 inhabitants". La población distrital de
   Lima varía **1013×** (7,176 a 7,267,309 persona-años), así que ambos conteos escalan con
   el tamaño del distrito: ρ(población, denuncias) = 0.88–0.97 en TODAS las categorías. La
   tabla medía **tamaño poblacional**, no concordancia. Per cápita, 4 de 5 categorías caen a
   indistinguible de cero.

2. **El benchmark composicional sumaba olas desiguales.** El vector de ENAPRES agregaba
   crudo sobre años, pero el robo callejero solo tiene 6 olas y las otras 4 categorías solo
   5 (en la ola 2024 el INEI dejó de preguntar geografía de ocurrencia para los no
   patrimoniales). Ahora se **anualiza** (víctimas por ola) antes de normalizar.

Ambos empujaban en la dirección de la conclusión. Ver `analysis/companion_unit_errors.md`
y `analysis/crosswalk_enapres_fix.md`.

Se reportan las DOS versiones (conteos y per cápita) a propósito: la de conteos es la que se
publicó, y dejarla visible al lado de la correcta es lo que hace evidente el confundidor.

⚠️ LÍMITE DE CIRCULARIDAD (sigue vigente): `r` se estimó de ENAPRES, así que validar el
latente contra victimización ENAPRES es **parcialmente auto-referencial**. La prueba externa
genuina es SINADEF (`validate_sinadef_convergence.py`), y NO sostiene la superficie.

Uso:  python3 scripts/validate_latent.py
Requiere: numpy, scipy. Reusa build_reporting_rate.load_cells + latent silver.
"""
import csv
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reporting_rate import load_cells, lima_districts_canonical  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAT_DIR = ROOT / "data/datasets/silver/latent_surface"
POB = ROOT / "data/datasets/silver/inei_poblacion/poblacion_distrital_2018_2026.csv"
YEARS = list(range(2019, 2025))
DEBIAS_CATS = ["extorsion", "secuestro", "violencia_familiar_sexual", "robo_hurto_callejero", "estafa"]


def load_person_years() -> dict[str, float]:
    """Persona-años 2019–2024 por distrito: el denominador de un conteo pooled multi-año."""
    py: dict[str, float] = defaultdict(float)
    with open(POB, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["anio"]) in YEARS:
                py[r["ubigeo"]] += float(r["poblacion"])
    return py


def load_latent(path: Path):
    obs, lat = defaultdict(float), defaultdict(float)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["anio"]) not in YEARS:
                continue
            k = (r["ubigeo"], r["categoria"])
            obs[k] += float(r["observado"])
            lat[k] += float(r["latente"])
    return obs, lat


def sp(a, b):
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unit", choices=("victim", "incident"), default="victim",
        help="superficie que se valida; victim conserva la salida canónica",
    )
    a = ap.parse_args()
    lat_path = LAT_DIR / (
        "latente_distrito_categoria_anio.csv"
        if a.unit == "victim"
        else "latente_distrito_categoria_anio_incident.csv"
    )
    out = LAT_DIR / ("validacion.json" if a.unit == "victim" else "validacion_incident.json")
    if not lat_path.exists():
        raise SystemExit(f"falta {lat_path}; genera primero la superficie {a.unit}")
    obs, lat = load_latent(lat_path)
    available_cats = [cat for cat in DEBIAS_CATS if any(c == cat for _, c in lat)]
    excluded_cats = [cat for cat in DEBIAS_CATS if cat not in available_cats]
    cells = load_cells(YEARS)
    districts = lima_districts_canonical()
    py = load_person_years()

    missing = [u for u in districts if u not in py]
    if missing:
        raise SystemExit(f"sin población para {missing} — corre build_inei_poblacion.py")
    pop = np.array([py[u] for u in districts], float)

    # V(d,c) = victimización ENAPRES ponderada, por ocurrencia. `waves` = en cuántas olas
    # existe cada categoría (robo: 6; el resto: 5) — necesario para anualizar.
    V = defaultdict(float)
    waves = defaultdict(set)
    for (y, c, u), d in cells.items():
        if isinstance(u, str) and u.startswith("1501") and d["sw"] > 0:
            V[(u, c)] += d["sw"]
            waves[c].add(y)

    def vec(src, cat):
        return np.array([src.get((u, cat), 0.0) for u in districts], float)

    print(f"panel: {len(districts)} distritos · pooled {YEARS[0]}–{YEARS[-1]}")
    print(f"población (persona-años): {pop.min():,.0f} – {pop.max():,.0f}  ({pop.max()/pop.min():.0f}× de rango)\n")

    # ── 1. ESPACIAL ────────────────────────────────────────────────────────────────────
    print("=== 1. Espacial — ρ de Spearman entre distritos ===")
    print(f"{'categoría':<28}{'ρ CONTEOS':>11}{'ρ PER CÁPITA':>14}{'p':>8}{'ρ(pob,denuncias)':>19}")
    print(f"{'':28}{'(publicado)':>11}{'(correcto)':>14}")
    print("─" * 80)
    spat = {}
    for cat in available_cats:
        v, o, lt = vec(V, cat), vec(obs, cat), vec(lat, cat)
        if v.sum() == 0 or o.sum() == 0:
            continue
        r_cnt, _ = sp(o, v)                                   # como se publicó: conteo vs conteo
        ro, p_o = sp(o / pop, v / pop)                        # correcto: tasa vs tasa
        rl, p_l = sp(lt / pop, v / pop)
        r_pob, _ = sp(pop, o)                                 # el confundidor
        spat[cat] = dict(rho_conteos=r_cnt, rho_obs=ro, p_obs=p_o, rho_lat=rl, p_lat=p_l, rho_pob=r_pob)
        flag = "" if p_o < 0.05 else "  ← ns"
        print(f"{cat:<28}{r_cnt:>11.3f}{ro:>14.3f}{p_o:>8.3f}{r_pob:>19.3f}{flag}")

    stable = [c for c in available_cats if c != "estafa"]  # estafa: r̂ muy bajo, punto no fiable
    vt = sum(vec(V, c) for c in stable)
    ot = sum(vec(obs, c) for c in stable)
    lt_ = sum(vec(lat, c) for c in stable)
    r_cnt_t, _ = sp(ot, vt)
    ro_t, p_ot = sp(ot / pop, vt / pop)
    rl_t, p_lt = sp(lt_ / pop, vt / pop)
    r_pob_t, _ = sp(pop, ot)
    print("─" * 80)
    print(f"{'TOTAL (sin estafa)':<28}{r_cnt_t:>11.3f}{ro_t:>14.3f}{p_ot:>8.3f}{r_pob_t:>19.3f}")

    print("\n  ¿el de-sesgo MEJORA la concordancia? (per cápita, ρ latente − ρ observado)")
    for cat, s in spat.items():
        d = s["rho_lat"] - s["rho_obs"]
        sig = "*" if s["p_lat"] < 0.05 else " "
        print(f"    {cat:<28} {s['rho_obs']:>6.3f} → {s['rho_lat']:>6.3f}{sig}  Δ={d:+.3f}")
    print(f"    {'TOTAL (sin estafa)':<28} {ro_t:>6.3f} → {rl_t:>6.3f}   Δ={rl_t-ro_t:+.3f}")

    # ── 2. COMPOSICIÓN ─────────────────────────────────────────────────────────────────
    print("\n=== 2. Composición — share de categoría (Lima) ===")
    cats = stable + (["estafa"] if "estafa" in available_cats else [])
    so = np.array([vec(obs, c).sum() for c in cats])
    sl = np.array([vec(lat, c).sum() for c in cats])
    # ⚠️ ANUALIZADO: robo tiene 6 olas, el resto 5. Sumar crudo compara 6 años contra 5.
    sv_raw = np.array([vec(V, c).sum() for c in cats])
    sv_ann = np.array([vec(V, c).sum() / len(waves[c]) for c in cats])
    so, sl = so / so.sum(), sl / sl.sum()
    sv_raw, sv_ann = sv_raw / sv_raw.sum(), sv_ann / sv_ann.sum()

    print(f"{'categoría':<28}{'olas':>5}{'observado':>11}{'latente':>10}{'π crudo':>10}{'π ANUAL':>10}")
    for i, c in enumerate(cats):
        print(f"{c:<28}{len(waves[c]):>5}{so[i]:>11.3f}{sl[i]:>10.3f}{sv_raw[i]:>10.3f}{sv_ann[i]:>10.3f}")

    l1_o = float(np.abs(so - sv_ann).sum())
    l1_l = float(np.abs(sl - sv_ann).sum())
    better = l1_l < l1_o
    print(f"\n  L1 contra π ANUALIZADO:  observado={l1_o:.3f}  latente={l1_l:.3f}")
    print(f"  → la corrección {'MEJORA' if better else 'EMPEORA'} la composición "
          f"({(l1_l-l1_o)/l1_o*100:+.0f}%)")

    # ── VEREDICTO ──────────────────────────────────────────────────────────────────────
    ok_pc = [c for c, s in spat.items() if s["p_obs"] < 0.05]
    ns_pc = [c for c, s in spat.items() if s["p_obs"] >= 0.05]
    print("\n=== VEREDICTO ===")
    print(f"  NIVEL de validez convergente (per cápita): significativa en {ok_pc or 'ninguna'};")
    print(f"    NULA en {ns_pc}. Los ρ de conteos (0.67–0.94) medían población:")
    print(f"    ρ(población, denuncias) = {min(s['rho_pob'] for s in spat.values()):.2f}–"
          f"{max(s['rho_pob'] for s in spat.values()):.2f}.")
    print(f"  EFECTO del de-sesgo: mejora la concordancia en "
          f"{sum(1 for s in spat.values() if s['rho_lat'] > s['rho_obs'])}/{len(spat)} categorías.")
    print(f"  COMPOSICIÓN: la corrección {'mejora' if better else 'EMPEORA'} la mezcla "
          f"(L1 {l1_o:.3f}→{l1_l:.3f}).")
    print("  CIRCULARIDAD: r viene de ENAPRES → esta prueba es parcialmente auto-referencial.")
    print("    La externa (SINADEF, validate_sinadef_convergence.py) NO sostiene la superficie.")
    if excluded_cats:
        print(f"  Sin superficie finita: {', '.join(excluded_cats)}.")

    out.write_text(json.dumps({
        "nota": "ρ per cápita = correcto. ρ conteos = como se publicó (confundido por población).",
        "denominador": "persona-años 2019–2024, INEI",
        "categorias_sin_superficie": excluded_cats,
        "espacial_por_categoria": spat,
        "espacial_total_sin_estafa": dict(rho_conteos=r_cnt_t, rho_obs=ro_t, p_obs=p_ot,
                                          rho_lat=rl_t, p_lat=p_lt, rho_pob=r_pob_t),
        "composicion": {
            "cats": cats,
            "olas_por_cat": {c: len(waves[c]) for c in cats},
            "observado": so.tolist(), "latente": sl.tolist(),
            "enapres_crudo": sv_raw.tolist(), "enapres_anualizado": sv_ann.tolist(),
            "L1_vs_anualizado": {"observado": l1_o, "latente": l1_l},
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
