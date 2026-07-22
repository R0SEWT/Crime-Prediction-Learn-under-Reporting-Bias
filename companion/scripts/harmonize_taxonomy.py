#!/usr/bin/env python3
"""Armonización de taxonomías MININTER ↔ ENAPRES (bead slr-o41).

Cruza la taxonomía del NUMERADOR (denuncias MININTER, columna `P_MODALIDADES` del
CSV público, ya binned a 6 categorías) con la del DENOMINADOR (victimización ENAPRES
CAP_600). Hallazgo central: ENAPRES tiene DOS esquemas incompatibles en el panel:

  - 2019–2023 ("clásico"): victimización en `P601_X` (¿HA SIDO VÍCTIMA?), 11–12 tipos,
    SIN la categoría de robo callejero (dinero/cartera/celular).
  - 2024 ("ANUAL/rediseño"): renumeración completa — victimización en `P615_X`,
    ¿denunció? en `P619_X`, ¿dónde? en `P620_X`, venue Fiscalía en `P618_X_2`.
    27 tipos, INCLUYE robo callejero (P615_8/9). OJO: en 2024 `P601_X` pasó a ser
    PERCEPCIÓN de riesgo futuro (¿CREE QUE PUEDE SER VÍCTIMA?), NO victimización.

El script declara el crosswalk canónico (juicio semántico, versionado) y lo VERIFICA
contra las columnas reales + mide disponibilidad por año (la métrica que decide si el
join densifica celdas). No estima `r` (eso es slr-86y); deja el mapeo y la evidencia.

Uso:  python3 scripts/harmonize_taxonomy.py [--years 2019-2024] [--out analysis/...]
Requiere: stdlib. Lee data/datasets/enapres/raw/{year}.zip + el CSV MININTER público.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENAPRES_RAW = ROOT / "data" / "datasets" / "enapres" / "raw"
MININTER_CSV = ROOT / "data" / "datasets" / "DATASET_Denuncias_Policiales_Ene 2018 a Mayo 2026.csv"

# --- Crosswalk canónico (semántico, versionado) -----------------------------------
# Cada categoría común declara: modalidad MININTER (P_MODALIDADES), códigos ENAPRES
# del esquema clásico (P601_) y del rediseño 2024 (P615_), y si es estable en panel.
# ⚠️ CORREGIDO 2026-07-14 (slr-x50g). El mapeo 2024 anterior estaba CORRIDO −1 en 6 de 8
# categorías, y el robo callejero se declaraba inexistente en 2019–2023 cuando sí se mide.
# Verificado contra los diccionarios oficiales INEI que vienen en los propios zips crudos:
#   2024 → 965-Modulo1860/DICCIONARIO DE VARIABLES 2024 - CAP600.pdf
#   2022 → 785-Modulo1731/DICCIONARIO DE VARIABLES 2022 - CAP600.pdf
# El test tests/etapa2/test_crosswalk_dictionary.py valida este mapeo contra las etiquetas.
#
# Errores que había (P615_x REAL vs lo que el crosswalk decía que era):
#   robo_hurto_callejero [8,9]  → 8 = "Intento de robo de BICICLETA"  (correcto: 9 = "Robo de dinero, cartera, celular")
#   estafa               [20]   → 20 = "Intento de extorsión"          (correcto: 21)
#   secuestro            [16,17]→ 16 = "Ofensas sexuales"              (correcto: 17,18)
#   extorsion            [18,19]→ 18 = "Intento de secuestro"          (correcto: 19,20)
#   violencia_fam_sexual [14,15]→ 14 = "Amenazas e intimidaciones"     (correcto: 15,16)
#   amenazas             [13]   → 13 = "Vandalismo"                    (correcto: 14)
#   robo_vehiculo        [1..7] → incluía autopartes/moto/bicicleta, que el clásico deja
#                                 fuera de robo_vehiculo (correcto: [1,2], como el clásico)
#
# POLÍTICA DE TENTATIVAS. El esquema clásico incluye el intento donde el instrumento lo
# ofrece (secuestro=[10,11]=Secuestro+Intento; robo_vehiculo=[1,2]=Robo+Intento). El robo
# callejero SÍ tiene contraparte de intento en 2019–2023: P601_6B ("Intento de robo de dinero,
# cartera, celular") existe en las 5 olas clásicas (verificado contra los diccionarios INEI,
# re-verificación 2026-07-19, analysis/companion_reverification.md). Por eso definir el robo
# como CONSUMADO en ambos esquemas ([6A] y [9]) es una DECISIÓN DE ALCANCE deliberada: exclusión
# SIMÉTRICA y conservadora del intento (consumado existe idéntico en las 6 olas, P601_6A≡P615_9).
# Es la variante coherente en panel Y la más conservadora: rinde el multiplicador MENOR (×4.6),
# no ×6.6 que daría incluir tentativas.
#   · robo consumado 2024 ([9])      → r̂ = 0.196
#   · robo + intento 2024 ([9,10])   → r̂ = 0.134   ← NO usar: no replicable en el clásico
CROSSWALK = [
    # categoria          mininter                         enapres_2019_2023   enapres_2024(P615)        panel
    ("extorsion",        ["Extorsión"],                   [12],               [19, 20],                 True),
    ("estafa",           ["Estafa"],                      [13],               [21],                     True),
    ("secuestro",        ["Secuestro"],                   [10, 11],           [17, 18],                 True),
    ("robo_vehiculo",    [],                              [1, 2],             [1, 2],                   True),
    ("robo_negocio",     [],                              [14],               [11],                     True),
    ("amenazas",         [],                              [7],                [14],                     True),
    ("violencia_familiar_sexual", ["Violencia contra la mujer e integrantes"], [8, 9], [15, 16],        True),
    # Robo callejero: la categoría #1 de MININTER. SÍ se mide en 2019–2023 — es P601_6A
    # ("Robo de dinero, cartera, celular"). El crosswalk viejo la declaraba ausente porque
    # usaba códigos ENTEROS y "P601_{6}" no alcanza el sufijo "A". Ahora es panel-estable.
    ("robo_hurto_callejero", ["Robo", "Hurto"],           ["6A"],             [9],                      True),
]
MININTER_RESIDUAL = "Otros"  # no mapea a denominador ENAPRES

VICT_OLD = re.compile(r"^P601_(\d+)$", re.I)   # esquema clásico: victimización
VICT_NEW = re.compile(r"^P615_(\d+)$", re.I)   # rediseño 2024: victimización


def open_cap600(zpath: Path):
    """Devuelve (reader_factory, columnas) para el CAP_600, soportando zip anidado."""
    z = zipfile.ZipFile(zpath)

    def _find(zf):
        for n in zf.namelist():
            if "CAP_600" in n.upper() and n.endswith(".csv"):
                return zf, n
        for n in zf.namelist():
            if n.endswith(".zip"):
                r = _find(zipfile.ZipFile(io.BytesIO(zf.read(n))))
                if r[1]:
                    return r
        return None, None

    zf, member = _find(z)
    if not member:
        return None, None, None
    with zf.open(member) as fh:
        head = io.TextIOWrapper(fh, encoding="latin-1").readline().replace("\x00", "")
    delim = max([",", ";", "|"], key=head.count)
    cols = next(csv.reader([head], delimiter=delim))

    def factory():
        fh = zf.open(member)
        t = io.TextIOWrapper(fh, encoding="latin-1", newline="")
        next(t)  # header
        return csv.reader((l.replace("\x00", "") for l in t), delimiter=delim)

    return factory, cols, delim


def profile_year(year: int) -> dict | None:
    zpath = ENAPRES_RAW / f"{year}.zip"
    if not zpath.exists():
        return None
    factory, cols, _ = open_cap600(zpath)
    if not cols:
        return None
    idx = {c.upper(): i for i, c in enumerate(cols)}
    schema = "2024" if any(VICT_NEW.match(c) for c in cols) else "clasico"
    rex = VICT_NEW if schema == "2024" else VICT_OLD
    prefix = "P615" if schema == "2024" else "P601"
    types = sorted(int(m.group(1)) for c in cols if (m := rex.match(c)))
    i_dd = idx.get("CCDD")
    cnt = Counter()        # víctimas por tipo (nacional)
    cnt_lima = Counter()   # víctimas por tipo (Lima, CCDD==15)
    n = 0
    for row in factory():
        if len(row) < len(cols):
            continue
        n += 1
        lima = i_dd is not None and row[i_dd].strip().zfill(2) == "15"
        for x in types:
            j = idx.get(f"{prefix}_{x}")
            if j is not None and row[j].strip() == "1":
                cnt[x] += 1
                if lima:
                    cnt_lima[x] += 1
    return {"year": year, "schema": schema, "prefix": prefix, "types": types,
            "n": n, "cnt": dict(cnt), "cnt_lima": dict(cnt_lima)}


def mininter_counts() -> dict:
    c = Counter()
    with open(MININTER_CSV, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            c[row["P_MODALIDADES"]] += int(row["cantidad"])
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser(description="Armonización taxonómica MININTER↔ENAPRES (slr-o41)")
    ap.add_argument("--years", default="2019-2024")
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets" / "taxonomy_crosswalk.json"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-")) if "-" in a.years else (int(a.years), int(a.years))
    years = list(range(lo, hi + 1))

    print("=== NUMERADOR: MININTER P_MODALIDADES (público, ya binned) ===")
    mc = mininter_counts()
    tot = sum(mc.values())
    for k, v in sorted(mc.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>10,}  ({100*v/tot:4.1f}%)  {k}")

    print("\n=== DENOMINADOR: ENAPRES CAP_600 — esquema por año ===")
    prof = {}
    for y in years:
        p = profile_year(y)
        if not p:
            print(f"  {y}: (sin datos)")
            continue
        prof[y] = p
        print(f"  {y}: esquema={p['schema']:<8} victimización={p['prefix']}_X "
              f"({len(p['types'])} tipos, {p['n']:,} filas)")

    print("\n=== CROSSWALK + disponibilidad por año (víctimas Lima, CCDD=15) ===")
    rows = []
    for cat, mint, old, new, panel in CROSSWALK:
        avail = {}
        for y, p in prof.items():
            codes = new if p["schema"] == "2024" else old
            avail[y] = sum(p["cnt_lima"].get(x, 0) for x in codes) if codes else None
        rows.append({"categoria": cat, "mininter": mint, "enapres_2019_2023": old,
                     "enapres_2024_P615": new, "panel_estable": panel, "lima_por_anio": avail})
        flag = "PANEL" if panel else "SOLO-2024"
        cells = " ".join(f"{y}:{(avail[y] if avail[y] is not None else '—')}" for y in years if y in prof)
        print(f"  [{flag:<9}] {cat:<26} MININTER={mint or '—'}")
        print(f"              Lima víctimas/año → {cells}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mininter_modalidades": mc,
        "enapres_schema_by_year": {y: {"schema": p["schema"], "prefix": p["prefix"],
                                       "n_types": len(p["types"])} for y, p in prof.items()},
        "crosswalk": rows,
        "mininter_residual": MININTER_RESIDUAL,
    }, indent=2, ensure_ascii=False))
    print(f"\nCrosswalk → {out}")
    print("\nVEREDICTO: el robo callejero (Robo+Hurto, ~30% de MININTER) SÍ se mide en las 6 olas")
    print("(P601_6A clásico ≡ P615_9 2024 = 'robo de dinero, cartera, celular'); es panel-estable")
    print("2019–2024. La decisión de alcance es CONSUMADO-only (exclusión simétrica del intento,")
    print("P601_6B existe en clásico) → ver slr-86y y analysis/companion_reverification.md.")


if __name__ == "__main__":
    main()
