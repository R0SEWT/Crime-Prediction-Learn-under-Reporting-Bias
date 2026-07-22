#!/usr/bin/env python3
"""Validación convergente SINADEF vs superficie latente (bead slr-53n.3; corregido slr-x50g).

Pregunta: ¿la superficie latente (denuncias MININTER / r_ENAPRES) correlaciona
mejor que el observado MININTER con un validador externo libre de sesgo de
denuncia? Validador: tasa distrital de homicidios SINADEF. Es la ÚNICA prueba
externa del paper que no es circular respecto de ENAPRES.

⚠️ BUG CORREGIDO EL 2026-07-14 (slr-x50g). Este script correlacionaba
`tasa_100k` (homicidios POR 100k) contra `observado`/`latente` (denuncias en
CONTEO ABSOLUTO): una TASA contra un CONTEO. Y la prosa del paper (§5.5) afirma
que SINADEF se agregó "to district × year rates using INEI population
denominators" — describía la comparación correcta y ejecutaba otra. La columna
`latente_rate_100k` YA EXISTÍA en el silver y se ignoraba.

Lo que medía en realidad era POBLACIÓN: ρ(población, denuncias) = +0.914 y
ρ(población, tasa de homicidios) = +0.326, cuyo producto (≈0.30) reproduce casi
exacto el +0.336 publicado.

Corregido (n=258, 43 distritos × 6 años):
    publicado (tasa vs conteo):  ρ_obs = +0.336***   ρ_lat = +0.292***
    correcto  (tasa vs tasa):    ρ_obs = -0.031 (ns) ρ_lat = -0.173 (p=0.005)

⇒ En la comparación correcta el ancla externa NO sostiene la superficie: el
observado no tiene relación con ella y el latente correlaciona negativamente.
§8 del paper no puede usarla para "mitigate this circularity". Ver
`analysis/companion_unit_errors.md`.

Se reportan AMBAS versiones a propósito: dejar el número publicado al lado del
correcto es lo que hace evidente el confundidor.

La prueba es convergente, no causal: homicidio no está entre las categorías
de-sesgadas y SINADEF en este extracto solo ofrece UBIGEO de domicilio (no de
ocurrencia). Ambas cosas ATENÚAN cualquier correlación — así que el ρ negativo
es difícil de interpretar, pero no rescata el ρ positivo publicado, que es
artefacto poblacional.

Uso:  python3 scripts/validate_sinadef_convergence.py
Requiere: pandas, numpy, scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
LATENT_CSV = ROOT / "data" / "datasets" / "silver" / "latent_surface" / "latente_distrito_categoria_anio.csv"
SINADEF_CSV = ROOT / "data" / "silver" / "sinadef" / "homicidios_distrito_anio.csv"
SINADEF_META = ROOT / "data" / "silver" / "sinadef" / "homicidios_distrito_anio_metadata.json"
POB_CSV = ROOT / "data" / "datasets" / "silver" / "inei_poblacion" / "poblacion_distrital_2018_2026.csv"
OUT_MD = ROOT / "analysis" / "sinadef_validation.md"
YEARS = list(range(2019, 2025))


def rho_spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    xx = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    n = int(mask.sum())
    if n < 3 or len(np.unique(xx[mask])) < 2 or len(np.unique(yy[mask])) < 2:
        return math.nan, n
    return float(spearmanr(xx[mask], yy[mask]).statistic), n


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        if 1900 <= int(value) <= 2100:
            return str(int(value))
        return f"{int(value):,}"
    return str(value)


def md_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    keys = [key for key, _ in columns]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not LATENT_CSV.exists():
        raise FileNotFoundError(LATENT_CSV)
    if not SINADEF_CSV.exists():
        raise FileNotFoundError(f"{SINADEF_CSV} no existe; corre build_sinadef_homicidios.py primero")

    lat = pd.read_csv(LATENT_CSV, dtype={"ubigeo": "string"})
    sin = pd.read_csv(SINADEF_CSV, dtype={"ubigeo": "string"})
    lat["anio"] = pd.to_numeric(lat["anio"], errors="coerce").astype(int)
    sin["anio"] = pd.to_numeric(sin["anio"], errors="coerce").astype(int)
    for col in ["observado", "latente"]:
        lat[col] = pd.to_numeric(lat[col], errors="coerce").fillna(0.0)
    sin["tasa_100k"] = pd.to_numeric(sin["tasa_100k"], errors="coerce")
    sin["homicidios"] = pd.to_numeric(sin["homicidios"], errors="coerce").fillna(0).astype(int)

    meta = json.loads(SINADEF_META.read_text(encoding="utf-8")) if SINADEF_META.exists() else {}
    return lat, sin, meta


def build_panel(lat: pd.DataFrame, sin: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    lat = lat[lat["anio"].isin(YEARS)].copy()
    districts = sorted(lat["ubigeo"].dropna().unique().tolist())
    categories = sorted(lat["categoria"].dropna().unique().tolist())

    idx = pd.MultiIndex.from_product([districts, YEARS, categories], names=["ubigeo", "anio", "categoria"])
    by_cat = (
        lat.groupby(["ubigeo", "anio", "categoria"], as_index=True)[["observado", "latente"]]
        .sum()
        .reindex(idx, fill_value=0.0)
        .reset_index()
    )

    hom = sin.loc[sin["anio"].isin(YEARS), ["ubigeo", "anio", "homicidios", "tasa_100k"]].copy()
    hom = hom[hom["ubigeo"].isin(districts)]
    hom_idx = pd.MultiIndex.from_product([districts, YEARS], names=["ubigeo", "anio"])
    hom = (
        hom.groupby(["ubigeo", "anio"], as_index=True)[["homicidios", "tasa_100k"]]
        .sum()
        .reindex(hom_idx)
        .reset_index()
    )
    hom["homicidios"] = hom["homicidios"].fillna(0).astype(int)

    total = by_cat.groupby(["ubigeo", "anio"], as_index=False)[["observado", "latente"]].sum()
    total["categoria"] = "TOTAL"
    panel_total = total.merge(hom, on=["ubigeo", "anio"], how="left")
    panel_cat = by_cat.merge(hom, on=["ubigeo", "anio"], how="left")
    panel = pd.concat([panel_total, panel_cat], ignore_index=True)

    # Denominador poblacional: sin él, `observado`/`latente` son CONTEOS y `tasa_100k` es una
    # TASA — la comparación mide tamaño de distrito, no concordancia (ver docstring, bug #1).
    pob = pd.read_csv(POB_CSV, dtype={"ubigeo": "string"})
    pob["anio"] = pd.to_numeric(pob["anio"], errors="coerce").astype(int)
    panel = panel.merge(pob[["ubigeo", "anio", "poblacion"]], on=["ubigeo", "anio"], how="left")
    for col in ["observado", "latente"]:
        panel[f"{col}_rate_100k"] = panel[col] / panel["poblacion"] * 1e5
    return panel, districts, categories


def compute_metrics(panel: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = ["TOTAL", *categories]
    scopes: list[tuple[str, pd.Series]] = [(str(y), panel["anio"].eq(y)) for y in YEARS]
    scopes.append(("pooled", panel["anio"].isin(YEARS)))

    for scope, smask in scopes:
        for target in targets:
            df = panel.loc[smask & panel["categoria"].eq(target)].copy()
            # CORRECTO: tasa vs tasa (homicidios/100k vs denuncias/100k)
            ro, n_obs = rho_spearman(df["tasa_100k"], df["observado_rate_100k"])
            rl, n_lat = rho_spearman(df["tasa_100k"], df["latente_rate_100k"])
            # COMO SE PUBLICÓ: tasa vs CONTEO (el bug) — se conserva para que el confundidor
            # quede a la vista al lado del número correcto.
            ro_cnt, _ = rho_spearman(df["tasa_100k"], df["observado"])
            rl_cnt, _ = rho_spearman(df["tasa_100k"], df["latente"])
            # El confundidor: ρ(población, conteo de denuncias)
            r_pob, _ = rho_spearman(df["poblacion"], df["observado"])
            # CONTEO-vs-CONTEO (homicidios vs denuncias): la comparación que ilustra que
            # ambos lados cargan la huella de población — sube a ~0.87 sin decir nada de
            # concordancia espacial de tasas (slr-x50g.6).
            rcc_o, _ = rho_spearman(df["homicidios"], df["observado"])
            rcc_l, _ = rho_spearman(df["homicidios"], df["latente"])
            rows.append({
                "scope": scope,
                "categoria": target,
                "n": min(n_obs, n_lat),
                "rho_observado": ro,
                "rho_latente": rl,
                "delta_lat_menos_obs": rl - ro if np.isfinite(ro) and np.isfinite(rl) else math.nan,
                "rho_obs_CONTEOS_publicado": ro_cnt,
                "rho_lat_CONTEOS_publicado": rl_cnt,
                "rho_poblacion_denuncias": r_pob,
                "rho_countcount_obs": rcc_o,
                "rho_countcount_lat": rcc_l,
            })
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, panel: pd.DataFrame, districts: list[str], categories: list[str], meta: dict[str, object]) -> None:
    total_rows = metrics[metrics["categoria"].eq("TOTAL")].to_dict("records")
    cat_pooled = metrics[(metrics["scope"].eq("pooled")) & (~metrics["categoria"].eq("TOTAL"))].to_dict("records")
    cat_year = metrics[(~metrics["scope"].eq("pooled")) & (~metrics["categoria"].eq("TOTAL"))].to_dict("records")

    pooled_total = metrics[(metrics["scope"].eq("pooled")) & (metrics["categoria"].eq("TOTAL"))].iloc[0]
    delta = float(pooled_total["delta_lat_menos_obs"])
    if delta > 1e-12:
        central = "Sí: en el pooled total el latente correlaciona más que el observado."
    elif delta < -1e-12:
        central = "No: en el pooled total el observado correlaciona más que el latente."
    else:
        central = "Empate práctico: en el pooled total el latente y el observado correlacionan igual."

    # ¿el ancla sostiene ALGO? (en la comparación correcta, tasa-vs-tasa)
    ro_p, rl_p = float(pooled_total["rho_observado"]), float(pooled_total["rho_latente"])
    if max(ro_p, rl_p) < 0.1:
        anchor = ("**El ancla externa NO sostiene ninguna de las dos superficies.** En la "
                  "comparación correcta (tasa contra tasa) ambas ρ son nulas o negativas, así "
                  "que esta prueba **no puede** usarse para mitigar la circularidad del resto "
                  "del diseño.")
    else:
        anchor = "El ancla externa muestra concordancia positiva con al menos una superficie."

    total_better = int((metrics[metrics["categoria"].eq("TOTAL")]["delta_lat_menos_obs"] > 0).sum())
    total_defined = int(metrics[metrics["categoria"].eq("TOTAL")]["delta_lat_menos_obs"].notna().sum())
    cat_better = int((metrics[~metrics["categoria"].eq("TOTAL")]["delta_lat_menos_obs"] > 0).sum())
    cat_defined = int(metrics[~metrics["categoria"].eq("TOTAL")]["delta_lat_menos_obs"].notna().sum())

    hom_summary = (
        panel[panel["categoria"].eq("TOTAL")]
        .groupby("anio", as_index=False)
        .agg(homicidios=("homicidios", "sum"), tasa_media_100k=("tasa_100k", "mean"))
    )
    hom_rows = hom_summary.to_dict("records")

    district_prefixes = sorted({u[:4] for u in districts})
    geo_caveat = meta.get(
        "geo_caveat",
        "SINADEF entregado no contiene UBIGEO de ocurrencia; se usa domicilio del fallecido.",
    )

    content = f"""# Validación convergente con SINADEF homicidios

> ⚠️ **CORREGIDO 2026-07-14 (slr-x50g).** Este reporte antes correlacionaba una **tasa**
> (homicidios/100k) contra un **conteo** (denuncias absolutas), y publicaba
> ρ(observado)=+0.336 / ρ(latente)=+0.292. Eso medía **población**:
> ρ(población, denuncias)=+0.914 × ρ(población, tasa de homicidios)=+0.326 ≈ 0.30.
> Ahora ambos lados van en tasas por 100k. Los números publicados se conservan en las tablas,
> en la columna «(conteos, publicado)», para que el confundidor quede a la vista.

**Pregunta central.** {central}

En el total pooled {YEARS[0]}-{YEARS[-1]}, **tasa contra tasa**:
ρ(observado, homicidios)=`{fmt(float(pooled_total['rho_observado']))}` y
ρ(latente, homicidios)=`{fmt(float(pooled_total['rho_latente']))}`; Δ=`{fmt(delta)}`.
(Como se publicó, tasa-vs-conteo: `{fmt(float(pooled_total['rho_obs_CONTEOS_publicado']))}` y
`{fmt(float(pooled_total['rho_lat_CONTEOS_publicado']))}`.)

{anchor}

## Datos usados

- SINADEF silver: `data/silver/sinadef/homicidios_distrito_anio.csv`
- Filtro principal SINADEF: `MUERTE VIOLENTA == HOMICIDIO`.
- Sensibilidad CIE-X agresión X85-Y09: `{fmt(meta.get('agresion_cie_x85_y09', 'NA'))}` filas vs `{fmt(meta.get('homicidio_muerte_violenta', 'NA'))}` por muerte violenta en el archivo completo; diferencia `{fmt(meta.get('diferencia_agresion_cie_menos_homicidio_muerte_violenta', 'NA'))}`. En el subconjunto Perú+mapeado: `{fmt(meta.get('agresion_cie_x85_y09_peru_mapeado', 'NA'))}` vs `{fmt(meta.get('homicidio_muerte_violenta_peru_mapeado', 'NA'))}`; diferencia `{fmt(meta.get('diferencia_agresion_cie_menos_homicidio_muerte_violenta_peru_mapeado', 'NA'))}`.
- Geografía SINADEF: `{meta.get('geo_field_used', 'COD# UBIGEO DOMICILIO')}`. {geo_caveat}
- Superficie latente: `{len(districts)}` distritos presentes, años {YEARS[0]}-{YEARS[-1]}, categorías `{', '.join(categories)}`.
- Prefijos UBIGEO en la superficie validada: `{', '.join(district_prefixes)}`.

## Cobertura SINADEF en distritos validados

{md_table(hom_rows, [
    ('anio', 'año'),
    ('homicidios', 'homicidios'),
    ('tasa_media_100k', 'tasa media 100k'),
])}

## Resultado total

{md_table(total_rows, [
    ('scope', 'año/scope'),
    ('n', 'n'),
    ('rho_observado', 'ρ obs (tasa)'),
    ('rho_latente', 'ρ lat (tasa)'),
    ('delta_lat_menos_obs', 'Δ lat-obs'),
    ('rho_obs_CONTEOS_publicado', 'ρ obs (conteos, publicado)'),
    ('rho_poblacion_denuncias', 'ρ(pob, denuncias)'),
])}

Resumen total: el latente mejora al observado en `{total_better}/{total_defined}` scopes con resultado definido.

## Resultado por categoría, pooled

{md_table(cat_pooled, [
    ('categoria', 'categoría'),
    ('n', 'n'),
    ('rho_observado', 'ρ obs (tasa)'),
    ('rho_latente', 'ρ lat (tasa)'),
    ('delta_lat_menos_obs', 'Δ lat-obs'),
    ('rho_obs_CONTEOS_publicado', 'ρ obs (conteos, publicado)'),
])}

## Resultado por categoría y año

{md_table(cat_year, [
    ('scope', 'año'),
    ('categoria', 'categoría'),
    ('n', 'n'),
    ('rho_observado', 'ρ obs (tasa)'),
    ('rho_latente', 'ρ lat (tasa)'),
    ('delta_lat_menos_obs', 'Δ lat-obs'),
    ('rho_obs_CONTEOS_publicado', 'ρ obs (conteos, publicado)'),
])}

Resumen por categoría: el latente mejora al observado en `{cat_better}/{cat_defined}` celdas categoría-scope con resultado definido.

## Caveats

- La validación no es circular respecto de ENAPRES: SINADEF no depende de la decisión de denunciar ante la PNP. Ese es su valor como ancla externa.
- La geografía disponible es domicilio del fallecido, no ocurrencia del homicidio. Si hay desplazamiento residencia-ocurrencia, la correlación distrital se atenúa o se desplaza.
- Homicidio no forma parte de las categorías de-sesgadas (`robo_hurto_callejero`, `estafa`, `extorsion`, `secuestro`, `violencia_familiar_sexual`). La prueba evalúa validez convergente de la superficie total de riesgo, no equivalencia categoría-a-categoría.
- Aunque el enunciado del trabajo habla de Lima/Callao, el archivo latente validado aquí contiene 43 distritos con prefijo `1501`; Callao no entra porque no está en esa superficie.
- El pooled combina distrito-año y debe leerse como resumen descriptivo; no corrige dependencia serial por distrito.
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(content, encoding="utf-8")


def main() -> int:
    global LATENT_CSV, OUT_MD
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unit", choices=("victim", "incident"), default="victim",
        help="superficie que se valida; victim conserva el reporte canónico",
    )
    args = ap.parse_args()
    if args.unit == "incident":
        LATENT_CSV = LATENT_CSV.with_name("latente_distrito_categoria_anio_incident.csv")
        OUT_MD = OUT_MD.with_name("sinadef_validation_incident.md")
    lat, sin, meta = load_inputs()
    panel, districts, categories = build_panel(lat, sin)
    metrics = compute_metrics(panel, categories)
    write_report(metrics, panel, districts, categories, meta)
    pooled_total = metrics[(metrics["scope"].eq("pooled")) & (metrics["categoria"].eq("TOTAL"))].iloc[0]
    print(f"distritos validados: {len(districts)}")
    print(f"pooled total: rho_obs={pooled_total['rho_observado']:.3f} rho_lat={pooled_total['rho_latente']:.3f} delta={pooled_total['delta_lat_menos_obs']:.3f}")
    print(f"reporte: {OUT_MD}")

    # Cifras portantes de §6.3 del companion → registro canónico (solo la variante
    # víctima, que es la superficie del paper). El ancla SINADEF es un resultado NULO
    # (rate-vs-rate ~0 o negativa): se ancla justamente para que nadie lo "mejore" a
    # mano de vuelta al +0.34 confundido por población (slr-x50g B3, slr-x50g.6).
    if args.unit == "victim":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import canon
        inputs = [LATENT_CSV, SINADEF_CSV, POB_CSV]
        rob = metrics[(metrics["scope"].eq("pooled"))
                      & (metrics["categoria"].eq("robo_hurto_callejero"))].iloc[0]
        for fam_base, row in [("TOTAL", pooled_total), ("robo_hurto_callejero", rob)]:
            for side in ("observado", "latente"):
                fam = f"sinadef_rho_{'obs' if side == 'observado' else 'lat'}.{fam_base}"
                canon.emit(
                    fam, float(row[f"rho_{side}"]), variant="rate",
                    unit="Spearman tasa-homicidios-100k vs tasa-denuncias-100k, pooled distrital",
                    estimator="rate-vs-rate (la comparación correcta post-B3)",
                    inputs=inputs, script=__file__,
                )
                canon.emit(
                    fam, float(row[f"rho_{'obs' if side == 'observado' else 'lat'}_CONTEOS_publicado"]),
                    variant="counts",
                    unit="Spearman tasa-homicidios vs CONTEO de denuncias (confundido por población)",
                    estimator="tasa-vs-conteo, como se publicó originalmente (bug B3)",
                    inputs=inputs, script=__file__,
                )
        canon.emit(
            "sinadef_rho_pop_denuncias.TOTAL", float(pooled_total["rho_poblacion_denuncias"]),
            variant="counts",
            unit="Spearman población distrital vs conteo de denuncias (el confundidor de B3)",
            estimator="pooled distrital 2019–2024",
            inputs=inputs, script=__file__,
        )
        for side in ("obs", "lat"):
            canon.emit(
                f"sinadef_rho_countcount_{side}.TOTAL",
                float(pooled_total[f"rho_countcount_{side}"]), variant="counts",
                unit="Spearman conteo-homicidios vs conteo-denuncias (ilustra la huella de población)",
                estimator="count-vs-count, pooled distrito×año 2019–2024",
                inputs=inputs, script=__file__,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
