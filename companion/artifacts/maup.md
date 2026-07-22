# Sensibilidad MAUP del de-sesgo — restringido a robo (bead slr-x50g.8)

_¿Las conclusiones de la corrección (construida a nivel distrito) sobreviven críticas de unidad areal modificable (MAUP)? Método: re-zonación sistemática (escala + zonación aleatoria), no una agrupación administrativa única. Benchmark = victimización ENAPRES **anualizada** (víctimas / nº de olas por categoría) sobre las **4 categorías comparables** (excl. violencia familiar, §3.1). La versión V10 sumaba olas desiguales e incluía violencia familiar → titular falso L1 0.99→0.11 (B1). Regenera: `python3 scripts/maup_sensitivity.py`. JSON: `data/silver/.../maup_sensitivity.json`._

## 1. Composición por escala — el de-sesgo la ALEJA del benchmark

Para cada escala k (nº de unidades; k=43 distritos → k=1 Lima total) se remuestrean **particiones aleatorias** de los 43 distritos y se mide la distancia L1 media (ponderada por víctimas) entre la composición de las 4 categorías comparables de cada unidad y la de la victimización anualizada. `ρ_rob` = Spearman zona-nivel del riesgo de **robo** (observado/latente) vs victimización, la métrica espacial libre de mezcla de olas:

| k (unidades) | reps | L1 observado | L1 latente | latente mejor | ρ_rob(obs,V) | ρ_rob(lat,V) |
|---|---|---|---|---|---|---|
| 43 | 1 | 0.328 | 0.625 | 0% | 0.94 | 0.95 |
| 24 | 200 | 0.322 ±0.00 | 0.648 ±0.01 | 0% | 0.91 | 0.94 |
| 12 | 200 | 0.317 ±0.00 | 0.667 ±0.01 | 0% | 0.87 | 0.91 |
| 6 | 200 | 0.315 ±0.00 | 0.676 ±0.01 | 0% | — | — |
| 3 | 200 | 0.313 ±0.00 | 0.682 ±0.00 | 0% | — | — |
| 1 | 1 | 0.312 | 0.686 | 0% | — | — |

**Hallazgo:** el de-sesgo **aleja** la composición del benchmark de encuesta a TODA escala y en ~todas las zonaciones (columna *latente mejor* ≈ 0%): el L1 latente supera al observado en cada corte. El resultado negativo de composición (§5.4) **no es un artefacto de la partición distrital** — persiste al re-zonar y al cambiar de escala. Es la confirmación MAUP de por qué la superficie validada se restringe a robo.

## 2. Magnitud — invariante a la partición

El factor de sub-representación 1/r̂ se estima por **categoría** (no por unidad espacial; r̂ es categoría×año), así que es un **cociente de sumas** e idéntico bajo cualquier zonación o escala — sin necesidad de simulación:

| categoría | factor 1/r̂ |
|---|---|
| estafa | ×76.5 |
| violencia_familiar_sexual | ×11.1 |
| secuestro | ×8.4 |
| extorsion | ×6.9 |
| robo_hurto_callejero | ×4.6 |

## 3. Ranking espacial de robo — estable bajo agregación

Para robo, ρ_rob(obs,V) ≈ ρ_rob(lat,V) a cada escala (columnas de la tabla §1): el de-sesgo **no reordena** el mapa de robo al cambiar la unidad — el valor de la corrección está en la **magnitud**, no en un reordenamiento espacial. MAUP afecta el valor absoluto de ρ (baja al desagregar, como es de esperar), pero no la conclusión. Métrica de robo solo → sin la mezcla de olas que contamina un total multi-categoría.

## 4. Límite ecológico (declarado)

La invariancia de magnitud a la partición se debe a que el de-sesgo opera a nivel **categoría**, no espacial. El reverso es el **límite ecológico**: `r` es una tasa de reporte por unidad (distrito/zona), no por individuo. Aplicarla uniformemente dentro de la unidad **no recupera** la heterogeneidad individual del reporte (dos víctimas del mismo distrito con distinta propensión a denunciar). El latente es un **target areal de-sesgado**, no una imputación a nivel persona; la variación intra-unidad es trabajo de la Etapa 2 (resolución fina) y queda fuera del alcance identificable aquí. Esto se reporta como límite, no se sobre-vende.
