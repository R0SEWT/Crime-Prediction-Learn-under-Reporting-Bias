# Robustez de métricas intra-distritales (slr-0vh)

**Protocolo:** train 2019-2022, test 2023, oráculo geocodificado. Bootstrap B=2000 sobre **distritos** (la unidad de réplica de la métrica intra), percentiles 2.5/97.5. HGB con params default del ladder.

## 1. CI por escalón (macro sobre categorías)

| escalón | macro ρ [CI 95%] | captura@10% [CI 95%] |
|:--|:--|:--|
| M1_demografia | 0.380 [0.341, 0.417] | 0.276 [0.242, 0.314] |
| M1b_manzana | 0.455 [0.418, 0.489] | 0.342 [0.304, 0.381] |
| M2_osm | 0.459 [0.423, 0.492] | 0.339 [0.303, 0.375] |
| M4c_transporte | 0.464 [0.428, 0.494] | 0.326 [0.292, 0.359] |

## 2. Deltas PAREADOS entre escalones (¿significativos?)

| comparación | Δ macro ρ [CI 95%] | ¿CI excluye 0? |
|:--|:--|:--|
| M1b_manzana − M1_demografia | +0.0749 [+0.0471, +0.1023] | SÍ |
| M2_osm − M1b_manzana | +0.0040 [-0.0178, +0.0265] | no |
| M4c_transporte − M2_osm | +0.0049 [-0.0122, +0.0225] | no |
| M4c_transporte − M1b_manzana | +0.0089 [-0.0096, +0.0288] | no |

## 3. Por categoría (M4c): ρ vs captura@10% con CI

| categoría | ρ [CI] | captura@10% [CI] | n distritos |
|:--|:--|:--|:--|
| robo_hurto_callejero | 0.593 [0.530, 0.657] | 0.372 [0.302, 0.458] | 41 |
| extorsion | 0.401 [0.322, 0.476] | 0.281 [0.218, 0.348] | 40 |
| estafa | 0.489 [0.404, 0.565] | 0.321 [0.249, 0.399] | 39 |
| violencia_familiar_sexual | 0.595 [0.535, 0.653] | 0.372 [0.301, 0.451] | 41 |
| secuestro | 0.240 [0.158, 0.321] | 0.282 [0.184, 0.386] | 34 |

## 4. Sensibilidad al sesgo del oráculo (M4c)

Split por mediana de `real_coord_ratio` distrital (0.798).

| mitad | macro ρ [CI 95%] | n distritos |
|:--|:--|:--|
| ALTA geocodificación | 0.477 [0.425, 0.525] | 19 |
| BAJA geocodificación | 0.466 [0.422, 0.509] | 19 |

## 5. Variante per cápita (M4c)

ρ intra de pred/pop vs target/pop: **0.349** [0.316, 0.384] (conteos: 0.464). Si cae mucho, parte del ranking
de conteos era volumen poblacional, no riesgo.

## Veredicto (lectura)

1. **El ladder tiene UN escalón real y el resto es ruido.** El único delta
   significativo es INEI manzana sobre demografía (+0.075 [+0.047, +0.102]).
   OSM, salud, vigilancia y transporte — 9 fuentes integradas después — son
   individualmente y en conjunto indistinguibles de 0 (+0.009 [−0.010, +0.029]
   acumulado). El claim de saturación del techo pasa de descriptivo a formal.
2. **Métrica primaria recomendada por régimen:** categorías densas → ρ intra
   (CIs estrechos, robo/violencia ~0.59); raras → captura@10% como co-primaria
   (secuestro ρ=0.24 [0.16, 0.32] no toca 0, pero la captura es más estable y
   más interpretable operativamente).
3. **El sesgo del oráculo no explica los resultados:** alta vs baja
   geocodificación distrital 0.477 vs 0.466, CIs solapados.
4. **Conteos ≠ riesgo:** per cápita 0.349 vs conteos 0.464. La tesis debe
   declarar qué claim afirma: localizar *volumen* (0.46) o *riesgo per cápita*
   (0.35). Ambos defendibles; confundirlos no.

(Tablas regenerables con `python3 scripts/eval_metric_robustness.py`; esta
sección es lectura del análisis y se re-agrega a mano si se regenera.)
