# Transferencia espacial Lima → Cusco (slr-211/slr-vkj)

**Protocolo:** modelo HGB (params del ladder) con el bloque tabular TRANSFERIBLE (18 features: WorldPop, manzana censal INEI, OSM sin features Lima-específicas). Train Lima 2019-2022 → test **Cusco 2023** vs oráculo geocodificado propio. Spearman intra-distrital macro (distritos con ≥5 celdas).

| categoría | transfer (Lima→AQP) | local (AQP→AQP) | persistencia | prior pob. |
|:--|--:|--:|--:|--:|
| robo_hurto_callejero | 0.590 | 0.785 | 0.781 | 0.680 |
| extorsion | 0.327 | 0.526 | 0.451 | 0.352 |
| estafa | 0.523 | 0.623 | 0.598 | 0.541 |
| violencia_familiar_sexual | 0.588 | 0.751 | 0.771 | 0.640 |
| secuestro | 0.374 | 0.334 | 0.224 | 0.365 |
| **MACRO** | **0.480** | **0.604** | **0.565** | **0.516** |

Distritos evaluables: 8. Celdas test: 868.

## Deltas pareados (bootstrap sobre distritos, B=2000)

| comparación | Δ macro ρ [CI 95%] | ¿CI excluye 0? |
|:--|:--|:--|
| transfer − prior pob. | -0.0352 [-0.0853, +0.0104] | no |
| transfer − persistencia | -0.0865 [-0.1834, +0.0044] | no |
| local − transfer | +0.1233 [+0.0523, +0.2106] | SÍ |

Referencias Lima (mismo protocolo): bloque completo M4c 0.464; prior histórico 0.394; per cápita 0.349.
