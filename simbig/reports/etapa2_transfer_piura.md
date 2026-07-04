# Transferencia espacial Lima → Piura (slr-211/slr-vkj)

**Protocolo:** modelo HGB (params del ladder) con el bloque tabular TRANSFERIBLE (18 features: WorldPop, manzana censal INEI, OSM sin features Lima-específicas). Train Lima 2019-2022 → test **Piura 2023** vs oráculo geocodificado propio. Spearman intra-distrital macro (distritos con ≥5 celdas).

| categoría | transfer (Lima→AQP) | local (AQP→AQP) | persistencia | prior pob. |
|:--|--:|--:|--:|--:|
| robo_hurto_callejero | 0.593 | 0.740 | 0.758 | 0.688 |
| extorsion | 0.545 | 0.430 | 0.470 | 0.515 |
| estafa | 0.615 | 0.576 | 0.546 | 0.583 |
| violencia_familiar_sexual | 0.728 | 0.824 | 0.733 | 0.704 |
| secuestro | 0.270 | 0.227 | 0.220 | 0.248 |
| **MACRO** | **0.550** | **0.559** | **0.545** | **0.548** |

Distritos evaluables: 6. Celdas test: 1056.

## Deltas pareados (bootstrap sobre distritos, B=2000)

| comparación | Δ macro ρ [CI 95%] | ¿CI excluye 0? |
|:--|:--|:--|
| transfer − prior pob. | +0.0022 [-0.0363, +0.0344] | no |
| transfer − persistencia | +0.0123 [-0.0372, +0.0665] | no |
| local − transfer | +0.0095 [-0.0218, +0.0451] | no |

Referencias Lima (mismo protocolo): bloque completo M4c 0.464; prior histórico 0.394; per cápita 0.349.
