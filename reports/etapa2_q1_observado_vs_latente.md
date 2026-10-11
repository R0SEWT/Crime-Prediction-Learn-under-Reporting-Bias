# Q1 — Observado vs latente

**Anio principal:** 2023. **Oraculo:** denuncias con coordenada real agregadas a H3.

## Resultado principal

- Intra-distrital: observado macro rho=0.345, latente macro rho=0.345.
- Global contra oraculo geocodificado: observado macro rho=0.709, latente macro rho=0.700.
- Proporcionalidad H3 dentro de distrito/categoria/anio: 100.0% de grupos con factor constante en tolerancia 1e-05; max std=7.63e-06.

Interpretacion: el target latente cambia escala y composicion, pero la superficie H3 actual no agrega nueva senal intra-distrital porque observado y latente usan el mismo peso de redistribucion dentro de cada distrito.

## Macro 2023

| predictor   |   macro_intra_spearman |   macro_spearman_global |   macro_intra_recall10 |   macro_recall10 |   macro_ndcg |   macro_nmae |
|:------------|-----------------------:|------------------------:|-----------------------:|-----------------:|-------------:|-------------:|
| observado   |                  0.345 |                   0.709 |                  0.259 |            0.48  |        0.779 |        2.334 |
| latente     |                  0.345 |                   0.7   |                  0.259 |            0.433 |        0.775 |       71.393 |

## Detalle por categoria 2023

| predictor   | crime_cat                 |   intra_spearman |   spearman_global |   intra_recall10 |   recall10 |   ndcg |    nmae |
|:------------|:--------------------------|-----------------:|------------------:|-----------------:|-----------:|-------:|--------:|
| observado   | robo_hurto_callejero      |            0.457 |             0.853 |            0.246 |      0.451 |  0.814 |   2.137 |
| observado   | extorsion                 |            0.388 |             0.71  |            0.249 |      0.459 |  0.83  |   3.888 |
| observado   | estafa                    |            0.357 |             0.762 |            0.261 |      0.504 |  0.784 |   2.716 |
| observado   | violencia_familiar_sexual |            0.409 |             0.843 |            0.241 |      0.444 |  0.814 |   1.18  |
| observado   | secuestro                 |            0.113 |             0.373 |            0.299 |      0.54  |  0.653 |   1.751 |
| latente     | robo_hurto_callejero      |            0.457 |             0.853 |            0.246 |      0.457 |  0.814 |  15.603 |
| latente     | extorsion                 |            0.388 |             0.695 |            0.249 |      0.448 |  0.85  |  31.725 |
| latente     | estafa                    |            0.357 |             0.755 |            0.261 |      0.448 |  0.773 | 276.764 |
| latente     | violencia_familiar_sexual |            0.409 |             0.842 |            0.241 |      0.424 |  0.811 |  20.162 |
| latente     | secuestro                 |            0.113 |             0.354 |            0.299 |      0.387 |  0.628 |  12.71  |

## Composicion H3 2023

| crime_cat                 |   share_observado |   share_latente |   delta_share_latente_minus_obs |
|:--------------------------|------------------:|----------------:|--------------------------------:|
| robo_hurto_callejero      |             0.636 |           0.333 |                          -0.302 |
| extorsion                 |             0.029 |           0.019 |                          -0.01  |
| estafa                    |             0.046 |           0.338 |                           0.292 |
| violencia_familiar_sexual |             0.288 |           0.308 |                           0.02  |
| secuestro                 |             0.001 |           0.001 |                          -0     |

## Validacion Etapa 1 contra ENAPRES

- Composicion L1 vs ENAPRES: observado=0.762, latente=0.303.
- Ranking distrital total sin estafa: observado rho=0.965, latente rho=0.961.

Lectura: el de-sesgo mejora claramente composicion/magnitud, pero no mejora el ranking espacial distrital.

## Veredicto Q1

El target latente si sirve para la pregunta de crimen no reportado: corrige magnitud y composicion frente al observado. No sirve, tal como esta construido a H3, para demostrar resolucion fina intra-distrital. Para una STGNN, esto implica que entrenar sobre la superficie latente poblacional-distribuida no prueba aprendizaje de riesgo latente intra-H3; solo prueba aprendizaje del prior de redistribucion.

Metricas completas: `data/silver/predictions/q1_observed_vs_latent_metrics.csv`.