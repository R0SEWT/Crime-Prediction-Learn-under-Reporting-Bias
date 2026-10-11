# Interpretabilidad de la atencion cross-modal (slr-8s3)

**Protocolo:** STGNN `target=hybrid`, `fusion=attention`, seeds [42, 7, 123], 150 epocas. La atencion se captura en `eval()` con `need_weights=True`, `average_attn_weights=True`; por tanto cada fila query suma 1 y no esta distorsionada por dropout.

**Celdas H3:** 4172. **Oraculo espacial:** suma 2019-2022 de `data/silver/h3_features/h3_observed_geocoded.parquet`.

## Peso medio recibido por modalidad

| modality   | metric          |   mean |    std |
|:-----------|:----------------|-------:|-------:|
| SV         | received_weight | 0.2666 | 0.0697 |
| estatico   | received_weight | 0.5297 | 0.0785 |
| temporal   | received_weight | 0.2037 | 0.0301 |

## Correlacion espacial con densidad observada por categoria

| crime_cat                 | modality   | metric                           |    mean |    std |
|:--------------------------|:-----------|:---------------------------------|--------:|-------:|
| estafa                    | SV         | spearman_weight_vs_obs_2019_2022 |  0.0334 | 0.4657 |
| estafa                    | estatico   | spearman_weight_vs_obs_2019_2022 |  0.329  | 0.4096 |
| estafa                    | temporal   | spearman_weight_vs_obs_2019_2022 | -0.5041 | 0.0964 |
| extorsion                 | SV         | spearman_weight_vs_obs_2019_2022 |  0.0154 | 0.4252 |
| extorsion                 | estatico   | spearman_weight_vs_obs_2019_2022 |  0.2846 | 0.358  |
| extorsion                 | temporal   | spearman_weight_vs_obs_2019_2022 | -0.4336 | 0.0963 |
| robo_hurto_callejero      | SV         | spearman_weight_vs_obs_2019_2022 |  0.1135 | 0.4752 |
| robo_hurto_callejero      | estatico   | spearman_weight_vs_obs_2019_2022 |  0.3811 | 0.5016 |
| robo_hurto_callejero      | temporal   | spearman_weight_vs_obs_2019_2022 | -0.617  | 0.0694 |
| secuestro                 | SV         | spearman_weight_vs_obs_2019_2022 | -0.0022 | 0.3438 |
| secuestro                 | estatico   | spearman_weight_vs_obs_2019_2022 |  0.2074 | 0.2717 |
| secuestro                 | temporal   | spearman_weight_vs_obs_2019_2022 | -0.3136 | 0.0855 |
| violencia_familiar_sexual | SV         | spearman_weight_vs_obs_2019_2022 |  0.1287 | 0.461  |
| violencia_familiar_sexual | estatico   | spearman_weight_vs_obs_2019_2022 |  0.3888 | 0.5081 |
| violencia_familiar_sexual | temporal   | spearman_weight_vs_obs_2019_2022 | -0.635  | 0.0655 |

## Distritos con alto vs bajo conteo de secuestro

| modality   | group           | metric       |    mean |    std |
|:-----------|:----------------|:-------------|--------:|-------:|
| SV         | alto_menos_bajo | delta_weight |  0.0026 | 0.0233 |
| SV         | alto_secuestro  | mean_weight  |  0.2683 | 0.0622 |
| SV         | bajo_secuestro  | mean_weight  |  0.2657 | 0.0741 |
| estatico   | alto_menos_bajo | delta_weight |  0.0506 | 0.0265 |
| estatico   | alto_secuestro  | mean_weight  |  0.5642 | 0.0616 |
| estatico   | bajo_secuestro  | mean_weight  |  0.5136 | 0.0865 |
| temporal   | alto_menos_bajo | delta_weight | -0.0532 | 0.0074 |
| temporal   | alto_secuestro  | mean_weight  |  0.1675 | 0.0314 |
| temporal   | bajo_secuestro  | mean_weight  |  0.2206 | 0.0297 |

## Contraste Street View disponible vs ausente

| modality   | group          | metric       |    mean |    std |
|:-----------|:---------------|:-------------|--------:|-------:|
| SV         | sv1_menos_sv0  | delta_weight |  0.0144 | 0.0854 |
| SV         | sv_available_0 | mean_weight  |  0.2639 | 0.0829 |
| SV         | sv_available_1 | mean_weight  |  0.2783 | 0.0418 |
| estatico   | sv1_menos_sv0  | delta_weight |  0.0623 | 0.1012 |
| estatico   | sv_available_0 | mean_weight  |  0.5184 | 0.0968 |
| estatico   | sv_available_1 | mean_weight  |  0.5807 | 0.0079 |
| temporal   | sv1_menos_sv0  | delta_weight | -0.0767 | 0.0201 |
| temporal   | sv_available_0 | mean_weight  |  0.2176 | 0.0302 |
| temporal   | sv_available_1 | mean_weight  |  0.1409 | 0.0349 |

## Veredicto

**NO-GO.** hay senal descriptiva, pero no es lo bastante nitida y estable para sostener un mecanismo en el manuscrito. Evidencia confirmada: la modalidad estatica recibe el mayor peso medio en las tres semillas y el peso temporal cae en distritos con alto conteo de secuestro. Evidencia contra un claim fuerte: el contraste `sv_available=1` vs `0` para el peso SV cambia de signo entre semillas, y las correlaciones espaciales de SV/estatico con el oraculo tambien cambian de signo. Inferencia: usar, como maximo, una nota/apendice de diagnostico; no presentarlo como mecanismo principal de rescate de categorias raras.

Datos tabulares: `data/silver/predictions/attention_weights_summary.csv`.
