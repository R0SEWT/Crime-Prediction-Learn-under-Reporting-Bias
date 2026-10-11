# Techo tabular: intentos de afinamiento (slr-cdo)

**Fecha:** 2026-07-03 · Métrica: Spearman intra-distrital macro vs oráculo geocodificado,
train 2019-2022 / test 2023 (mismo protocolo del ladder Q4).

Dos palancas probadas para subir el techo M4c = 0.4636:

## 1. Historial geocodificado como feature (M7) — NEGATIVO

| step | features añadidas | macro ρ |
|:--|:--|--:|
| M4c_transporte (ref) | — | **0.4636** |
| M7a_persist | + conteo geocodificado t-1 | 0.460 |
| M7b_historial_esp | + lag espacial k-ring1 + share intra-distrital t-1 | 0.454 |

**Lectura:** a cadencia anual, la forma urbana ya *subsume* la persistencia:
dónde se concentró el crimen el año pasado es predecible desde la estructura
(vías, NBI de manzana, POIs), así que dárselo explícito solo añade ruido con
4 años de train. Consistente con que la persistencia sola (M5 = 0.364,
prior_historical = 0.394) pierda contra features. Implementado en
`eval_modality_ladder.py` (steps M7a/M7b, columnas construidas en `load_panel`).

## 2. Sweep de hiperparámetros HGB — +0.005

Grid 18 configs (max_iter × max_leaf_nodes × learning_rate), **selección por
categoría en validación temporal 2022** (train 2019-2021), evaluación final
única en test 2023. Sin contaminación del test.

| categoría | M4c default | tuned | Δ | config ganadora (iter, hojas, lr) |
|:--|--:|--:|--:|:--|
| robo_hurto_callejero | 0.5928 | 0.5945 | +0.002 | ≈default |
| extorsión | 0.4008 | 0.4008 | 0.000 | default |
| estafa | 0.4893 | 0.4897 | +0.000 | (140, 15, 0.03) |
| violencia_familiar_sexual | 0.5948 | 0.6073 | +0.013 | (140, 63, 0.03) |
| secuestro | 0.2402 | 0.2507 | +0.011 | (140, 31, 0.06) |
| **MACRO** | **0.4636** | **0.4686** | **+0.005** | — |

## Veredicto

El techo tabular está **saturado en ~0.464–0.469** para este target/split:
ni el historial como feature ni el tuning lo mueven materialmente. La ganancia
del tuning (+0.005) se concentra en las categorías con menos señal (violencia
profunda, secuestro lr moderado) y no cambia ningún argumento: el orden
sigue siendo DL híbrido 0.276 < prior 0.394 < tabular ~0.47. Lo que queda por
encima de 0.47 no es modelo — es cadencia (mensual ya probado, marginal),
cobertura del oráculo (~31% coords reales) y generalización espacial.

Reproducir: `python3 scripts/tune_tabular_ceiling.py` (reusa `load_panel`/
`evaluate_prediction` del ladder); M7 integrado en `scripts/eval_modality_ladder.py`.
