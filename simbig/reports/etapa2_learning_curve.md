# Curva de aprendizaje local vs transferencia — Arequipa (slr-b4b)

**Pregunta:** ¿cuántos años de historia local necesita Arequipa para superar al modelo transferido de Lima (entrenado 2019-2022, sin datos locales)? Test 2023, Spearman intra-distrital macro, bootstrap pareado sobre distritos (B=2000).

Transfer fijo (Lima→Arequipa): **0.509**

| historia local | macro ρ | Δ (local − transfer) [CI 95%] | ¿CI excluye 0? |
|:--|--:|:--|:--|
| 1 año (2022) | 0.524 | +0.0146 [-0.0322, +0.0602] | no |
| 2 años (2021-22) | 0.552 | +0.0428 [+0.0028, +0.0824] | SÍ |
| 3 años (2020-22) | 0.558 | +0.0491 [+0.0176, +0.0841] | SÍ |
| 4 años (2019-22) | 0.562 | +0.0528 [+0.0164, +0.0875] | SÍ |

## Por categoría (transfer vs local 4 años)

| categoría | transfer | local 4 años |
|:--|--:|--:|
| robo_hurto_callejero | 0.580 | 0.713 |
| extorsion | 0.436 | 0.425 |
| estafa | 0.552 | 0.621 |
| violencia_familiar_sexual | 0.553 | 0.685 |
| secuestro | 0.426 | 0.366 |
